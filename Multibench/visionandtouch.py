import h5py
import numpy as np
from typing import Tuple
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import ImageFilter
from typing import List, Dict, Union, Callable, Any
from torchvision.transforms import (RandomApply, Compose, RandomChoice,
                                    RandomGrayscale, RandomResizedCrop,
                                    ColorJitter, RandomHorizontalFlip,
                                    ToTensor, Normalize)
from pytorch_lightning import LightningDataModule
from collections.abc import Iterable


class MultiBenchDataModule(LightningDataModule):
    """VISION&TOUCH (vision, force, proprioception) with n=117k (train) + 29k (test)
           Shape: vision=(*, 3, 128, 128), force=(*, T, 6) [after truncation], proprio=(*, 8),
           label=(*, 4) if task=="ee_yaw_next" (continuous) or (*,) if task=="contact_next" (binary)

        Sequence T varies for each sample in each dataset, up to 50 (max padding length).
        Each batch of data consists of pairs (X, y) where X is a tuple/list of `n` modalities and y is a Tensor.
        Each modality is a Tensor of shape (*, T, p) or (*, p) depending on the modality (sequence or tabular data)
    """
    def __init__(self, model: str,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 **kwargs):
        """
        Args:
            model: in {'Sup', 'FactorCL'}
                The model defines the augmentations to apply:
                    - Sup: no augmentation, returns the modalities + label
                    - FactorCL: augmentation + original modality
            batch_size: Batch size given to dataloader (train, val, test)
            num_workers: Number of CPU workers for data loading
            kwargs: keyword args given to the torch `Dataset`
        """
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = kwargs
        root="/fastdata/visionandtouch/triangle_real_data"
        if self.model == "Sup":
            self.train_dataset = MultimodalManipulationDataset(
                root,
                split="train",
                transform=transforms.Compose(
                    [
                        ProcessForce(32, "force", tanh=True),
                        ProcessImage(128)
                    ]),
                **kwargs)
            self.val_dataset = MultimodalManipulationDataset(
                root,
                split="val",
                transform=transforms.Compose(
                    [
                        ProcessForce(32, "force", tanh=True),
                        ProcessImage(128)
                    ]),
                **kwargs)
            self.test_dataset = MultimodalManipulationDataset(
                root,
                split="test",
                transform=transforms.Compose(
                    [
                        ProcessForce(32, "force", tanh=True),
                        ProcessImage(128)
                    ]),
                **kwargs)
        elif self.model == "FactorCL":
            self.augment = MultiBenchAugmentations("simclr")
            self.augment_unique = MultiBenchAugmentations("noise+drop")
            self.train_dataset = BimodalManipulationFactorCL(
                root,
                split="train",
                transform=transforms.Compose(
                    [
                        ProcessForce(32, "force", tanh=True),
                        ProcessImage(128)
                    ]),
                augment=self.augment,
                augment_unique=self.augment_unique,
                **kwargs)
            self.val_dataset = BimodalManipulationFactorCL(
                root,
                split="val",
                transform=transforms.Compose(
                    [
                        ProcessForce(32, "force", tanh=True),
                        ProcessImage(128)
                    ]),
                augment=self.augment,
                augment_unique=self.augment_unique,
                **kwargs)
            self.test_dataset = BimodalManipulationFactorCL(
                root,
                split="test",
                transform=transforms.Compose(
                    [
                        ProcessForce(32, "force", tanh=True),
                        ProcessImage(128)
                    ]),
                augment=self.augment,
                augment_unique=self.augment_unique,
                **kwargs)
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)


class MultimodalManipulationDataset(Dataset):
    """Multimodal Manipulation dataset [1], adapted from
    https://github.com/stanford-iprl-lab/multimodal_representation/tree/master/multimodal/dataloaders


    [1] Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations
        for Contact-Rich Tasks, Lee, Zhu et al., ICRA 2019

    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        task: str = "ee_yaw_next",
        modalities: Tuple[str] = ("image", "force"),
        test_ratio: float = 0.2,
        transform=None,
        episode_length=50,
        n_time_steps=1,
        action_dim=4
    ):
        """Initialize dataset.

        Args:
            root (str):
            split (str): "train", "val" or "test"
            task (str): "ee_yaw_next" for end-effector position prediction task (regression) or
                "contact_next" for next contact prediction (binary classifications)
            modalities: Tuple of modalities to return, in {"image", "force", "proprio"}
            test_ratio (float): test split ratio for train/test partition
            transform (fn, optional): Optional function to transform data. Defaults to None.
            episode_length (int, optional): Length of each episode. Defaults to 50.
            n_time_steps (int, optional): Number of time steps. Defaults to 1.
            action_dim (int, optional): Action dimension. Defaults to 4.
        """
        assert task in {"ee_yaw_next", "contact_next"}, f"Unknown task: {task}"

        self.root = root
        self.split = split
        self.task = task
        self.modalities = modalities
        self.test_ratio = test_ratio
        self.dataset_path = self._get_filenames(seed=42)
        self.transform = transform
        self.episode_length = episode_length
        self.n_time_steps = n_time_steps
        self.dataset = {}
        self.action_dim = action_dim

    def _get_filenames(self, seed: int):
        """Get the filenames by split (reproducible with a fixed seed)"""
        filename_list = []
        for file in sorted(os.listdir(self.root)):
            if file.endswith(".h5"):
                filename_list.append(os.path.join(self.root, file))
        filename_list = np.array(filename_list)
        rng = np.random.default_rng(seed)
        idx = np.arange(len(filename_list))
        rng.shuffle(idx)
        n_test = int(self.test_ratio * len(filename_list))
        n_train = len(filename_list) - n_test
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        if self.split == "train":
            return filename_list[train_idx]
        elif self.split in ["val", "test"]:
            return filename_list[test_idx]
        raise ValueError(f"Unknown split: {self.split}")

    def __len__(self):
        """Get number of items in dataset."""
        return len(self.dataset_path) * (self.episode_length - self.n_time_steps)

    def __getitem__(self, idx):
        """Get item in dataset at index idx."""
        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)

        if dataset_index >= self.episode_length - self.n_time_steps - 1:
            dataset_index = np.random.randint(
                self.episode_length - self.n_time_steps - 1
            )

        sample = self._get_single(
            self.dataset_path[list_index],
            dataset_index
        )

        # Filter modalities and returns required target
        X, y = [], None
        for m in self.modalities:
            assert m in sample, f"Unknown modality: {m}"
            X.append(sample[m])
        y = sample[self.task]

        return X, y

    def _get_single(
        self, dataset_name, dataset_index
    ):

        with h5py.File(dataset_name, "r", swmr=True, libver="latest") as dataset:

            image = dataset["image"][dataset_index]
            depth = dataset["depth_data"][dataset_index]
            proprio = dataset["proprio"][dataset_index][:8]
            force = dataset["ee_forces_continuous"][dataset_index]

            if image.shape[0] == 3:
                image = np.transpose(image, (2, 1, 0))

            if depth.ndim == 2:
                depth = depth.reshape((128, 128, 1))

            flow = np.array(dataset["optical_flow"][dataset_index])
            flow_mask = np.expand_dims(
                np.where(
                    flow.sum(axis=2) == 0,
                    np.zeros_like(flow.sum(axis=2)),
                    np.ones_like(flow.sum(axis=2)),
                ),
                2,
            )

            sample = {
                "image": Image.fromarray(image),
                "depth": depth,
                "flow": flow,
                "flow_mask": flow_mask,
                "action": dataset["action"][dataset_index + 1],
                "force": force.astype(np.float32),
                "proprio": proprio,
                "ee_yaw_next": dataset["proprio"][dataset_index + 1][:self.action_dim],
                "contact_next": np.array(
                    dataset["contact"][dataset_index + 1].sum() > 0
                ).astype(int)
            }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(n={len(self)}, split={self.split}, task={self.task})"


class BimodalManipulationFactorCL(MultimodalManipulationDataset):

    def __init__(self, *args, augment, augment_unique, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment
        self.augment_unique = augment_unique
        if self.transform is not None: # small hack
            self.img_transform = self.transform.transforms.pop(-1)

    def __getitem__(self, i):
        (x1, x2), _ = super().__getitem__(i)
        aug1 = self.augment([x1])[0]
        aug2 = self.augment_unique([x2])[0]
        if self.img_transform is not None: # small hack
            self.transform.transforms.append(self.img_transform)
            (x1, x2), _ = super().__getitem__(i)
            self.img_transform = self.transform.transforms.pop(-1)
        return x1, x2, aug1, aug2


class ProcessForce(object):
    """Truncate a time series of force readings with a window size.
    Args:
        window_size (int): Length of the history window that is
            used to truncate the force readings
    """

    def __init__(self, window_size, key='force', tanh=False):
        """Initialize ProcessForce object.

        Args:
            window_size (int): Windows size
            key (str, optional): Key where data is stored. Defaults to 'force'.
            tanh (bool, optional): Whether to apply tanh to output or not. Defaults to False.
        """
        assert isinstance(window_size, int)
        self.window_size = window_size
        self.key = key
        self.tanh = tanh

    def __call__(self, sample):
        """Get data from sample."""
        force = sample[self.key]
        force = force[-self.window_size:]
        if self.tanh:
            force = np.tanh(force)  # remove very large force readings
        sample[self.key] = force
        return sample


class ProcessImage(object):
    """
        Transform numpy image (HWC format) to torch tensor (CHW format).
    """

    def __init__(self, size=128):
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, sample):
        new_dict = dict()
        for k, v in sample.items():
            if k == "image":
                new_dict[k] = self.tf(v)
            else:
                new_dict[k] = v
        return new_dict


class AugMapper:
    """ Map a list of modalities X to a list of
        augmented modalities X' through a list of transformations T
        such that X' = [T[0](X[0)), ..., T[n](X[n])]
    """

    def __init__(self, tfs: List[Callable]):
        self.transforms = tfs

    def __call__(self, x: List[Any]):
        assert len(x) == len(self.transforms), "Number of modalities must match number of transformations"
        return [tf([x[i]])[0] for i, tf in enumerate(self.transforms)]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class SimCLRAug:
    """Apply SimCLR augmentation to an input image."""
    def __init__(self, size=224):
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        self.aug = Compose([
            RandomResizedCrop(size, scale=(0.08, 1.)),
            RandomApply([
                ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            RandomGrayscale(p=0.2),
            RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])

    def __call__(self, list_x):
        x_tf = []
        for x in list_x:
            x_tf.append(self.aug(x))
        return x_tf


class MultiBenchAugmentations(torch.nn.Module):
    """ Defines a set of augmentations to apply to time-series/tabular data based on [1] and
        to images based on [2]
    [1] Factorized Contrastive Learning: Going Beyond Multi-view Redundancy, Liang et al., NeurIPS 2023
    [2] A Simple Framework for Contrastive Learning of Visual Representations, Chen et al., ICML 2020
    """

    def __init__(self, augmentations: Union[str, List[str]] = None,
                 p: float = 0.5,
                 random_choice: bool = False):
        """
        :param p: probability for applying each augmentation individually if `random_choice` is false
        :param augmentations: transformations to apply either:
            - individually to each modality (if List) => unimodal aug
            - together on all modalities (if str) => multi-modal aug
        :param random_choice: randomly choose one transformation to apply
        """
        super().__init__()
        if augmentations is None:
            augmentations = []
        assert isinstance(augmentations, str) or isinstance(augmentations, Iterable), \
            f"Unknown type: {type(augmentations)}"
        if isinstance(augmentations, str):
            augmentations = [augmentations]
        transforms = []
        for augmentations_ in augmentations:
            augmentations_ = augmentations_.split("+")
            tf = []
            for aug in augmentations_:
                if aug == "permute":
                    tf.append(self.permute)
                elif aug == "noise":
                    tf.append(self.noise)
                elif aug in ["drop", "multi_drop"]:
                    tf.append(lambda x: self.drop(x, multimodal=(aug=="multi_drop")))
                elif aug in ["drop_consecutive", "multi_drop_consecutive"]:
                    tf.append(lambda x: self.drop_consecutive(x, multimodal=(aug=="multi_drop_consecutive")))
                elif aug in ["crop", "multi_crop"]:
                    tf.append(lambda x: self.crop(x, multimodal=(aug=="multi_crop")))
                elif aug == "mixup":
                    tf.append(self.mixup)
                elif aug == "simclr":
                    tf.append(SimCLRAug(size=128))
                else:
                    raise ValueError(f"Unknown augmentation: {aug}")

            if random_choice:
                transforms.append(RandomChoice(tf))
            else:
                transforms.append(Compose([RandomApply([tf_], p=p)
                                           if not isinstance(tf_, SimCLRAug) else tf_ for tf_ in tf]))

        if len(transforms) == 1:
            self.transforms = transforms[0]
        elif len(transforms) > 1:
            self.transforms = AugMapper(transforms) # map each tf to a modality
        else:
            self.transforms = lambda x: x

    @staticmethod
    def parse_kwargs(kwargs: Dict):
        """Parse and return keywords arguments relevant for this class.
        It is remove in-place from input `kwargs`"""
        aug_kwargs = {}
        if "augmentations" in kwargs:
            aug_kwargs["augmentations"] = kwargs.pop("augmentations")
        if "random_choice" in kwargs:
            aug_kwargs["random_choice"] = kwargs.pop("random_choice")
        return aug_kwargs

    def forward(self, x):
        # x: List[np.ndarray]
        #  List of arrays (one per modality) of shape (T, p)
        #  where `T`==seq length and `p`==num features
        return self.transforms(x)

    @staticmethod
    def permute(x, multimodal=True):
        if len(x) > 0:
            # shuffle the sequence order
            if multimodal:
                idx = np.random.permutation(x[0].shape[0])
                return [x_[idx] for x_ in x]
            x = [x_[np.random.permutation(x_.shape[0])] for x_ in x]
        return x

    @staticmethod
    def noise(x, std=0.1):
        return [x_ + np.random.randn(*x_.shape).astype(np.float32) * std for x_ in x]

    @staticmethod
    def drop(x, frac=(0, 0.8), multimodal=True):
        # drop from 0% to 80% of the sequences
        def get_drop(x_):
            frac_ = np.random.uniform(*frac)
            drop_num = round(frac_ * len(x_))
            drop_idxs = np.random.choice(len(x_), drop_num, replace=False)
            return drop_idxs
        if len(x) > 0:
            if multimodal:
                drop_idxs = get_drop(x[0])
            x_aug = []
            for x_ in x:
                x_aug_ = np.copy(x_)
                if not multimodal:
                    drop_idxs = get_drop(x_)
                x_aug_[drop_idxs] = 0.0
                x_aug.append(x_aug_)
            return x_aug
        return x

    @staticmethod
    def drop_consecutive(x, frac=(0, 0.8), multimodal=True):
        def get_drop(x_):
            frac_ = np.random.uniform(*frac)
            drop_num = round(frac_ * len(x_))
            start_idx = np.random.randint(0, max(len(x_) - drop_num, 1))
            return start_idx, drop_num
        # drop consecutively from 0% to 80% of the sequence
        if len(x) > 0:
            if multimodal:
                start_idx, drop_num = get_drop(x[0])
            x_aug = []
            for x_ in x:
                x_aug_ = np.copy(x_)
                if not multimodal:
                    start_idx, drop_num = get_drop(x_)
                x_aug_[start_idx:start_idx+drop_num] = 0.0
                x_aug.append(x_aug_)
            return x_aug
        return x

    @staticmethod
    def crop(x, size=(0.08, 1), multimodal=True):
        # crop from 8% to 100% of the sequence
        def get_crop(x_):
            size_ = np.random.uniform(*size)
            crop_num = round(size_ * len(x_))
            start_idx = np.random.randint(0, max(len(x_) - crop_num, 1))
            return start_idx, crop_num
        if len(x) > 0:
            if multimodal:
                start_idx, crop_num = get_crop(x[0])
            x_aug = []
            for x_ in x:
                x_aug_ = np.copy(x_)
                if not multimodal:
                    start_idx, crop_num = get_crop(x_)
                x_aug_[:start_idx] = 0.0
                x_aug_[start_idx + crop_num:] = 0.0
                x_aug.append(x_aug_)
            return x_aug
        return x

    @staticmethod
    def mixup(x, alpha=1.0):
        if len(x) > 0:
            indices = np.random.permutation(x[0].shape[0])
            lam = np.random.beta(alpha, alpha)
            x = [x_ * lam + x_[indices] * (1 - lam) for x_ in x]
        return x


if __name__ == "__main__":
    ds = MultimodalManipulationDataset("/fastdata/visionandtouch/triangle_real_data", task="ee_yaw_next")