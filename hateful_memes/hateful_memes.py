import json
import os
from PIL import Image
import torch
import warnings
import random
from PIL import ImageFilter
from torchvision import transforms
from pytorch_lightning import LightningDataModule


# Disable decompression bombs warning for large images
Image.MAX_IMAGE_PIXELS = None
# Silence repeated user warnings from scikit-learn multilabel binarizer for unknown classes.
warnings.filterwarnings("ignore", category=UserWarning)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class HatefulMemesDataModule(LightningDataModule):
    """
    Data module for Hateful Memes vision-language dataset [1] including
    memes (images) + captions describing hateful intentions (text).
    The downstream task is to predict whether the meme promotes hateful intentions or not.

    [1] The hateful memes challenge: Detecting hate speech in multimodal memes. Douwe Kiela, et al., NeurIPS 2020.
    """

    def __init__(self, root: str,
                 model: str,
                 tokenizer=None,
                 batch_size: int = 64,
                 num_workers: int = 0,
                 ):

        """
        :param model: {'Sup', 'FactorCL'}
            The model defines the augmentations to apply to the data.
        :param tokenizer: Which tokenizer use for encoding text with integers
        :param batch_size: Batch size to pass to Dataloaders
        :param num_workers: Number of workers to pass to Dataloaders
        """

        super().__init__()

        self.dataset = "hateful_memes"
        self.model = model
        self.root = root
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.img_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize
        ])

        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        self.setup("test")

    def setup(self, stage: str):
        self.val_dataset = None
        root, metadata = self.root, self.root

        if self.model == 'Sup':
            self.train_dataset = HatefulMemesDatasetSup(root, metadata, "train", self.test_transform, self.tokenizer)
            self.val_dataset = HatefulMemesDatasetSup(root, metadata, "dev", self.test_transform, self.tokenizer)
            self.test_dataset = HatefulMemesDatasetSup(root, metadata, "dev", self.test_transform, self.tokenizer)
        elif self.model == 'FactorCL':
            self.train_dataset = HatefulMemesDatasetFactorCL(root, metadata, self.img_transform, self.augment, "train", self.tokenizer)
            self.val_dataset = HatefulMemesDatasetFactorCL(root, metadata, self.img_transform, self.augment, "dev", self.tokenizer)
            self.test_dataset = HatefulMemesDatasetFactorCL(root, metadata, self.img_transform, self.augment, "dev", self.tokenizer)
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=True, drop_last=False)
        return None


    def test_dataloader(self):
        return torch.utils.data.DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=True, drop_last=False)


class HatefulMemesDatasetBase(torch.utils.data.Dataset):
    def __init__(self, root: str, metadata: str, split: str = "train"):
        """
        :param root: /path/to/HatefulMemes
        :param metadata: /path/to/HatefulMemes/split/ where `split.json` is located
        :param split: "train", "dev" (i.e. validation) or "test"
        """
        self.root = root
        self.split = split
        self.samples = []
        metadata = os.path.join(metadata, f"{split}.jsonl")
        with open(metadata, 'r') as json_file:
            infos = list(json_file)
        for info in infos:
            info = json.loads(info)
            self.samples.append((info["img"], info["text"], info["label"]))

    @staticmethod
    def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def get_raw_item(self, i):
        img_path, text, is_hateful = self.samples[i]
        path = os.path.join(self.root, img_path)
        img = self.pil_loader(path)
        return img, text, is_hateful

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class HatefulMemesDatasetSup(HatefulMemesDatasetBase):
    def __init__(self, root, metadata, split: str = "train", transform=None, tokenizer=None):
        super().__init__(root, metadata, split=split)

        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, text, is_hateful = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # tokenize text
        if self.tokenizer is not None:
            text = self.tokenizer(text)

        return (img, text), is_hateful


class HatefulMemesDatasetFactorCL(HatefulMemesDatasetBase):
    def __init__(self, root, metadata, transform, augment, split: str = "train", tokenizer=None):
        super().__init__(root, metadata, split=split)

        self.transform = transform
        self.augment = augment
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, text, _ = self.get_raw_item(i)

        image = self.transform(img)
        aug = self.augment(img)

        # tokenize text
        if self.tokenizer is not None:
            text = self.tokenizer(text)

        return image, text, aug, text


