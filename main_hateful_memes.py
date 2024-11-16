import sys

import timm
import torch.random
import pickle
sys.path.extend(["/home/bdufumier/code/FactorCL"])
import numpy as np
import argparse
import os
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from hateful_memes.hateful_memes import HatefulMemesDataModule
from hateful_memes.transformer import LanguageEncoder
from hateful_memes.vit import VisionTransformer
from multibench_model import train_ssl_hatefulmemes, FactorCLSSL

import numpy as np
import torch


class LogisticRegressionTorch:
    def __init__(self, C, max_iter, verbose, random_state=None, **kwargs):
        self.C = C
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.max_iter = max_iter
        self.random_state = random_state
        self.logreg = None
        self.verbose = verbose

    def compute_loss(self, feats, labels):
        loss = self.loss_func(feats, labels)
        wreg = 0.5 * self.logreg.weight.norm(p=2)
        return loss.mean() + (1.0 / self.C) * wreg

    def predict_proba(self, feats):
        assert self.logreg is not None, "Need to fit first before predicting probs"
        return self.logreg(feats).softmax(dim=-1)

    def predict(self, feats):
        assert self.logreg is not None, "Need to fit first before predicting classes"
        return self.predict_proba(feats).argmax(dim=-1)

    def fit(self, feats, labels):
        feat_dim = feats.shape[1]
        num_classes = len(torch.unique(labels))

        # set random seed
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self.logreg = torch.nn.Linear(feat_dim, num_classes, bias=True)
        self.logreg.weight.data.fill_(0.0)
        self.logreg.bias.data.fill_(0.0)

        # move everything to CUDA .. otherwise why are we even doing this?!
        self.logreg = self.logreg.to(feats.device)

        # define the optimizer
        opt = torch.optim.LBFGS(
            self.logreg.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=self.max_iter,
        )
        if self.verbose:
            pred = self.logreg(feats)
            loss = self.compute_loss(pred, labels)
            print(f"(Before Training) Loss: {loss:.3f}")

        def loss_closure():
            opt.zero_grad()
            pred = self.logreg(feats)
            loss = self.compute_loss(pred, labels)
            loss.backward()
            return loss

        opt.step(loss_closure)  # get loss, use to update wts

        if self.verbose:
            pred = self.logreg(feats)
            loss = self.compute_loss(pred, labels)
            print(f"(After Training) Loss: {loss:.3f}")

        return self

if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--root_dataset", type=str, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)

    # Parse the arguments
    args = parser.parse_args()
    seed = 42 + int(args.run)
    results = dict(acc1_share=None, acc1_unique1=None, acc1_unique2=None, acc1_synergy=None)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_module = HatefulMemesDataModule(args.root_dataset, model="FactorCL", batch_size=64, num_workers=32)
    train_loader = data_module.train_dataloader()
    encoders = [
        VisionTransformer("vit_base_patch32_clip_224.openai", pretrained=True, freeze=True),
        LanguageEncoder("clip-ViT-B-32-multilingual-v1",
                        output_value='sentence_embedding', normalize_embeddings=True, mask_prob=0.2,
                        use_dataset_cache=False, freeze=True)
    ]
    factorcl_ssl = FactorCLSSL(encoders=encoders, feat_dims=[512, 512], y_ohe_dim=3, lr=args.lr).cuda()
    train_ssl_hatefulmemes(factorcl_ssl, train_loader, num_epoch=100, num_club_iter=1)

    factorcl_ssl.eval()

    data_module_test = HatefulMemesDataModule(args.root_dataset, model="Sup", batch_size=64, num_workers=32)
    eval_train_loader = data_module_test.train_dataloader()
    eval_test_loader = data_module_test.test_dataloader()
    X_train, y_train = [], []
    for data in eval_train_loader:
        X_train.extend(factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][1]).detach().cpu().numpy())
        y_train.extend(data[1].detach().cpu().numpy())
    X_test, y_test = [], []
    for data in eval_test_loader:
        X_test.extend(factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][1]).detach().cpu().numpy())
        y_test.extend(data[1].detach().cpu().numpy())
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    # Train Logistic Classifier
    clf = LogisticRegressionCV(Cs=5, max_iter=100, verbose=False).fit(X_train, y_train)
    acc_score = accuracy_score(y_test, clf.predict(X_test))
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"run={args.run}, acc={acc_score}, AUC={roc_auc}")
