import sys
import torch.random
import pickle
sys.path.extend(["/home/bdufumier/code/FactorCL"])
import numpy as np
import argparse
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from trifeatures.alexnet import AlexNetEncoder
from trifeatures.trifeatures import TrifeaturesDataModule
from trifeatures.trifeatures_model import train_ssl_trifeatures, FactorCLSSL
from torchmetrics.classification import MulticlassAccuracy

if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--root_dataset", type=str, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--biased", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)

    # Parse the arguments
    args = parser.parse_args()
    seed = 42 + int(args.run)
    results = dict(acc1_share=None, acc1_unique1=None, acc1_unique2=None, acc1_synergy=None)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_module = TrifeaturesDataModule(args.root_dataset, model="FactorCL", batch_size=64, num_workers=16, biased=(args.biased == "true"))
    train_loader = data_module.train_dataloader()
    encoders = [AlexNetEncoder(512), AlexNetEncoder(512)]
    factorcl_ssl = FactorCLSSL(encoders=encoders, feat_dims=[512, 512], y_ohe_dim=3, lr=args.lr).cuda()
    train_ssl_trifeatures(factorcl_ssl, train_loader, num_epoch=100, num_club_iter=1)
    factorcl_ssl.eval()

    tasks = ["share", "unique1", "unique2", "synergy"]
    for i, t in enumerate(tasks):
        data_module_test = TrifeaturesDataModule(args.root_dataset, model="Sup", batch_size=64, num_workers=16, task=t, biased=False)
        eval_train_loader = data_module_test.train_dataloader()
        eval_test_loader = data_module_test.test_dataloader()

        X_train, y_train = [], []
        for data in eval_train_loader:
            X_train.extend(factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][1].cuda()).detach().cpu().numpy())
            y_train.extend(data[1].detach().cpu().numpy())
        X_test, y_test = [], []
        for data in eval_test_loader:
            X_test.extend(factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][1].cuda()).detach().cpu().numpy())
            y_test.extend(data[1].detach().cpu().numpy())

        # Train Logistic Classifier
        clf = LogisticRegressionCV(Cs=5, max_iter=100, n_jobs=20).fit(X_train, y_train)
        C = len(set(y_test))
        score = MulticlassAccuracy(num_classes=C, average="macro")(torch.tensor(y_test), torch.from_numpy(clf.predict(X_test)))
        print(f"biased={args.biased}, run={args.run}, task={t}, score={score}")
        results[f"acc1_{t}"] = score
    with open(os.path.join(args.root, f"factorCL_biased={args.biased}_run-{args.run}_lr-{args.lr}.pkl"), "wb") as f:
        pickle.dump(results, f)
