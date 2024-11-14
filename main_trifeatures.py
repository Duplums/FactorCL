import sys
import torch.random
import pickle
sys.path.extend(["/home/bdufumier/code/FactorCL"])
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from trifeatures.alexnet import AlexNetEncoder
from trifeatures.trifeatures import TrifeaturesDataModule
from trifeatures.trifeatures_model import train_ssl_trifeatures, FactorCLSSL



if __name__ == "__main__":
    seed = 42
    for run in range(1):
        for biased in [True, False]:
            for lr in [1e-2, 1e-3, 1e-5]:
                results = dict(acc1_share=None, acc1_unique1=None, acc1_unique2=None, acc1_synergy=None)
                np.random.seed(seed)
                torch.manual_seed(seed)
                seed += 1
                data_module = TrifeaturesDataModule(model="FactorCL", batch_size=64, num_workers=16, biased=biased)
                train_loader = data_module.train_dataloader()
                encoders = [AlexNetEncoder(512), AlexNetEncoder(512)]
                factorcl_ssl = FactorCLSSL(encoders=encoders, feat_dims=[512, 512], y_ohe_dim=3, lr=lr).cuda()
                train_ssl_trifeatures(factorcl_ssl, train_loader, num_epoch=100, num_club_iter=1)
                factorcl_ssl.eval()

                tasks = ["share", "unique1", "unique2", "synergy"]
                for i, t in enumerate(tasks):
                    data_module_test = TrifeaturesDataModule(model="Sup", batch_size=64, num_workers=16, task=t, biased=False)
                    eval_train_loader = data_module_test.train_dataloader()
                    eval_test_loader = data_module_test.test_dataloader()
                    train_embeds_x1 = np.concatenate([factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][1].cuda())[0].detach().cpu().numpy() for data in eval_train_loader])
                    train_embeds_x2 = np.concatenate([factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][1].cuda())[1].detach().cpu().numpy() for data in eval_train_loader])
                    train_embeds = np.concatenate([train_embeds_x1, train_embeds_x2], axis=1)
                    train_labels = np.concatenate([data[1].detach().cpu().numpy() for data in eval_train_loader])

                    test_embeds_x1 = np.concatenate([factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][1].cuda())[0].detach().cpu().numpy() for data in eval_test_loader])
                    test_embeds_x2 = np.concatenate([factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][1].cuda())[1].detach().cpu().numpy() for data in eval_test_loader])
                    test_embeds = np.concatenate([test_embeds_x1, test_embeds_x2], axis=1)
                    test_labels = np.concatenate([data[1].detach().cpu().numpy() for data in eval_test_loader])

                    # Train Logistic Classifier
                    clf = LogisticRegression(max_iter=200).fit(train_embeds, train_labels)
                    score = balanced_accuracy_score(test_labels, clf.predict(test_embeds))
                    print(f"biased={biased}, run={run}, task={t}, score={score}")
                    results[f"acc1_{t}"] = score
                with open(f"factorCL_biased={biased}_run-{run}_lr-{lr}.pkl", "wb") as f:
                    pickle.dump(results, f)