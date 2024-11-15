import sys
import torch.random
import pickle
sys.path.extend(["/home/bdufumier/code/FactorCL"])
import numpy as np
import argparse
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from hateful_memes.hateful_memes import HatefulMemesDataModule
from hateful_memes.transformer import LanguageEncoder
from hateful_memes.vit import VisionTransformer
from multibench_model import train_ssl_hatefulmemes, FactorCLSSL


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
        VisionTransformer(
            "vit_base_patch32_clip_224.openai", pretrained=True,
            output_value="embedding", freeze=True),
        LanguageEncoder("clip-ViT-B-32-multilingual-v1",
                        output_value='sentence_embedding', normalize_embeddings=True,
                        use_dataset_cache=False, freeze=True)
    ]
    factorcl_ssl = FactorCLSSL(encoders=encoders, feat_dims=[512, 512], y_ohe_dim=3, lr=args.lr).cuda()
    train_ssl_hatefulmemes(factorcl_ssl, train_loader, num_epoch=100, num_club_iter=1)

    factorcl_ssl.eval()

    data_module_test = HatefulMemesDataModule(args.root_dataset, model="Sup", batch_size=64, num_workers=32)
    eval_train_loader = data_module_test.train_dataloader()
    eval_test_loader = data_module_test.test_dataloader()
    train_embeds_x1 = np.concatenate(
        [factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][1]).detach().cpu().numpy() for data in
         eval_train_loader], axis=0)
    train_embeds_x2 = np.concatenate(
        [factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][1]).detach().cpu().numpy() for data in
         eval_train_loader], axis=0)

    train_embeds = np.concatenate([train_embeds_x1, train_embeds_x2], axis=1)
    train_labels = np.concatenate([data[1].detach().cpu().numpy() for data in eval_train_loader])

    test_embeds_x1 = np.concatenate(
        [factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][1]).detach().cpu().numpy() for data in
         eval_test_loader], axis=0)
    test_embeds_x2 = np.concatenate(
        [factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][1]).detach().cpu().numpy() for data in
         eval_test_loader], axis=0)
    test_embeds = np.concatenate([test_embeds_x1, test_embeds_x2], axis=1)
    test_labels = np.concatenate([data[1].detach().cpu().numpy() for data in eval_test_loader])

    # Train Logistic Classifier
    clf = LogisticRegressionCV(Cs=5, max_iter=200, n_jobs=20).fit(train_embeds, train_labels)
    acc_score = accuracy_score(test_labels, clf.predict(test_embeds))
    roc_auc = roc_auc_score(test_labels, clf.predict_proba(test_embeds)[:, 1])
    print(f"run={args.run}, acc={acc_score}, AUC={roc_auc}")
