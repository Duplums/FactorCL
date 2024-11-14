import sys
sys.path.append("../../MultiBench")
import os
import sys
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

from unimodals.common_models import Transformer, MLP
from unimodals.common_models import MLP, GRUWithLinear, GRU
from dataset.affect.get_data import get_dataloader

sys.path.append(os.getcwd())
from multibench_model import *

if __name__ == "__main__":

    train_loader, valid_loader, test_loader = get_dataloader('/fastdata/mosei/mosei_senti_data.pkl',
                                                             robust_test=False,
                                                             batch_size=32,
                                                             train_shuffle=True)

    eval_train_loader, eval_valid_loader, eval_test_loader = get_dataloader('/fastdata/mosei/mosei_senti_data.pkl',
                                                                            robust_test=False,
                                                                            batch_size=32,
                                                                            train_shuffle=False)

    encoders = [Transformer(35, 40), Transformer(300, 40)]
    factorcl_ssl = FactorCLSSL(encoders=encoders, feat_dims=[40, 40], y_ohe_dim=3).cuda()
    train_ssl_mosi(factorcl_ssl, train_loader, num_epoch=100, num_club_iter=1)

    factorcl_ssl.eval()

    train_embeds_x1 = np.concatenate([factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][2].cuda())[0].detach().cpu().numpy() for data in eval_train_loader])
    train_embeds_x2 = np.concatenate([factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][2].cuda())[1].detach().cpu().numpy() for data in eval_train_loader])
    train_embeds = np.concatenate([train_embeds_x1, train_embeds_x2], axis=1)
    train_labels = np.concatenate([data[3].detach().cpu().numpy() for data in eval_train_loader])
    train_labels = mosi_label(train_labels)

    test_embeds_x1 = np.concatenate([factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][2].cuda())[0].detach().cpu().numpy() for data in eval_test_loader])
    test_embeds_x2 = np.concatenate([factorcl_ssl.get_embedding(data[0][0].cuda(), data[0][2].cuda())[1].detach().cpu().numpy() for data in eval_test_loader])
    test_embeds = np.concatenate([test_embeds_x1, test_embeds_x2], axis=1)
    test_labels = np.concatenate([data[3].detach().cpu().numpy() for data in eval_test_loader])
    test_labels = mosi_label(test_labels)

    # Train Logistic Classifier
    #clf = LogisticRegression(max_iter=200).fit(train_embeds, train_labels)
    #score = clf.score(test_embeds, test_labels)
    #%%
    #print(score)