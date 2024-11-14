from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import timm
import numpy as np
import tqdm
import torch
from multibench_model import FactorCLSSL, train_ssl_visionandtouch, train_sup_visionandtouch
from Multibench.visionandtouch import MultiBenchDataModule
from Multibench.robotics import ForceEncoder
import logging

def linear_probing(model, train_loader, test_loader):
    model.eval()
    train_embeds = []
    train_labels = []
    for data in tqdm.tqdm(train_loader):
        train_embeds.extend(model.get_embedding(data[0][0].cuda(), data[0][1].cuda()).detach().cpu().numpy())
        train_labels.extend(data[1].detach().cpu().numpy())
    train_embeds = np.array(train_embeds)
    test_embeds = []
    test_labels = []
    for data in tqdm.tqdm(test_loader):
        test_embeds.extend(model.get_embedding(data[0][0].cuda(), data[0][1].cuda()).detach().cpu().numpy())
        test_labels.extend(data[1].detach().cpu().numpy())
    test_embeds = np.array(test_embeds)
    # Train Logistic Classifier
    # Initialize Ridge
    ridge = Ridge()
    params = {'alpha': [0.1, 1.0, 10.0]}
    # Use GridSearchCV to perform RidgeCV with parallelization
    grid_search = GridSearchCV(estimator=ridge, param_grid=params, cv=3, n_jobs=10)
    # Fit the model
    grid_search.fit(train_embeds, train_labels)
    # Predict and check the score
    clf = grid_search.best_estimator_
    y_pred = clf.predict(test_embeds)
    result = mean_squared_error(test_labels, y_pred)
    return result

if __name__ == "__main__":
    data_module = MultiBenchDataModule('FactorCL', batch_size=64, num_workers=10)
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    data_module_eval = MultiBenchDataModule("Sup", batch_size=64, num_workers=10)
    eval_train_loader = data_module_eval.train_dataloader()
    eval_test_loader = data_module_eval.test_dataloader()

    logging.basicConfig(
        filename='results.log',
        level=logging.INFO,
    )

    run_ssl = False
    run_finetuning = True

    if run_ssl:
        # SSL pre-training
        for run in range(0, 4):
            encoders = [timm.create_model(model_name="resnet18", pretrained=False, num_classes=0),
                        ForceEncoder(z_dim=128)]

            factorcl_ssl = FactorCLSSL(encoders=encoders, feat_dims=[512, 128], y_ohe_dim=3, lr=5e-5).cuda()
            train_ssl_visionandtouch(factorcl_ssl, train_loader, num_epoch=100, num_club_iter=1)
            result = linear_probing(factorcl_ssl, eval_train_loader, eval_test_loader)
            print(result)
            logging.info(f"MSE (run {run}) = {result}")
            torch.save(factorcl_ssl.state_dict(), f"factorCL_visionandtouch_epoch100-run{run}.pkl")

    if run_finetuning:
        # Fine-tuning
        for run in range(3, 4):
            encoders = [timm.create_model(model_name="resnet18", pretrained=False, num_classes=0),
                        ForceEncoder(z_dim=128)]
            model = FactorCLSSL(encoders=encoders, feat_dims=[512, 128], y_ohe_dim=3, lr=5e-5).cuda()
            saved_weights = torch.load(f"factorCL_visionandtouch_epoch100-run{run}.pkl")
            print(model.load_state_dict(saved_weights))
            train_sup_visionandtouch(model, eval_train_loader, embed_size=3200, num_epoch=50, lr=1e-4)
            result = linear_probing(model, eval_train_loader, eval_test_loader)
            print(result)
            logging.info(f"MSE (run {run}) = {result}")
            torch.save(model.state_dict(), f"factorCL_visionandtouch-sup_epoch100-run{run}.pkl")













