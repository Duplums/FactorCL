import tqdm
import torch
import torch.nn as nn
from critic_objectives import InfoNCECritic, CLUBInfoNCECritic
import torch.optim as optim

def mlp_head(dim_in, feat_dim):
    return nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

class FactorCLSSL(nn.Module):
    def __init__(self, encoders, feat_dims, y_ohe_dim, temperature=1, activation='relu', lr=1e-4, ratio=1):
        super(FactorCLSSL, self).__init__()
        self.critic_hidden_dim = 512
        self.critic_layers = 1
        self.critic_activation = 'relu'
        self.lr = lr
        self.ratio = ratio
        self.y_ohe_dim = y_ohe_dim
        self.temperature = temperature

        # encoder backbones
        self.feat_dims = feat_dims
        self.backbones = nn.ModuleList(encoders)

        # linear projection heads
        self.linears_infonce_x1x2 = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])
        self.linears_club_x1x2_cond = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])

        self.linears_infonce_x1y = mlp_head(self.feat_dims[0], self.feat_dims[0])
        self.linears_infonce_x2y = mlp_head(self.feat_dims[1], self.feat_dims[1])
        self.linears_infonce_x1x2_cond = nn.ModuleList(
            [mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])
        self.linears_club_x1x2 = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])

        # critics
        self.infonce_x1x2 = InfoNCECritic(self.feat_dims[0], self.feat_dims[1], self.critic_hidden_dim,
                                          self.critic_layers, activation, temperature=temperature)
        self.club_x1x2_cond = CLUBInfoNCECritic(self.feat_dims[0] * 2, self.feat_dims[1] * 2,
                                                self.critic_hidden_dim, self.critic_layers, activation,
                                                temperature=temperature)

        self.infonce_x1y = InfoNCECritic(self.feat_dims[0], self.feat_dims[0], self.critic_hidden_dim,
                                         self.critic_layers, activation, temperature=temperature)
        self.infonce_x2y = InfoNCECritic(self.feat_dims[1], self.feat_dims[1], self.critic_hidden_dim,
                                         self.critic_layers, activation, temperature=temperature)
        self.infonce_x1x2_cond = InfoNCECritic(self.feat_dims[0] * 2, self.feat_dims[1] * 2,
                                               self.critic_hidden_dim, self.critic_layers, activation,
                                               temperature=temperature)
        self.club_x1x2 = CLUBInfoNCECritic(self.feat_dims[0], self.feat_dims[1], self.critic_hidden_dim,
                                           self.critic_layers, activation, temperature=temperature)

    def ohe(self, y):
        N = y.shape[0]
        y_ohe = torch.zeros((N, self.y_ohe_dim))
        y_ohe[torch.arange(N).long(), y.T[0].long()] = 1
        return y_ohe

    def forward(self, x1, x2, x1_aug, x2_aug):
        # Get embeddings
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        x1_aug_embed = self.backbones[0](x1_aug)
        x2_aug_embed = self.backbones[1](x2_aug)

        # compute losses
        uncond_losses = [
            self.infonce_x1x2(self.linears_infonce_x1x2[0](x1_embed), self.linears_infonce_x1x2[1](x2_embed)),
            self.club_x1x2(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
            self.infonce_x1y(self.linears_infonce_x1y(x1_embed), self.linears_infonce_x1y(x1_aug_embed)),
            self.infonce_x2y(self.linears_infonce_x2y(x2_embed), self.linears_infonce_x2y(x2_aug_embed))
            ]

        cond_losses = [self.infonce_x1x2_cond(torch.cat([self.linears_infonce_x1x2_cond[0](x1_embed),
                                                         self.linears_infonce_x1x2_cond[0](x1_aug_embed)], dim=1),
                                              torch.cat([self.linears_infonce_x1x2_cond[1](x2_embed),
                                                         self.linears_infonce_x1x2_cond[1](x2_aug_embed)], dim=1)),
                       self.club_x1x2_cond(torch.cat([self.linears_club_x1x2_cond[0](x1_embed),
                                                      self.linears_club_x1x2_cond[0](x1_aug_embed)], dim=1),
                                           torch.cat([self.linears_club_x1x2_cond[1](x2_embed),
                                                      self.linears_club_x1x2_cond[1](x2_aug_embed)], dim=1))
                       ]

        return sum(uncond_losses) + sum(cond_losses)

    def learning_loss(self, x1, x2, x1_aug, x2_aug):
        # Get embeddings
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        x1_aug_embed = self.backbones[0](x1_aug)
        x2_aug_embed = self.backbones[1](x2_aug)

        # Calculate InfoNCE loss for CLUB-NCE
        learning_losses = [
            self.club_x1x2.learning_loss(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
            self.club_x1x2_cond.learning_loss(torch.cat([self.linears_club_x1x2_cond[0](x1_embed),
                                                         self.linears_club_x1x2_cond[0](x1_aug_embed)], dim=1),
                                              torch.cat([self.linears_club_x1x2_cond[1](x2_embed),
                                                         self.linears_club_x1x2_cond[1](x2_aug_embed)], dim=1))
            ]
        return sum(learning_losses)

    def get_embedding(self, x1, x2):
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        x1_reps = [self.linears_infonce_x1x2[0](x1_embed),
                   self.linears_club_x1x2[0](x1_embed),
                   self.linears_infonce_x1y(x1_embed),
                   self.linears_infonce_x1x2_cond[0](x1_embed),
                   self.linears_club_x1x2_cond[0](x1_embed)]

        x2_reps = [self.linears_infonce_x1x2[1](x2_embed),
                   self.linears_club_x1x2[1](x2_embed),
                   self.linears_infonce_x2y(x2_embed),
                   self.linears_infonce_x1x2_cond[1](x2_embed),
                   self.linears_club_x1x2_cond[1](x2_embed)]

        return torch.cat(x1_reps, dim=1), torch.cat(x2_reps, dim=1)

    def get_optims(self):
        non_CLUB_params = [self.backbones.parameters(),
                           self.infonce_x1x2.parameters(),
                           self.infonce_x1y.parameters(),
                           self.infonce_x2y.parameters(),
                           self.infonce_x1x2_cond.parameters(),
                           self.linears_infonce_x1x2.parameters(),
                           self.linears_infonce_x1y.parameters(),
                           self.linears_infonce_x2y.parameters(),
                           self.linears_infonce_x1x2_cond.parameters(),
                           self.linears_club_x1x2_cond.parameters(),
                           self.linears_club_x1x2.parameters()]

        CLUB_params = [self.club_x1x2_cond.parameters(),
                       self.club_x1x2.parameters()]

        non_CLUB_optims = [optim.Adam(param, lr=self.lr) for param in non_CLUB_params]
        CLUB_optims = [optim.Adam(param, lr=self.lr) for param in CLUB_params]

        return non_CLUB_optims, CLUB_optims



def train_ssl_trifeatures(model, train_loader, num_epoch=50, num_club_iter=1):
    non_CLUB_optims, CLUB_optims = model.get_optims()
    losses = []

    for _iter in tqdm.tqdm(range(num_epoch)):
        for i_batch, data_batch in enumerate(train_loader):

            x1_batch, x2_batch, x1_aug, x2_aug = data_batch
            x1_batch, x2_batch, x1_aug, x2_aug = x1_batch.cuda(), x2_batch.cuda(), x1_aug.cuda(), x2_aug.cuda()
            loss = model(x1_batch, x2_batch, x1_aug, x2_aug)
            losses.append(loss.detach().cpu().numpy())

            for optimizer in non_CLUB_optims:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in non_CLUB_optims:
                optimizer.step()

            for _ in range(num_club_iter):

                learning_loss = model.learning_loss(x1_batch, x2_batch, x1_aug, x2_aug)

                for optimizer in CLUB_optims:
                    optimizer.zero_grad()

                learning_loss.backward()

                for optimizer in CLUB_optims:
                    optimizer.step()

            if i_batch % 100 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())

    return

