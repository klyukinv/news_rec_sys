import torch
import torch.nn as nn
import numpy as np
from engine import Engine
from utils import use_cuda, resume_checkpoint
from torchvision.models import resnet18


PADDING_IDX = 0


class BertCNN(torch.nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        if config['title_embeddings']:
            title_embeddings = np.load(config['title_embeddings'])
            self.item_title = torch.nn.Embedding(self.num_items, title_embeddings.shape[1])
            self.item_title.weights = torch.nn.Parameter(torch.as_tensor(title_embeddings[:-1, :]))
            self.item_title.weights.requires_grad = True
        
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)

        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Flatten()
        )
        
        self.linear = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )
        
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices).view(-1, self.latent_dim, 1)
        title_embedding = self.item_title(item_indices).view(-1, 1, 128)
        emb_maps = user_embedding @ title_embedding
        conv_net_res = self.conv_net(emb_maps.unsqueeze(1))
        logits = self.linear(conv_net_res)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass


class BertCNNEngine(Engine):
    """Engine for training & evaluating BertCNN model"""
    def __init__(self, config):
        self.model = BertCNN(config)
        if config['use_cuda']:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(BertCNNEngine, self).__init__(config)
        print(self.model)