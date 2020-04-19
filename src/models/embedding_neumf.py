import torch
import numpy as np
from engine import Engine
from utils import use_cuda, resume_checkpoint


class EmbeddingNeuMF(torch.nn.Module):
    def __init__(self, config):
        super(EmbeddingNeuMF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        
        if config['title_embeddings']:
            title_embeddings = np.load(config['title_embeddings'])
        if config['content_embeddings']:
            content_embeddings = np.load(config['content_embeddings'])

        self.item_title = torch.nn.Embedding(self.num_items, title_embeddings.shape[1])
        self.item_title.weights = torch.nn.Parameter(torch.as_tensor(title_embeddings[:-1, :]))
        self.item_title.weights.requires_grad = True
        
        if config['content_embeddings']:
            self.item_content = torch.nn.Embedding(self.num_items, content_embeddings.shape[1])
            self.item_content.weights = torch.nn.Parameter(torch.as_tensor(content_embeddings))
            self.item_content.weights.requires_grad = False
        
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        self.relu_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        self.do_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.bn_layers.append(torch.nn.BatchNorm1d(out_size))
            self.relu_layers.append(torch.nn.ReLU(inplace=True))
            self.do_layers.append(torch.nn.Dropout(p=0.2))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        title_embedding = self.item_title(item_indices)
        if self.config['content_embeddings']:
            content_embedding = self.item_content(item_indices)
            vector = torch.cat([user_embedding, title_embedding, content_embedding], dim=-1)  # the concat latent vector
        else:
            vector = torch.cat([user_embedding, title_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = self.bn_layers[idx](vector)
            vector = self.relu_layers[idx](vector)
            vector = self.do_layers[idx](vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating


class EmbeddingNeuMFEngine(Engine):
    """Engine for training & evaluating embedding model"""
    def __init__(self, config):
        self.model = EmbeddingNeuMF(config)
        if config['use_cuda']:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(EmbeddingNeuMFEngine, self).__init__(config)
        print(self.model)