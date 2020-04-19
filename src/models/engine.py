import torch
import sys
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import save_checkpoint, use_optimizer

sys.path.append('..')

from metrics.evaluation import MetronAtK


class UserItemDataset(Dataset):
    """Wrapper, convert <user, item> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

class Engine(object):
    """Meta Engine for training & evaluating NCF model
    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=20)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # implicit feedback
        self.crit = torch.nn.BCELoss()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda']:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            # assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]
            if self.config['use_cuda']:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                
            test_dataset = UserItemDataset(test_users, test_items)
            test_dataloader = DataLoader(test_dataset, batch_size=self.config['batch_size'])
            test_scores = []
            with tqdm(total=len(test_dataloader)) as pbar:
                for i, (test_users_batch, test_items_batch) in enumerate(test_dataloader):
                    test_scores.append(self.model(test_users_batch, test_items_batch).cpu().numpy().reshape(-1))
                    pbar.update()
                                       
            test_scores = np.concatenate(test_scores, axis=0)
            test_scores = list(test_scores)
                    
            if self.config['use_cuda']:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                negative_users = negative_users.cuda()
                negative_items = negative_items.cuda()
            
            negative_dataset = UserItemDataset(negative_users, negative_items)
            negative_dataloader = DataLoader(negative_dataset, batch_size=self.config['batch_size'])
            negative_scores = []
            
            with tqdm(total=len(negative_dataloader)) as pbar:
                for neg_users_batch, neg_items_batch in negative_dataloader:
                    negative_scores.append(self.model(neg_users_batch, neg_items_batch).cpu().numpy().reshape(-1))
                    pbar.update()
                    
            torch.cuda.empty_cache()
            negative_scores = np.concatenate(negative_scores, axis=0)
            negative_scores = list(negative_scores)
                
            if self.config['use_cuda']:
                negative_users = negative_users.cpu()
                negative_items = negative_items.cpu()
            self._metron.subjects = [test_users.data.view(-1).tolist(),
                                     test_items.data.view(-1).tolist(),
                                     test_scores,
                                     negative_users.data.view(-1).tolist(),
                                     negative_items.data.view(-1).tolist(),
                                     negative_scores]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        print('[Evaluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
        return hit_ratio, ndcg

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
