import torch
import random
import numpy as np
from collections import namedtuple
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)
    
    
class LastKRatingDataset(Dataset):
    def __init__(self, items_idx, sequence, last_items=10):
        self.sequence = sequence
        self.samples = list(items_idx.items())
        self.last_items = last_items

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        samples_idx = [self.samples[id_] for id_ in idx]
        y_true = torch.tensor([self.sequence[ids[0]][seq_id][1] for ids, seq_id in samples_idx])
        user_history_idx = []
        for (user_id, item_id), seq_id in samples_idx:
            if seq_id > last_items:
                user_history_idx.append(self.sequence[user_id][seq_id-self.last_items:seq_id])
            else:
                user_history_idx.append([(UNK, 0) for _ in range(self.last_items - seq_id)] + \
                                self.sequence[user_id][:seq_id])
        user_history_idx = torch.tensor(user_history_idx, dtype=torch.long)
        user_idx = torch.tensor([sample[0][0] for sample in samples_idx])
        item_idx = torch.tensor([sample[0][1] for sample in samples_idx])
        return user_idx, user_history_idx, item_idx, y_true


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings, test):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        assert 'userId' in test.columns
        assert 'itemId' in test.columns
        assert 'rating' in test.columns

        self.ratings = ratings
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        # split data for NCF learning
        self.train_ratings = self.ratings
        self.test_ratings, self.negatives = self._split_test(test)

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings

    def _binarize(self, ratings):
        """binarize into 0 or 1, implicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    def _split_test(self, test):
        """test split on positive and negative samples"""
        positive = test[test['rating'] == 1]
        negative = test[test['rating'] == 0]
        return positive[['userId', 'itemId', 'rating']], negative[['userId', 'itemId', 'rating']]

    def instance_a_train_loader(self, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        for row in self.train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self):
        """create evaluate data"""
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in self.test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
        for row in self.negatives.itertuples():
            negative_users.append(int(row.userId))
            negative_items.append(int(row.itemId))
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]
    

UserItemRating = namedtuple('UserItemRating', 'userId itemId rating')
    

class NegativeSamplingLastKGenerator(SampleGenerator):
    """Construct dataset with negative sampling"""

    def __init__(self, ratings, test):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        assert 'userId' in test.columns
        assert 'itemId' in test.columns
        assert 'rating' in test.columns

        self.user_pool = sorted(ratings['userId'].unique())
        
        self.train_ratings = self._split_train(ratings)
        self.test_ratings, self.negatives = self._split_test(test)
        
    def _split_train(self, train):
        train_by_users = []
        for userId, user_df in train.groupby('userId'):
            positives = []
            for row in user_df[user_df['rating'] == 1].itertuples():
                positives.append(UserItemRating(userId, int(row.itemId), float(row.rating)))
            negatives = []
            for row in user_df[user_df['rating'] == 0].itertuples():
                negatives.append(UserItemRating(userId, int(row.itemId), float(row.rating)))
            train_by_users.append((positives, negatives))
        return train_by_users

    def instance_a_train_loader(self, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        for positives, negatives in self.train_ratings:
            for row in positives:
                users.append(row.userId)
                items.append(row.itemId)
                ratings.append(row.rating)
            neg_chosen = np.random.choice(len(negatives), min(int(0.5 * len(positives)), len(negatives)))
            for choice in neg_chosen:
                users.append(negatives[choice].userId)
                items.append(negatives[choice].itemId)
                ratings.append(negatives[choice].rating)
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class SequenceGenerator(SampleGenerator):
    """Construct dataset with negative sampling """

    def __init__(self, ratings, test):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        assert 'userId' in test.columns
        assert 'itemId' in test.columns
        assert 'rating' in test.columns

        self.user_pool = sorted(ratings['userId'].unique())
        
        self.train_ratings = self._split_train(ratings)
        self.test_ratings, self.negatives = self._split_test(test)
        
    def _split_train(self, train):
        train_by_users = []
        for userId, user_df in train.groupby('userId'):
            positives = []
            for row in user_df[user_df['rating'] == 1].itertuples():
                positives.append(UserItemRating(userId, int(row.itemId), float(row.rating)))
            negatives = []
            for row in user_df[user_df['rating'] == 0].itertuples():
                negatives.append(UserItemRating(userId, int(row.itemId), float(row.rating)))
            train_by_users.append((positives, negatives))
        return train_by_users

    def instance_a_train_loader(self, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        for positives, negatives in self.train_ratings:
            for row in positives:
                users.append(row.userId)
                items.append(row.itemId)
                ratings.append(row.rating)
            neg_chosen = np.random.choice(len(negatives), 5 * min(len(positives), len(negatives)))
            for choice in neg_chosen:
                users.append(negatives[choice].userId)
                items.append(negatives[choice].itemId)
                ratings.append(negatives[choice].rating)
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

