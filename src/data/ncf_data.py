import torch
import random
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
        for row in self.negatives:
            negative_users.append(int(row.userId))
            negative_items.append(int(row.itemId))
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]
