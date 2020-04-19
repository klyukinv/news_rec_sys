import pandas as pd
import numpy as np


class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
        # the full set
        full = pd.DataFrame({'user': test_users + neg_users,
                             'item': test_items + neg_items,
                             'score': test_scores + neg_scores,
                             'is_positive': [1] * len(test_scores) + [0] * len(neg_scores)})
        full.sort_values(['user', 'score', 'is_positive'], ascending=(True, False, True), inplace=True)
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        # rank the items according to the scores for each user
        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full

    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        # positive items hit in the top_K items
        test_in_top_k = top_k.groupby('user')['is_positive'].sum()
        top_k_count = top_k.groupby('user')['item'].nunique()
        return (test_in_top_k / top_k_count).mean()

    def cal_ndcg(self):
        full, top_k = self._subjects, self._top_k
        full['dcg'] = 0
        full.loc[full['rank'] <= top_k, 'dcg'] = full.loc[full['rank'] <= top_k].apply(
            lambda x: x[3] / np.log(1 + x[4]), axis=1
        )
        full['max_dcg'] = full.groupby('user')['rank'].transform('max')
        full['max_dcg'] = full.apply(lambda x: np.sum(1 / np.log(1 + np.arange(1, min(x[6], top_k) + 1))), axis=1)
        full['ndcg'] = full['dcg'] / full['max_dcg']
        return full['ndcg'].sum() * 1.0 / full['user'].nunique()


if __name__ == '__main__':
    metr = MetronAtK(3)
    metr.subjects = [
        (0, 0, 0, 1, 1, 1), (0, 1, 2, 0, 1, 2), (0, 0, 1, 1, 1, 1),
        (0, 0, 0, 1, 1, 1), (3, 4, 5, 3, 4, 5), (1, 1, 0, 0, 0, 0)
    ]
    print(metr.cal_hit_ratio(), metr.cal_ndcg())  # 0.6666666666666666 0.7346393630113781
