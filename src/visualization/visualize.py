import matplotlib.pyplot as plt

from src.metrics.ndcg import ndcg_score
from src.metrics.map import map_score


def visualize(actual, predicted, suptitle, figsize=(12, 10), to_save=False, path=None):
    plt.figure(figsize=figsize)
    plt.suptitle(suptitle)
    if isinstance(predicted, dict):
        names = predicted.keys()

        plt.subplot(121)
        plt.title('NDCG@20')
        plt.bar(x=names, y=(ndcg_score(actual, predicted[name]) for name in names))

        plt.subplot(122)
        plt.title('MAP@20')
        plt.bar(x=names, y=(map_score(actual, predicted[name]) for name in names))
    else:
        plt.subplot(121)
        plt.title('NDCG@20')
        plt.axhline(ndcg_score(actual, predicted))

        plt.subplot(122)
        plt.title('MAP@20')
        plt.axhline(map_score(actual, predicted))

    if to_save:
        plt.savefig(path)
    plt.show()
