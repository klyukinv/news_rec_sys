import pandas as pd
import numpy as np
import sys
import torch
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from embedding_neumf import EmbeddingNeuMFEngine
from bert_cnn import BertCNNEngine
sys.path.append('..')
from data.ncf_data import SampleGenerator, NegativeSamplingLastKGenerator

gmf_config = {'alias': 'gmf_factor8-implicit_2nd',
              'num_epoch': 200,
              'batch_size': 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 42977,
              'num_items': 328050,
              'latent_dim': 8,
              'l2_regularization': 0.,
              'use_cuda': True,
              'device_id': 0,
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8_bs512_reg_1e-7',
              'num_epoch': 200,
              'batch_size': 512,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 42977,
              'num_items': 328050,
              'latent_dim': 8,
              'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8-implicit_2nd_Epoch0_HR0.1412_NDCG0.1398.model'),
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'pretrain_neumf_factor16',
                'num_epoch': 200,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 42977,
                'num_items': 328050,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.0000001,
                'use_cuda': True,
                'device_id': 0,
                'pretrain': True,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8-implicit_2nd_Epoch0_HR0.1412_NDCG0.1398.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8_bs512_reg_1e-7_Epoch0_HR0.1674_NDCG0.1781.model'),
                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

w2v_neumf = {'alias': 'w2v_neumf_title',
               'num_epoch': 200,
               'batch_size': 256,
               'optimizer': 'adam',
               'adam_lr': 1e-3,
               'num_users': 42977,
               'num_items': 328050,  # not(add one embedding for PAD item to fill CNN-map (last row stands for it))
               'latent_dim': 20,
               'layers': [300 + 20, 256, 128, 32, 8],  # layers[0] is the concat of latent user vector & latent item vectors
               'l2_regularization': 0.,
               'use_cuda': True,
               'device_id': 0,
               'pretrain': False,
               'title_embeddings': '/data/vnkljukin/title_embeddings_w2v.npy',
               'content_embeddings': None,
               'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
               }

bert_neumf = {'alias': 'bert_neumf_title',
               'num_epoch': 200,
               'batch_size': 1024,
               'optimizer': 'adam',
               'adam_lr': 1e-3,
               'num_users': 42977,
               'num_items': 328050,  # not(add one embedding for PAD item to fill CNN-map (last row stands for it))
               'latent_dim': 32,
               'layers': [128 + 32, 128, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vectors
               'l2_regularization': 0.,
               'use_cuda': True,
               'device_id': 0,
               'pretrain': False,
               'title_embeddings': '/data/vnkljukin/encoded_bert_128.npy',
               'content_embeddings': None,
               'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
               }

bert_conv_outer = {'alias': 'bert_conv_outer_simple_5',
                   'num_epoch': 200,
                   'batch_size': 2048,
                   'optimizer': 'adam',
                   'adam_lr': 1e-3,
                   'num_users': 42977,
                   'num_items': 328050,
                   'latent_dim': 64,
                   'l2_regularization': 0.,
                   'use_cuda': True,
                   'device_id': 0,
                   'pretrain': False,
                   'title_embeddings': '/data/vnkljukin/encoded_bert_128.npy',
                   'content_embeddings': None,
                   'model_dir': '/data/vnkljukin/checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                   }

# Specify the exact model
model = 'bert_conv_outer'

# Load Data
zen_dir = '/data/vnkljukin/train.csv'
zen_rating = pd.read_csv(zen_dir)
zen_rating.rating += 1
zen_rating.rating //= 2
print('Range of userId is [{}, {}]'.format(zen_rating.userId.min(), zen_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(zen_rating.itemId.min(), zen_rating.itemId.max()))
# Test
test_dir = '/data/vnkljukin/solution.csv'
test_rating = pd.read_csv(test_dir)
test_rating.rating += 1
test_rating.rating //= 2
# DataLoader for training
if model == 'w2v_neumf':
    sample_generator = SampleGenerator(ratings=zen_rating, test=test_rating)
    config = w2v_neumf
    engine = EmbeddingNeuMFEngine(config)
elif model == 'bert_neumf':
    sample_generator = NegativeSamplingLastKGenerator(ratings=zen_rating, test=test_rating)
    config = bert_neumf
    engine = EmbeddingNeuMFEngine(config)
elif model == 'bert_conv_outer':
    sample_generator = NegativeSamplingLastKGenerator(ratings=zen_rating, test=test_rating)
    config = bert_conv_outer
    engine = BertCNNEngine(config)
else:
    sample_generator = SampleGenerator(ratings=zen_rating, test=test_rating)
    # config = gmf_config
    # engine = GMFEngine(config)
    # config = mlp_config
    # engine = MLPEngine(config)
    config = neumf_config
    engine = BertCNNEngine(config)
evaluate_data = sample_generator.evaluate_data

for epoch in range(config['num_epoch']):
    torch.cuda.empty_cache()
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    engine.save(config['alias'], epoch, hit_ratio, ndcg)
