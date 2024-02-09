from learner.dataset import FragmentDataset
from learner.trainer import Trainer
from utils.config import Config

if __name__ == '__main__':
    args = {
        'command': 'train',
        'dataset': 'ZINC',
        'use_gpu': True,
        'use_mask': False,
        'batch_size': 16,
        'embed_size': 100,
        'num_epochs': 15,
        'hidden_layers': 3,
        'hidden_size': 200,
        'latent_size': 200,
        'pooling': 'sum_fingerprints',
        'pred_logp': True
    }
    command = args.pop('command')
    config = Config(args.pop('dataset'), **args)
    dataset = FragmentDataset(config)
    vocab = dataset.get_vocab()
    trainer = Trainer(config, vocab)
    trainer.train(dataset.get_loader(), 0)
    #python  manage.py train --dataset ZINC --use_gpu --no_mask --batch_size 16 --embed_size 100 --num_epochs 15 --hidden_layers 3 --hidden_size 200 --latent_size 200 --pooling sum_fingerprints --pred_sas --pred_logp