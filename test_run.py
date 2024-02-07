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
        'num_epochs': 10,
        'hidden_layers': 1,
        'hidden_size': 100,
        'dropout': 0
    }
    command = args.pop('command')
    config = Config(args.pop('dataset'), **args)
    dataset = FragmentDataset(config)
    vocab = dataset.get_vocab()
    trainer = Trainer(config, vocab)
    trainer.train(dataset.get_loader(), 0)