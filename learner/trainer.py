import time
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter


from .model import Loss, Frag2Mol
from .sampler import Sampler
from utils.filesystem import load_dataset
from utils.postprocess import score_samples
from molecules.fragmentation import reconstruct

### Import dataset
from learner.dataset import FragmentDataset

SCORES = ["validity", "novelty", "uniqueness"]


def save_ckpt(trainer, epoch, filename):
    path = trainer.config.path('ckpt') / filename
    torch.save({
        'epoch': epoch,
        'best_loss': trainer.best_loss,
        'losses': trainer.losses,
        'best_score': trainer.best_score,
        'scores': trainer.scores,
        'model': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
        'scheduler': trainer.scheduler.state_dict(),
        'criterion': trainer.criterion.state_dict()
    }, path)


def load_ckpt(trainer, last=True):
    filename = 'last.pt' if last is True else 'best_loss.pt'
    path = trainer.config.path('ckpt') / filename

    if trainer.config.get('use_gpu') is False:
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(path)

    print(f"loading {filename} at epoch {checkpoint['epoch']+1}...")

    trainer.model.load_state_dict(checkpoint['model'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    trainer.scheduler.load_state_dict(checkpoint['scheduler'])
    trainer.criterion.load_state_dict(checkpoint['criterion'])
    trainer.best_loss = checkpoint['best_loss']
    trainer.losses = checkpoint['losses']
    trainer.best_score = checkpoint['best_score']
    trainer.scores = checkpoint['scores']
    return checkpoint['epoch']


def get_optimizer(config, model):
    return Adam(model.parameters(), lr=config.get('optim_lr'))


def get_scheduler(config, optimizer):
    return StepLR(optimizer,
                  step_size=config.get('sched_step_size'),
                  gamma=config.get('sched_gamma'))


def dump(config, losses, CE_loss, KL_loss, pred_logp_loss, pred_sas_loss, beta_list, scores):
    df = pd.DataFrame(list(zip(losses, CE_loss, KL_loss, beta_list)),
                      columns=["Total loss", "CE loss", "KL loss", "beta"])
    if config.get('pred_logp'):
        df["logP loss"] = pred_logp_loss
    if config.get('pred_sas'):
        df["SAS loss"] = pred_sas_loss
    filename = config.path('performance') / "loss.csv"
    df.to_csv(filename)

    if scores != []:
        df = pd.DataFrame(scores, columns=SCORES)
        filename = config.path('performance') / "scores.csv"
        df.to_csv(filename)


class TBLogger:
    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(config.path('tb').as_posix())
        config.write_summary(self.writer)

    def log(self, name, value, epoch):
        self.writer.add_scalar(name, value, epoch)


class Trainer:
    @classmethod
    def load(cls, config, vocab, last):
        trainer = Trainer(config, vocab)
        epoch = load_ckpt(trainer, last=last)
        return trainer, epoch

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab

        self.model = Frag2Mol(config, vocab)
        self.optimizer = get_optimizer(config, self.model)
        self.scheduler = get_scheduler(config, self.optimizer)
        self.criterion = Loss(config, vocab, pad=vocab.PAD)

        if self.config.get('use_gpu'):
            self.model = self.model.cuda()
        self.pred_logp_loss = None
        self.pred_sas_loss = None
        if self.config.get('pred_logp'):
            self.pred_logp_loss = []
        if self.config.get('pred_sas'):
            self.pred_sas_loss = []
        self.losses = []
        self.CE_loss = []
        self.KL_loss = []
        self.beta_list = []
        self.best_loss = float('inf')
        self.scores = []
        self.best_score = - float('inf')

    def _train_epoch(self, epoch, loader, penalty_weights, beta):
        self.model.train()
        dataset = FragmentDataset(self.config)
        epoch_pred_logp_loss = 0
        epoch_pred_sas_loss = 0
        labels_logp = None
        labels_sas = None
        epoch_loss = 0
        epoch_CE_loss = 0
        epoch_KL_loss = 0
        if epoch > 0 and self.config.get('use_scheduler'):
            self.scheduler.step()
        for idx, (src, tgt, lengths, data_index, tgt_str) in enumerate(loader):
            self.optimizer.zero_grad()
            src, tgt = Variable(src), Variable(tgt)
            if self.config.get('use_gpu'):
                src = src.cuda()
                tgt = tgt.cuda()
            if self.config.get('pred_logp') or self.config.get('pred_sas'):
                # get the target string list
                tgt_str_lst = [self.vocab.translate(target_i) for target_i in tgt.cpu().detach().numpy()]
                # join the target string list and separate by space to compare to data
                tgt_str_lst_join = [" ".join(self.vocab.translate(target_i)) for target_i in tgt.cpu().detach().numpy()]
                output, mu, sigma, z, pred_logp, pred_sas = self.model(src, lengths)
                # get the correct index
                molecules = dataset.data.iloc[list(data_index)]
                data_index_correct = [molecules[molecules['fragments'] == tgt_str_lst_join_i].index.values[0] for tgt_str_lst_join_i in tgt_str_lst_join]
                molecules_correct = dataset.data.iloc[data_index_correct]
                labels_logp = torch.tensor(molecules_correct.logP.values)
                labels_sas = torch.tensor(molecules_correct.SAS.values)
                if self.config.get('use_gpu'):
                    labels_logp = labels_logp.cuda()
                    labels_sas = labels_sas.cuda()
            else:
                output, mu, sigma, z = self.model(src, lengths)
            loss, CE_loss, KL_loss, logp_loss, sas_loss = self.criterion(output, tgt, mu, sigma, pred_logp, labels_logp, pred_sas, labels_sas, epoch, penalty_weights, beta)
            loss.backward()
            clip_grad_norm_(self.model.parameters(),
                            self.config.get('clip_norm'))
            epoch_loss += loss.item()
            epoch_CE_loss += CE_loss.item()
            epoch_KL_loss += KL_loss.item()
            if self.config.get('pred_logp'):
                epoch_pred_logp_loss += logp_loss.item()
            if self.config.get('pred_sas'):
                epoch_pred_sas_loss += sas_loss.item()
            self.optimizer.step()
            if idx == 0 or idx % 1000 == 0:
                print(f"Epoch: {epoch}, beta: {beta[epoch]:.2f}")
                print(f"index: {data_index}")
                if self.config.get('pred_logp'):
                    formatted_logp = [f"{value:.4f}" for value in pred_logp.flatten()]
                    formatted_logp_labels = [f"{value:.4f}" for value in labels_logp.tolist()]
                    logp_pred_str = f"pred logp: {formatted_logp}" if pred_logp is not None else "pred logp: None"
                    logp_label_str = f"labels logp: {formatted_logp_labels}"
                    logp_loss_str = f"logP Loss: {logp_loss.item():.4f}"
                    print(logp_pred_str)
                    print(logp_label_str)
                    print(logp_loss_str)
                if self.config.get('pred_sas'):
                    formatted_sas = [f"{value:.4f}" for value in pred_sas.flatten()]
                    formatted_sas_labels = [f"{value:.4f}" for value in labels_sas.tolist()]
                    sas_pred_str = f"pred sas: {formatted_sas}" if pred_sas is not None else "pred sas: None"
                    sas_label_str = f"labels sas: {formatted_sas_labels}"
                    sas_loss_str = f"SAS Loss: {sas_loss.item():.4f}"
                    print(sas_pred_str)
                    print(sas_label_str)
                    print(sas_loss_str)
                CE_loss_str = f"{CE_loss.item():.4f}"
                KL_loss_str = f"{KL_loss.item():.4f}"
                print(f"CE Loss: {CE_loss_str}, KL Loss: {KL_loss_str}")
        if self.config.get('pred_logp') or self.config.get('pred_sas'):
            return epoch_loss / len(loader), epoch_CE_loss / len(loader), epoch_KL_loss / len(loader), epoch_pred_logp_loss / len(loader), epoch_pred_sas_loss / len(loader)
        else:
            return epoch_loss / len(loader), epoch_CE_loss / len(loader), epoch_KL_loss / len(loader)

    def _valid_epoch(self, epoch, loader):
        use_gpu = self.config.get('use_gpu')
        self.config.set('use_gpu', False)

        num_samples = self.config.get('validation_samples')
        trainer, _ = Trainer.load(self.config, self.vocab, last=True)
        sampler = Sampler(self.config, self.vocab, trainer.model)
        samples = sampler.sample(num_samples, save_results=False)
        dataset = load_dataset(self.config, kind="test")
        _, scores = score_samples(samples, dataset)

        self.config.set('use_gpu', use_gpu)
        return scores

    def log_epoch(self, start_time, epoch, epoch_loss, epoch_scores):
        end = time.time() - start_time
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))

        print(f'epoch {epoch:06d} - '
              f'loss {epoch_loss:6.4f} - ',
              end=' ')

        if epoch_scores is not None:
            for (name, score) in zip(SCORES, epoch_scores):
                print(f'{name} {score:6.4f} - ', end='')

        print(f'elapsed {elapsed}')

    def train(self, loader, start_epoch):
        
        num_epochs = self.config.get('num_epochs')

        logger = TBLogger(self.config)

        ### Get counts of each fragments
        dataset = FragmentDataset(self.config)
        fragment_list = []
        for frag in tqdm(dataset.data.fragments):
            fragment_list.extend(frag.split())
        fragment_counts = pd.Series(fragment_list).value_counts()
        penalty = np.sum(np.log(fragment_counts + 1)) / np.log(fragment_counts + 1)
        penalty_weights = penalty / np.linalg.norm(penalty) * 50
        # full model is 5000
        #beta = [0, 0, 0, 0, 0, 0.002, 0.006, 0.01, 0.02, 0.04, 0.08, 0.1, 0.1, 0.1, 0.1]
        beta = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.1]
        print('beta:', beta)
        self.beta_list = beta

        for epoch in range(start_epoch, start_epoch + num_epochs):
            start = time.time()
            if self.config.get('pred_logp') or self.config.get('pred_sas'):
                epoch_loss, CE_epoch_loss, KL_epoch_loss, logp_loss, sas_loss = self._train_epoch(epoch, loader, penalty_weights, beta)
                if self.config.get('pred_logp'):
                    self.pred_logp_loss.append(logp_loss)
                if self.config.get('pred_sas'):
                    self.pred_sas_loss.append(sas_loss)
            else:
                epoch_loss, CE_epoch_loss, KL_epoch_loss = self._train_epoch(epoch, loader, penalty_weights, beta)
            self.losses.append(epoch_loss)
            self.CE_loss.append(CE_epoch_loss)
            self.KL_loss.append(KL_epoch_loss)
            logger.log('loss', epoch_loss, epoch)
            save_ckpt(self, epoch, filename="last.pt")
            
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                save_ckpt(self, epoch, filename=f'best_loss.pt')

            epoch_scores = None

            if epoch_loss < self.config.get('validate_after'):
                epoch_scores = self._valid_epoch(epoch, loader)
                self.scores.append(epoch_scores)

                if epoch_scores[2] >= self.best_score:
                    self.best_score = epoch_scores[2]
                    save_ckpt(self, epoch, filename=f'best_valid.pt')

                logger.log('validity', epoch_scores[0], epoch)
                logger.log('novelty', epoch_scores[1], epoch)
                logger.log('uniqueness', epoch_scores[2], epoch)

            self.log_epoch(start, epoch, epoch_loss, epoch_scores)
        dump(self.config, self.losses, self.CE_loss, self.KL_loss, self.pred_logp_loss, self.pred_sas_loss, self.beta_list, self.scores)