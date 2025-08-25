import abc
import os.path as osp
from dataloader.data_utils import *
from copy import deepcopy

from utils import (
    ensure_path,
    Averager, Timer, count_acc,
)


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = set_up_datasets(self.args)
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()
        
        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0

        # Sessional Statistics
        self.trlog['max_acc'] = [0.0] * args.sessions
        self.trlog['max_novel_acc'] = [0.0] * args.sessions
        self.trlog['max_base_acc'] = [0.0] * args.sessions
        self.trlog['max_hm'] = [0.0] * args.sessions
        self.trlog["max_am"] = [0.0] * args.sessions
        self.trlog["cw_acc"] = [0.0] * args.sessions
        self.trlog["max_hm_cw"] = [0.0] * args.sessions

        self.trlog['pretrain_knn_acc1'] = 0
        self.trlog['prep_knn_acc1'] = 0
        self.trlog['inc_knn_acc1'] = [0.0] * args.sessions
        self.trlog['cos_sims_inter'] = [0.0] * args.sessions
        self.trlog['cos_sims_intra'] = [0.0] * args.sessions

        self.trlog['base2base'] = [0.0] * args.sessions
        self.trlog['novel2novel'] = [0.0] * args.sessions
        self.trlog['novel2base'] = [0.0] * args.sessions
        self.trlog['base2novel'] = [0.0] * args.sessions

    @abc.abstractmethod
    def train(self):
        pass