import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Averager
from tqdm import tqdm

class CriterionAggregator(nn.Module):
    def __init__(self):
        super(CriterionAggregator, self).__init__()
        self.criterions = []

    def add_criterion(self, new):
        self.criterions.append(new)

    def forward(self, preds, targets):
        total_loss = None
        for c in self.criterions:
            loss = c(preds, targets)
            total_loss = loss if total_loss is None else total_loss + loss

        assert total_loss is not None, print("Loss could not be calculated. Check if your criterions functions properly")

        return total_loss

class BNCrossEntropyLoss(nn.Module):
    def __init__(self, args, weight=None, size_average=True):
        super(BNCrossEntropyLoss, self).__init__()
        self.args = args

    def forward(self, preds, targets, session, lymbda=1):
        """
            Expects input parameters "preds" of type dict {"base_logits":[Tensor], "novel_logits":[Tensor]}
        """

        # base_logits = preds["base_logits"]
        # interm_logits = preds["interm_logits"]
        # novel_logits = preds["novel_logits"]

        base_end_ix = self.args.base_class
        novel_start_ix = self.args.base_class + ((session - 1) * self.args.way)

        base_logits =   preds[:, :base_end_ix]
        interm_logits = preds[:, base_end_ix:novel_start_ix]
        novel_logits =  preds[:, novel_start_ix:novel_start_ix + self.args.way]

        # From the gist at: https://gist.github.com/yang-zhang/217dcc6ae9171d7a46ce42e215c1fee0
        s = novel_logits.exp() / (novel_logits.exp().sum() + base_logits.exp().sum()).unsqueeze(-1)
        
        # filler = torch.zeros(novel_logits.shape[0], targets.min()).cuda()
        s = torch.hstack((base_logits, interm_logits, s))   # s.shape == torch.Size([25, 60])
        
        loss = -s[range(targets.shape[0]), targets].log().mean()
        
        return loss


def KP_weight_constraint_loss(theta1, theta2, lymbda_kp):
    """
        Theta1 and 2 are state dictionaries of the models from phase 1 and 2 respectively
        TODO: Another parameter here could be the number of resent layers that will be used for the weight constraint calculation
    """
    loss = None

    for name, param in theta1.items():
        if "encoder" in name:
            param_l2 = (theta2[name] - param).pow(2).sum()
            loss = param_l2 if loss is None else loss + param_l2

    # TODO: https://github.com/Annusha/LCwoF/blob/f83dad1544b342323e33ea51f17bc03650e1e694/mini_imgnet/resnet12.py#L141 implement this to be able to adjust which layers are skipped and which are not

    assert loss is not None, print("Error computing loss for the Knowledge Preservation module")

    return loss * lymbda_kp
