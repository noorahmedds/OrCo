import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.Network import MYNET as Net
import numpy as np

class MYNET(Net):

    def __init__(self, args, mode="None"):
        super().__init__(args, "encode")    # The super class is used solely as an encoder so the mode is "encode"
        
        self.fcs = nn.ModuleList([nn.Linear(self.num_features, self.args.base_class, bias=False)])

    def add_fc(self, num_classes, bias=False):
        self.fcs.append(nn.Linear(self.num_features, num_classes, bias=False).cuda())  # Note that this module when added needs to be on the same device as the rest of the model

    def freeze_session_fc(self, session = 0):
        # Session=0 represents the base linear classifier
        # Any other session represents the linear classifiers from the other novel classes
        
        self.fcs[session].requires_grad_ = False # TODO: Check if this is accurate
    
    def freeze_session_fc_lt(self, session = 0):
        # Freeze fcs for all sessions including the passed argument and before

        for i in range(session):
            self.fcs[i].requires_grad_ = False # TODO: Check if this is accurate

    def unfreeze_all(self):
        # TODO: Implement

        return

    def forward(self, input):
        # Takes input and returns a dictionary output = {"base_logits": self.fcs[0](input), "novel_logits":self.fcs[-1](input)}
        input = self.encode(input)
        
        # Ref: https://discuss.pytorch.org/t/returning-a-dictionary-from-forward-call-breaks-ddp/114113 for dictionary output
        # TODO: or this for multi output https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440 Update logic to enable DDP
        preds = {}
        preds["base_logits"] = self.fcs[0](input)
        
        if (len(self.fcs) > 1):
            preds["novel_logits"] = self.fcs[-1](input)    # Logit activations just for the recent most fc layer that was added
            preds["interm_logits"] = torch.empty(0).cuda()
            if len(self.fcs) > 2:
                preds["interm_logits"] = torch.hstack([self.fcs[i](input) for i in range(1,len(self.fcs) - 1)])

        preds["output"] = torch.hstack(list(preds.values()))

        return preds