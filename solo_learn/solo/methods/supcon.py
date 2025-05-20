# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.simclr import simclr_loss_func
from solo.methods.base import BaseMethod


class SupCon(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SupCon (https://arxiv.org/abs/2004.11362).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        if cfg.method_kwargs.proj_type == "linear":
            self.projector = nn.Linear(self.features_dim, proj_output_dim)
        elif cfg.method_kwargs.proj_type == "proj":
            # projector
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )
        elif cfg.method_kwargs.proj_type == "proj_alice":
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )
        elif cfg.method_kwargs.proj_type == "proj_ncfscil":
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, self.features_dim * 2),
                nn.BatchNorm1d(self.features_dim * 2),
                nn.LeakyReLU(0.1),
                nn.Linear(self.features_dim * 2, self.features_dim * 2),
                nn.BatchNorm1d(self.features_dim * 2),
                nn.LeakyReLU(0.1),
                nn.Linear(self.features_dim * 2, proj_output_dim, bias=False),
            )
        
        try:
            self.apply_infonce = cfg.method_kwargs.apply_infonce
        except:
            self.apply_infonce = False

        try:
            self.simclr_weight = cfg.method_kwargs.simclr_weight
            self.convex_loss_comb = cfg.method_kwargs.convex_loss_comb
        except:
            self.simclr_weight = 0.1
            self.convex_loss_comb = False
            
    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SupCon, SupCon).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().forward(X)

        z = self.projector(out["feats"])
        # z = self.projector(F.normalize(out["feats"], dim = -1))   # ours
        
        return {**out, "z": z}

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        # z = self.projector(F.normalize(out["feats"], dim = -1))
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SupCon reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SupCon loss and classification loss.
        """

        targets = batch[-1]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z = torch.cat(out["z"])

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        targets = targets.repeat(n_augs)

        nce_loss = simclr_loss_func(
            z,
            indexes=targets,
            temperature=self.temperature,
        )

        if self.apply_infonce:
            indexes = batch[0]
            indexes = indexes.repeat(n_augs)
            selfsuploss = simclr_loss_func(
                z,
                indexes=indexes,
                temperature=self.temperature,
            )
            
            if not self.convex_loss_comb:
                nce_loss += self.simclr_weight * selfsuploss
            else:
                nce_loss = ((1-self.simclr_weight) * nce_loss) + (self.simclr_weight*selfsuploss)

        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss
