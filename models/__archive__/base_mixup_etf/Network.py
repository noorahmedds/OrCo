import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet12_encoder import *
# import models.resnet18_lcwof as lcwof_net
from .helper import *
from .mixup import *

from tqdm import tqdm

from utils import *

from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

# class DRLoss(nn.Module):
#     def __init__(self,
#                  reduction='mean',
#                  loss_weight=1.0,
#                  reg_lambda=0.
#                  ):
#         super().__init__()

#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self.reg_lambda = reg_lambda

#     def forward(self,feat,target,h_norm2=None,m_norm2=None,avg_factor=None):
#         assert avg_factor is None
#         dot = torch.sum(feat * target, dim=1)
#         if h_norm2 is None:
#             h_norm2 = torch.ones_like(dot)
#         if m_norm2 is None:
#             m_norm2 = torch.ones_like(dot)

#         loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)

#         return loss * self.loss_weight


def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    # import pdb; pdb.set_trace()
    # assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
    #     "The max irregular value is : {}".format(
    #         torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec

def dot_loss(output, label, cur_M, criterion, H_length, reg_lam=0):
    target = cur_M.weight[label] ## B, d  output: B, d
    if criterion == 'dot_loss':
        loss = - torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1).mean()
    elif criterion == 'reg_dot_loss':
        output = F.normalize(output, p=2, dim=-1) # Normalise the projection because our etf is also nromalised
        dot = torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1) #+ classifier.module.bias[label].view(-1)

        with torch.no_grad():
            M_length = torch.sqrt(torch.sum(target ** 2, dim=1, keepdims=False))

        # loss = (1/2) * torch.mean(((dot-(M_length * H_length)) ** 2) / H_length)
        loss = (1/2) * torch.mean(((dot - 1) ** 2)) # The dot product needs to be as close to one as possible. The output projection needs to be normalised tho

        # if torch.isnan(loss):
        #     import pdb; pdb.set_trace()

        if reg_lam > 0:
            reg_Eh_l2 = torch.mean(torch.sqrt(torch.sum(output ** 2, dim=1, keepdims=True)))
            loss = loss + reg_Eh_l2*reg_lam

    return loss

def dot_reg_loss(proj_encoding, targets, fixed_fc):
    with torch.no_grad():
        proj_encoding_nograd = proj_encoding.detach()
        H_length = torch.clamp(torch.sqrt(torch.sum(proj_encoding_nograd ** 2, dim=1, keepdims=False)), 1e-8)
    reg_lam = 0.0 # TODO: Check
    criterion = "reg_dot_loss"
    loss = dot_loss(proj_encoding, targets, fixed_fc, criterion, H_length, reg_lam=reg_lam)
    return loss


class MYNET(nn.Module):

    def __init__(self, args, mode=None, writer=None):
        super().__init__()

        self.mode = mode
        self.args = args
        
        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
            # Total trainable parameters: 11207232 
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512

        self.writer = writer

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)       # Note the entire number of classes are already added

        # Projection head
        # The projection layer is required to push the projections closer to the fixed fc layer. They remain unchanged throughout the training.
        self.projection_layer = nn.Sequential(
            nn.Linear(512, 2048),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.Dropout(0.2),
            nn.ReLU()
        )

        # Setup etf layer
        orth_vec = generate_random_orthogonal_matrix(self.num_features, self.args.num_classes)
        i_nc_nc = torch.eye(self.args.num_classes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(self.args.num_classes, self.args.num_classes), (1 / self.args.num_classes))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(self.args.num_classes / (self.args.num_classes - 1)))

        # n_extra_classes = 0 # 400
        # num_projection_classes = args.num_classes + n_extra_classes
        # self.temp = torch.stack([torch.normal(torch.zeros(512), torch.ones(512)) for i in range(0,num_projection_classes)])
        # # import pdb; pdb.set_trace()
        self.fixed_fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.fixed_fc.weight.data.copy_(etf_vec.T)
        self.fixed_fc.requires_grad = False

        
    def forward_metric(self, x):
        if "multi" in self.mode:
            enc = self.encode(x)
            proj = self.projection_layer(enc)
            
            proj_out = F.linear(F.normalize(proj, p=2, dim=-1), F.normalize(self.fixed_fc.weight, p=2, dim=-1)) # Cosine classifier
            enc_out = self.fc(enc)  # Simple dot product
            return enc_out, proj_out, proj

        x = self.encode(x)
        if 'cos' in self.mode:
            # x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1)) # Cosine classifier
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1)) # Cosine classifier
            x = self.args.temperature * x
        elif 'dot' in self.mode:
            x = self.fc(x)
            
            

        return x

    def encode(self, x):
        x = self.encoder(x)

        x = F.adaptive_avg_pool2d(x, 1)
        # x = self.avgpool(x)
        
        x = x.squeeze(-1).squeeze(-1)

        return x

    def forward_mixup(self, input, **kwargs):
        out = self.encoder(input, **kwargs)
        
        # Unpacking from mixup
        x, y_a, y_b, lam = out

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)

        if 'cos' in self.mode:
            # x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1)) # Cosine classifier
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fixed_fc.weight, p=2, dim=-1)) # Cosine classifier
            x = self.args.temperature * x
        elif 'dot' in self.mode:
            x = self.fixed_fc(x)

        return x, y_a, y_b, lam

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')
    
    def freeze_backbone(self):
        params_to_train = ['fc.weight']
        for name, param in self.named_parameters():
            # if name in params_to_train: 
            param.requires_grad = True if name in params_to_train else False

    def update_fc(self, trainloader, testloader, class_list, session):
        for batch in trainloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        if self.args.not_data_init:
            # >>> Setting a random set of parameters as the final classification layer for the novel classifier
            new_fc = nn.Parameter(torch.rand(len(class_list), self.num_features, device="cuda"),requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            # self.update_fc_ft(new_fc,data,label,session)
            # Pass here also the testloader and args and session
            return self.update_fc_ft_novel(new_fc, trainloader, testloader, session)

    def update_fc_avg(self,data,label,class_list):
        """
            Using the exemplars available during training to instantiate the classifier for the novel setting
        """
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)     # Mean along the 0th axis of all the embeddings for this particular class index
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto

        # Note the original protonet sums the latent vectors over the support set, and divides by the number of classes. Not by the number of data points
        new_fc = torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self, x, fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_new))
            for epoch in tqdm_gen:
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)

                loss = F.cross_entropy(logits, label) # Technically this is base normalised cross entropy. Because denominator has base sum and targets are only for the novel classes

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)

    def get_optimizer_new(self, optimized_parameters):
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=self.args.decay_new)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(optimized_parameters, lr=self.args.lr_new, weight_decay=self.args.decay_new)
        return optimizer

    def update_fc_ft_novel(self, new_fc, trainloader, testloader, session):
        new_fc=new_fc.clone().detach()
        # TODO: Think of what to do here:
        # ReTrain the projection layer to accomodate the novel classes. The fc layer remains unchanged.

        new_fc=new_fc.detach().clone()
        new_fc.requires_grad=True
        # optimized_parameters = [{'params': new_fc}]
        # optimized_parameters = [{"params":new_fc}, {"params":self.projection_layer.parameters()}]
        optimized_parameters = [{"params":self.projection_layer.parameters()}]
        optimizer = self.get_optimizer_new(optimized_parameters)

        best_loss = None
        best_acc = None
        best_hm = None

        best_fc = None
        label_offset = (session - 1) * self.args.way
        test_class = self.args.base_class + self.args.way * session

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_novel))
            old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()

            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                # for i, batch in enumerate(tqdm_gen, 1):
                for data, label in trainloader:
                    concat_fc = torch.cat([old_fc, new_fc], dim=0)
                    # TODO: Check
                    # concat_fc = self.fixed_fc.weight[:self.args.base_class + self.args.way * (session), :]

                    data = data.cuda()
                    label = label.cuda()

                    # if self.args.mixup_novel:
                    #     # data will now contain the mixed up novel session data
                    #     data, targets_a, targets_b, lam = mixup_data(data, label, alpha = self.args.mixup_alpha)
                    #     data, targets_a, targets_b = map(Variable, (data, targets_a, targets_b))

                    encoding = self.encode(data).detach() # <<< The encoder is essentially frozen
                    proj = self.projection_layer(encoding)

                    # logits = self.get_logits(encoding, concat_fc)
                    # cls_out = self.get_logits(encoding, concat_fc)
                    proj_out = F.linear(F.normalize(proj, p=2, dim=-1), F.normalize(self.fixed_fc.weight, p=2, dim=-1)) # Cosine classifier
                    proj_out = proj_out[:, :test_class]

                    # if self.args.mixup_novel:
                    #     loss = mixup_criterion(F.cross_entropy, cls_out, targets_a, targets_b, lam)
                    # else:
                    #     # The loss now in the denominator accounts for all the novel classes + base classes
                    #     loss = F.cross_entropy(cls_out, label) # Note, no label smoothing here

                    # projection_loss = F.cross_entropy(proj_out, label)
                    projection_loss = dot_reg_loss(proj, label, self.fixed_fc)
                    # loss = loss + projection_loss
                    loss = projection_loss

                    
                    # ta.add(count_acc(cls_out, label))
                    ta.add(count_acc(proj_out, label))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # Model Saving
                if self.args.validation_metric_novel == "none":
                    out_string = '(Novel) Session {}, current_loss {:.3f}, train_acc {:.3f}'\
                        .format(session, total_loss, float('%.3f' % (ta.item() * 100.0)),)
                    # best_fc = new_fc.clone()
                    best_fc = new_fc.detach().clone()
                else:
                    vl, va, vaNovel, vaBase, vhm, vam = self.test_fc(concat_fc, testloader, epoch, session)
                    if self.args.validation_metric_novel == "hm":
                        # Validation
                        if best_hm is None or vhm > best_hm:
                            best_hm = vhm
                            # best_fc = new_fc.clone()
                            # best_fc = new_fc.detach().clone()
                        out_string = '(Novel) Sess: {}, loss {:.3f}, trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}'\
                            .format(
                                session,
                                total_loss,
                                float('%.3f' % (ta.item() * 100.0)),
                                float("%.3f" % (va * 100.0)),
                                float("%.3f" % (best_hm * 100.0)),)
                    elif self.args.validation_metric_novel == "loss":
                        if best_loss is None or vl < best_loss:
                            best_loss = vl
                            # best_fc = new_fc.clone()
                            # best_fc = new_fc.detach().clone()
                        out_string = '(Novel) Session {}, current_loss {:.3f}, train_acc {:.3f}'\
                            .format(session, total_loss,float('%.3f' % (ta.item() * 100.0)),)
                    elif self.args.validation_metric_novel == "acc":
                        if best_acc is None or va > best_acc:
                            best_acc = va
                            # best_fc = new_fc.clone()
                            # best_fc = new_fc.detach().clone()
                        out_string = '(Novel) Session {}, current_loss {:.3f}, train_acc {:.3f}'\
                            .format(session, total_loss,float('%.3f' % (ta.item() * 100.0)),)

                tqdm_gen.set_description(out_string)

                self.writer.add_scalars(f"(Novel) Session {session} Training Graph", {
                    "nta": ta.item(),
                    "nva": 0,
                    # "nvaNovel": vaNovel
                }, epoch)
        
        # Update the weights
        # Deprecated
        # self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(best_fc.data)
        # with torch.no_grad():
        # Updating the fc layer
        with torch.no_grad(): self.fc.weight[:self.args.base_class + self.args.way * session, :] = nn.Parameter(torch.cat([old_fc, best_fc]))


    def update_fc_joint(self,jointloader, testloader, class_list, session):
        if 'ft' in self.args.new_mode:  # further finetune
            # self.update_fc_ft(new_fc,data,label,session)
            return self.update_fc_ft_joint(jointloader, testloader, class_list, session)

    def update_fc_ft_joint(self, jointloader, testloader, class_list, session):
        """
            Extract the parameters associated with the given class list.
            So only base classifier and the novel classifier
            Creating a new classifier for the current classes and the new novel classes
            Note that this new classifier is a contiguous classifier so it needs to be trained with taregts greater 
            than 60 to be offset by the number of novel classes we have seen
        """

        novel_ix_start = self.args.base_class + self.args.way * (session - 1)
        novel_ix_end = self.args.base_class + self.args.way * (session)

        # Create a base fc and novel fc and copy these into the old fc to be trained together
        # By creating separate classifier we ensure that the intermediate fcs donot effect the output for the joint dataset
        base_fc = nn.Parameter(torch.zeros(self.args.base_class, self.num_features, device="cuda"), requires_grad=True)
        new_fc = nn.Parameter(torch.zeros(self.args.way, self.num_features, device="cuda"), requires_grad=True)
        inter_fc = self.fc.weight[self.args.base_class:self.args.base_class + (self.args.way * (session-1)), :].detach()

        # Set the values for the base and new fc from the model fc
        base_fc.data.copy_(self.fc.weight.data[:self.args.base_class, :])
        new_fc.data.copy_(self.fc.weight.data[novel_ix_start:novel_ix_end, :])

        optimized_parameters = [base_fc, new_fc]
        # optimized_parameters = [{"params":[base_fc, new_fc]}, {"params":self.projection_layer.parameters()}]
        optimized_parameters = [{"params":self.projection_layer.parameters()}]
        optimizer = self.get_optimizer_new(optimized_parameters)

        best_loss = None
        best_hm = None
        best_acc = None

        best_fc = None
        label_offset = (session - 1) * self.args.way
        test_class = self.args.base_class + self.args.way * session

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_joint))
            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                # for i, batch in enumerate(tqdm_gen, 1):
                for data, label in jointloader:
                    concat_fc = torch.cat([base_fc, inter_fc, new_fc], dim=0)
                    data = data.cuda()
                    label = label.cuda()

                    if self.args.mixup_joint:
                        data, targets_a, targets_b, lam = mixup_data(data, label, alpha = self.args.mixup_alpha)
                        data, targets_a, targets_b = map(Variable, (data, targets_a, targets_b))

                    encoding = self.encode(data).detach() #<<< The encoder is essentially frozen
                    proj = self.projection_layer(encoding)

                    cls_out = self.get_logits(encoding, concat_fc)
                    proj_out = F.linear(F.normalize(proj, p=2, dim=-1), F.normalize(self.fixed_fc.weight, p=2, dim=-1)) # Cosine classifier
                    proj_out = proj_out[:, :test_class]
                    
                    # label = label - label_offset # Offset labels only for labels with value greater than base_class
                    # label[label >= self.args.base_class] -= label_offset

                    # if self.args.mixup_joint:
                    #     loss = mixup_criterion(F.cross_entropy, cls_out, targets_a, targets_b, lam)
                    # else:
                    #     # The loss now in the denominator accounts for all the novel classes + base classes
                    #     loss = F.cross_entropy(cls_out, label) # Note, no label smoothing here

                    # projection_loss = F.cross_entropy(proj_out, label)
                    projection_loss = dot_reg_loss(proj, label, self.fixed_fc)
                    # loss = loss + projection_loss
                    loss = projection_loss
                    
                    # ta.add(count_acc(cls_out, label))
                    ta.add(count_acc(proj_out, label))

                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt.clip)
                    optimizer.step()

                    total_loss += loss.item()

                # Model Saving
                if self.args.validation_metric_joint == "none":
                    out_string = '(Joint) Session {}, current_loss {:.3f}, train_acc {:.3f}'\
                        .format(session, total_loss, float('%.3f' % (ta.item() * 100.0)),)
                    # best_fc = new_fc.clone()
                    # best_fc = new_fc.detach().clone()
                    best_fc = concat_fc.detach().clone()
                    self.writer.add_scalars(f"(Joint) Session {session} Training Graph", {
                        "jta": ta.item(),
                        "jva": 0,
                        # "jvaNovel": vaNovel
                    }, epoch)
                else:
                    vl, va, vaNovel, vaBase, vhm, vam = self.test_fc(concat_fc, testloader, epoch, session)
                    if self.args.validation_metric_joint == "hm":
                        # Validation
                        if best_hm is None or vhm > best_hm:
                            best_hm = vhm
                            # best_fc = concat_fc.clone()
                            best_fc = concat_fc.detach().clone()
                        out_string = '(Joint) Sess: {}, loss {:.3f}, trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}'\
                            .format(
                                session, 
                                total_loss,
                                float('%.3f' % (ta.item() * 100.0)),
                                float("%.3f" % (va * 100.0)),
                                float("%.3f" % (best_hm * 100.0)),)
                    elif self.args.validation_metric_joint == "loss":
                        if best_loss is None or vl < best_loss:
                            best_loss = vl
                            # best_fc = concat_fc.clone()
                            best_fc = concat_fc.detach().clone()
                        out_string = 'Session {}, current_loss {:.3f}, train_acc {:.3f}'\
                            .format(session, total_loss,float('%.3f' % (ta.item() * 100.0)),)
                    elif self.args.validation_metric_joint == "acc":
                        if best_acc is None or va > best_acc:
                            best_acc = va
                            # best_fc = concat_fc.clone()
                            best_fc = concat_fc.detach().clone()
                        out_string = 'Session {}, current_loss {:.3f}, train_acc {:.3f}'\
                            .format(session, total_loss,float('%.3f' % (ta.item() * 100.0)),)

                    self.writer.add_scalars(f"(Joint) Session {session} Training Graph", {
                        "jta": ta.item(),
                        "jva": va,
                        # "jvaNovel": vaNovel
                    }, epoch)

                tqdm_gen.set_description(out_string)

        
        # Update the weights
        # self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(best_fc.data)

        # self.fc.weight.data[:self.args.base_class, :].copy_(best_fc.data[:self.args.base_class, :])
        # self.fc.weight.data[novel_ix_start:novel_ix_end, :].copy_(best_fc.data[novel_ix_start:novel_ix_end, :])
        with torch.no_grad(): self.fc.weight[:self.args.base_class + self.args.way * session, :] = nn.Parameter(best_fc)
        


    def test_fc(self, fc, testloader, epoch, session):
        """
            Get the testing score for the fc that is being currently trained
        """
        test_class = self.args.base_class + session * self.args.way     # Final class idx that we could be in the labels this session
        vl = Averager()
        va = Averager()

        # >>> Addition
        vaBase = Averager() # Averager for novel classes only        
        vaNovel = Averager() # Averager for novel classes only

        # test_fc = fc.clone().detach()
        self.eval()

        with torch.no_grad():
            # tqdm_gen = tqdm(testloader)
            for batch in testloader:
                data, test_label = [_.cuda() for _ in batch]
                encoding = self.encode(data).detach()

                proj = self.projection_layer(encoding)
                proj_out = F.linear(F.normalize(proj, p=2, dim=-1), F.normalize(self.fixed_fc.weight, p=2, dim=-1)) # Cosine classifier
                proj_out = proj_out[:, :test_class]

                # logits = self.get_logits(encoding, fc)
                # # TODO: Instead of just taking the dot product during testing
                # # We need to integrate the dot product as well as the l2 distance between the prototype and the feature extractor output

                # logits = logits[:, :test_class]

                # loss = F.cross_entropy(logits, test_label)
                # acc = count_acc(logits, test_label)
                loss = dot_reg_loss(proj, test_label, self.fixed_fc)
                acc = count_acc(proj_out, test_label)

                # >>> Addition
                novelAcc, baseAcc = count_acc_(proj_out, test_label, test_class, self.args)

                vaNovel.add(novelAcc)
                vaBase.add(baseAcc)
                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()

            # >>> Addition 
            vaNovel = vaNovel.item()
            vaBase = vaBase.item()

        vhm = hm(vaNovel, vaBase)
        vam = am(vaNovel, vaBase)

        return vl, va, vaNovel, vaBase, vhm, vam