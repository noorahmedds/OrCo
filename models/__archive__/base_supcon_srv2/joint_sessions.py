def update_fc_ft_joint_online(self, trainloader, testloader, session, gaussian_prototypes, prototypeLabels, classVariances):
    """
        Extract the parameters associated with the given class list.
        So only base classifier and the novel classifier
        Creating a new classifier for the current classes and the new novel classes
        Note that this new classifier is a contiguous classifier so it needs to be trained with taregts greater 
        than 60 to be offset by the number of novel classes we have seen
    """
    optimizer = self.get_optimizer_new(self.projector.parameters())

    if self.args.joint_schedule == "Milestone":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80],gamma=self.args.gamma)
    else:
        warmup_epochs = 10
        min_lr = 0.0
        scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=10,
                max_epochs=self.args.epochs_joint,
                warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_new,
                eta_min=min_lr
            )

    criterion = nn.CrossEntropyLoss()

    best_loss = None
    best_hm = None
    best_acc = None

    best_fc = None
    label_offset = 0
    if self.args.base_only:
        label_offset = (session - 1) * self.args.way

    hm_patience = self.args.hm_patience
    hm_patience_count = 0
    
    novel_class_start = self.args.base_class + (self.args.way * (session-1))
    test_class = self.args.base_class + session * self.args.way

    with torch.enable_grad():
        tqdm_gen = tqdm(range(self.args.epochs_joint))

        # To emulate augmentation in generated data we need to mixup the generated data
        # Basically a pair of samples from a single class will be averaged to create a new instance, the choice of pair is entirely random.

        for epoch in tqdm_gen:
            total_loss = 0
            ta = Averager()
            taNovel = Averager()
            taBase = Averager()
            # for i, batch in enumerate(tqdm_gen, 1):
            for i, (data, label) in enumerate(trainloader):
                data = data.cuda()
                label = label.cuda()

                # Instance mixing novel classes
                data, label = instance_mixup_data(data, label)
                data, label = map(Variable, (data, label))

                label[label >= self.args.base_class] -= label_offset

                # encoding = self.encode(data).detach() #<<< The encoder is essentially frozen
                encoding = self.encode(data, detach_f=True)

                # Sample gaussian data directly with 1 shot only
                gaussian_encoding, gaussian_labels = gaussian_utils.generateMutivariateData(gaussian_prototypes, prototypeLabels, classVariances, 1)
                gaussian_encoding = torch.tensor(gaussian_encoding).to(torch.float).cuda()
                gaussian_encoding = self.projector(gaussian_encoding)

                # Concatenating the augmented gaussian data and the current novel class encodings
                encoding = torch.cat((gaussian_encoding, encoding)).to(torch.float)
                label = torch.cat((torch.tensor(gaussian_labels).cuda(), label)).to(torch.long)

                # Get the cosine similarity to the classifier
                logits = self.fc(encoding)

                if self.args.mixup_joint:
                    # loss = mixup_criterion(F.cross_entropy, logits, targets_a, targets_b, lam)
                    # loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                    pass
                else:
                    if self.args.joint_loss in ['ce_even', 'ce_inter']:
                        losses = []
                        novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                        if self.args.joint_loss == "ce_even":
                            base_classes_idx = torch.argwhere(label < novel_class_start).flatten()
                        elif self.args.joint_loss == "ce_inter":
                            # TODO: For new sessions weight everybody together
                            inter_classes_idx = torch.argwhere((label >= self.args.base_class) & (label < novel_class_start)).flatten()
                            base_classes_idx = torch.argwhere(label < self.args.base_class).flatten()

                            if inter_classes_idx.numel() != 0:
                                inter_loss = criterion(logits[inter_classes_idx, :], label[inter_classes_idx])
                                losses.append(inter_loss)

                        if novel_classes_idx.numel() != 0:
                            # Loss computed using the novel classes
                            novel_loss = criterion(logits[novel_classes_idx, :], label[novel_classes_idx])
                            losses.append(novel_loss)

                        if base_classes_idx.numel() != 0:
                            base_loss = criterion(logits[base_classes_idx, :], label[base_classes_idx])
                            losses.append(base_loss)

                        loss = 0
                        loss_string = f"Losses =>"
                        idx2name = {0:"novel", 1:"base"}
                        for idx, l in enumerate(losses): 
                            loss += l
                            loss_string += f" loss {idx2name[idx]}: {loss:.3f},"
                        # tqdm_gen.set_description(loss_string)
                        loss /= len(losses)
                    elif self.args.joint_loss == "ce_weighted":
                        novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                        base_classes_idx = torch.argwhere(label < novel_class_start).flatten()

                        w_current = (1/(session+1))     # 0.5, 0.333, 0.25
                        w_replay = 1-w_current          # 0.5, 0.667, ...

                        loss = 0
                        if novel_classes_idx.numel() != 0:
                            novel_loss = criterion(logits[novel_classes_idx, :], label[novel_classes_idx])
                            loss += w_current * novel_loss

                        if base_classes_idx.numel() != 0:
                            base_loss = criterion(logits[base_classes_idx, :], label[base_classes_idx])
                            loss += w_replay * base_loss
                    else:
                        # Original ce
                        loss = criterion(logits, label) # Note, no label smoothing here
                
                ta.add(count_acc(logits, label))

                n,b = count_acc_(logits, label, test_class, self.args)
                taNovel.add(n)
                taBase.add(b)

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt.clip)
                optimizer.step()

                total_loss += loss.item()

            # Model Saving
            vl, va, vaNovel, vaBase, vhm, vam, vbin, vaNN, vaBB = self.test_fc_head(self.fc, testloader, epoch, session)
            if self.args.validation_metric == "hm":
                # Validation
                if best_hm is None or vhm > best_hm:
                    best_hm = vhm
                    # best_fc = concat_fc.clone()
                    # best_fc = concat_fc.detach().clone()
                    best_fc = deepcopy(self.projector.state_dict())
                    hm_patience_count = 0
                else:
                    hm_patience_count += 1
                lrc = scheduler.get_last_lr()[0]
                out_string = '(Joint) Sess: {}, loss {:.3f}, trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}, (train) b/n:{:.3f}/{:.3f}, (test) b/n={:.3f}/{:.3f}, lrc={:.3f}'\
                    .format(
                        session, 
                        total_loss,
                        float('%.3f' % (ta.item() * 100.0)),
                        float("%.3f" % (va * 100.0)),
                        float("%.3f" % (best_hm * 100.0)),
                        float("%.3f" % (taBase.item() * 100.0)),
                        float("%.3f" % (taNovel.item() * 100.0)),
                        float("%.3f" % (vaBase * 100.0)),
                        float("%.3f" % (vaNovel * 100.0)),
                        float(lrc))

            tqdm_gen.set_description(out_string)

            if hm_patience_count > hm_patience:
                # faster joint session
                break

            scheduler.step()
    
    self.projector.load_state_dict(best_fc, strict=True)    


def update_fc_ft_joint_nc_fscil(self, trainloader, testloader, session, old_prototypes, old_labels):
    # Optimising only the projector
    optimizer = self.get_optimizer_new(self.projector.parameters())

    # Setting up the scheduler
    if self.args.joint_schedule == "Milestone":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80],gamma=self.args.gamma)
    else:
        warmup_epochs = 10
        min_lr = 0.0
        scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=10, max_epochs=self.args.epochs_joint,
                warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_new,
                eta_min=min_lr)

    criterion = nn.CrossEntropyLoss(label_smoothing = self.args.label_smoothing_joint)

    best_loss = None
    best_hm = None
    best_acc = None
    best_projector = None

    hm_patience = self.args.hm_patience
    hm_patience_count = 0
    
    average_gap = Averager()
    average_gap_n = Averager()
    average_gap_b = Averager()
    
    novel_class_start = self.args.base_class + (self.args.way * (session-1))

    self.eval() # Fixing batch norm

    old_prototypes = torch.tensor(old_prototypes).to(torch.float).cuda()
    old_labels = torch.tensor(old_labels).to(torch.long).cuda()

    with torch.enable_grad():
        tqdm_gen = tqdm(range(self.args.epochs_joint))
        for epoch in tqdm_gen:
            total_loss = 0
            ta = Averager()
            tl = {
                "novel": Averager(),
                "base": Averager()
            }
            ta = {
                "novel": Averager(),
                "base": Averager()
            }
            for data, label in trainloader:
                data = data.cuda()
                label = label.cuda()

                encoding = self.encode(data, detach_f=True)

                # Get loss for the current data
                # novel_logits = self.fc(encoding)/0.01    # Novel temperature which
                novel_logits = self.fc(encoding)
                novel_loss = criterion(novel_logits, label)

                old_encodings = self.projector(old_prototypes)
                base_logits = self.fc(old_encodings)
                base_loss = criterion(base_logits, old_labels)

                # denominator = label.shape[0] + old_labels.shape[0]  # New number of samples + old number of classes
                loss = novel_loss + (0.5*base_loss) # Reduce base loss so that novel classes can be trained better

                ta["novel"].add(count_acc(novel_logits, label))
                ta["base"].add(count_acc(base_logits, old_labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                tl["novel"].add(novel_loss)
                tl["base"].add(base_loss)
            
            vl, va, vaNovel, vaBase, vhm, vam, vbin, vaNN, vaBB = self.test_fc_head(self.fc, testloader, epoch, session)
            if self.args.validation_metric == "hm":
                # Validation
                if best_hm is None or vhm > best_hm:
                    best_hm = vhm
                    hm_patience_count = 0
                    best_projector = deepcopy(self.projector.state_dict())
                else:
                    hm_patience_count += 1
                out_string = '(Joint) Sess: {}, loss {:.3f}|(b/n)({:.3f}/{:.3f}), acc (b/n): {:.3f}/{:.3f}, testAcc {:.3f}, bestHM {:.3f}, (test) b/n={:.3f}/{:.3f}, ALG {:.1}, ALGb/n {:.1f}/{:.1f}'\
                    .format(
                        session, 
                        total_loss,
                        float('%.3f' % (tl["base"].item())),
                        float('%.3f' % (tl["novel"].item())),
                        # float('%.3f' % (ta.item() * 100.0)),
                        float('%.3f' % (ta["base"].item() * 100)),
                        float('%.3f' % (ta["novel"].item() * 100)),
                        float("%.3f" % (va * 100.0)),
                        float("%.3f" % (best_hm * 100.0)),
                        float("%.3f" % (vaBase * 100.0)),
                        float("%.3f" % (vaNovel * 100.0)),
                        float('%.3f' % (average_gap.item() * 100.0)),
                        float('%.3f' % (average_gap_b.item() * 100.0)),
                        float('%.3f' % (average_gap_n.item() * 100.0))
                        )

            tqdm_gen.set_description(out_string)
            if hm_patience_count > hm_patience: break
            scheduler.step()
    self.projector.load_state_dict(best_projector, strict=True)

def update_fc_ft_joint_gaussian_mixup(self, jointloader, testloader, class_list, session, generated):
    gaus, aug_gaus = generated
    gaussian_data, gaussian_labels = gaus
    aug_gaus_data, aug_gaus_labels = aug_gaus

    # optimized_parameters = [base_fc, new_fc]
    if self.args.train_novel_in_joint:
        self.fc.novel_requires_grad(value=True)

    optimizer = self.get_optimizer_new(self.fc.parameters())

    criterion = nn.CrossEntropyLoss(label_smoothing = self.args.label_smoothing_joint)

    best_loss = None
    best_hm = None
    best_acc = None

    best_fc = None
    label_offset = 0
    if self.args.base_only:
        # Remove the logits between the novel and base classes
        # or completely 0 them out.
        label_offset = (session - 1) * self.args.way

    hm_patience = 15
    hm_patience_count = 0

    # shuffle generated data
    shuffler = np.arange(gaussian_data.shape[0])
    np.random.shuffle(shuffler)
    gaussian_data = gaussian_data[shuffler]
    gaussian_labels = gaussian_labels[shuffler]
    
    # split equally
    total_iter = jointloader.__len__()
    generated_data = np.array_split(gaussian_data, total_iter)
    generated_labels = np.array_split(gaussian_labels, total_iter)

    novel_class_start = self.args.base_class + (self.args.way * (session-1))

    with torch.enable_grad():
        tqdm_gen = tqdm(range(self.args.epochs_joint))

        # To emulate augmentation in generated data we need to mixup the generated data
        # Basically a pair of samples from a single class will be averaged to create a new instance, the choice of pair is entirely random.

        for epoch in tqdm_gen:
            total_loss = 0
            ta = Averager()
            # for i, batch in enumerate(tqdm_gen, 1):
            for i, (data, label) in enumerate(jointloader):

                data = data.cuda()
                label = label.cuda()
                label[label >= self.args.base_class] -= label_offset

                if self.args.mixup_joint:
                    data, targets_a, targets_b, lam = mixup_data(data, label, alpha = self.args.mixup_alpha)
                    data, targets_a, targets_b = map(Variable, (data, targets_a, targets_b))

                encoding = self.encode(data).detach() #<<< The encoder is essentially frozen

                # logits = self.get_logits(encoding, concat_fc)
                logits_novel = self.fc.get_logits(encoding)
                loss_novel = criterion(logits_novel, label) # Note, no label smoothing here

                # From the gaussian generated data we now create base logits. but before that we must mixup the embeddings
                # augmented_data = gaussian_utils.augmentMultivariateData(generated_data[i], generated_labels[i], aug_gaus_data, aug_gaus_labels)
                augmented_data = torch.tensor(generated_data[i]).cuda().to(torch.float)
                augmented_label = torch.tensor(generated_labels[i]).cuda().to(torch.long)
                mixup_embedding_base, targets_a, targets_b, lam = mixup_data(augmented_data, augmented_label)
                logits_base = self.fc.get_logits(mixup_embedding_base)
                loss_base = mixup_criterion(criterion, logits_base, targets_a, targets_b, lam)

                # Combine losses
                loss = (loss_novel + loss_base) / 2.0
                
                ta.add(count_acc(logits_novel, label))
                ta.add(count_acc(logits_base, augmented_label))

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt.clip)
                optimizer.step()

                total_loss += loss.item()

            # Model Saving
            vl, va, vaNovel, vaBase, vhm, vam, vbin, vaNN, vaBB = self.test_fc_head(self.fc, testloader, epoch, session)
            if self.args.validation_metric == "hm":
                # Validation
                if best_hm is None or vhm > best_hm:
                    best_hm = vhm
                    # best_fc = concat_fc.clone()
                    # best_fc = concat_fc.detach().clone()
                    best_fc = deepcopy(self.fc.state_dict())
                    hm_patience_count = 0
                else:
                    hm_patience_count += 1
                out_string = '(Joint) Sess: {}, loss {:.3f}, trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}, b/n={:.3f}/{:.3f}'\
                    .format(
                        session, 
                        total_loss,
                        float('%.3f' % (ta.item() * 100.0)),
                        float("%.3f" % (va * 100.0)),
                        float("%.3f" % (best_hm * 100.0)),
                        float("%.3f" % (vaBase * 100.0)),
                        float("%.3f" % (vaNovel * 100.0)),)

            tqdm_gen.set_description(out_string)

            # self.writer.add_scalars(f"(Joint) Session {session} Training Graph", {
            #     "jta": ta.item(),
            #     "jva": va,
            #     # "jvaNovel": vaNovel
            # }, epoch)

            if hm_patience_count > hm_patience:
                # faster joint session
                break
    
    self.fc.load_state_dict(best_fc, strict=True)  


def update_fc_ft_joint_gaussian_loader(self, jointloader, testloader, class_list, session):
    """
        Extract the parameters associated with the given class list.
        So only base classifier and the novel classifier
        Creating a new classifier for the current classes and the new novel classes
        Note that this new classifier is a contiguous classifier so it needs to be trained with taregts greater 
        than 60 to be offset by the number of novel classes we have seen
    """
    if self.args.train_novel_in_joint:
        self.fc.novel_requires_grad(value=True)

    optimizer = self.get_optimizer_new(self.fc.parameters())

    if self.args.joint_schedule == "Milestone":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80],gamma=self.args.gamma)
    else:
        warmup_epochs = 10
        min_lr = 0.0
        scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=10,
                max_epochs=self.args.epochs_joint,
                warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_new,
                eta_min=min_lr
            )

    criterion = nn.CrossEntropyLoss(label_smoothing = self.args.label_smoothing_joint)

    best_loss = None
    best_hm = None
    best_acc = None

    best_fc = None
    label_offset = 0
    if self.args.base_only:
        # Remove the logits between the novel and base classes
        # or completely 0 them out.
        label_offset = (session - 1) * self.args.way

    hm_patience = self.args.hm_patience
    hm_patience_count = 0
    
    novel_class_start = self.args.base_class + (self.args.way * (session-1))
    test_class = self.args.base_class + session * self.args.way

    with torch.enable_grad():
        tqdm_gen = tqdm(range(self.args.epochs_joint))

        # To emulate augmentation in generated data we need to mixup the generated data
        # Basically a pair of samples from a single class will be averaged to create a new instance, the choice of pair is entirely random.

        for epoch in tqdm_gen:
            total_loss = 0
            ta = Averager()
            taNovel = Averager()
            taBase = Averager()
            # for i, batch in enumerate(tqdm_gen, 1):
            for i, (data, gauss, label, ei_flag) in enumerate(jointloader):
                encoding = torch.empty(0, self.proj_output_dim).cuda()

                data = data[ei_flag == 1]   # Filter the image data
                data = data.cuda()
                label = label.cuda()

                gauss = gauss[ei_flag == 0].cuda()

                label[label >= self.args.base_class] -= label_offset

                # Filter data
                if data.numel() > 0:
                    encoding = self.encode(data, detach_f=True) #<<< The encoder is essentially frozen
                
                if gauss.numel() > 0:
                    gauss = self.projector(gauss)
                    encoding = torch.cat((gauss, encoding))

                # Combine with the encodings in this batch
                label = torch.cat((label[ei_flag == 0], label[ei_flag == 1]))

                logits = self.fc.get_logits(encoding) 

                if self.args.mixup_joint:
                    # loss = mixup_criterion(F.cross_entropy, logits, targets_a, targets_b, lam)
                    loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                else:
                    if self.args.joint_loss in ['ce_even', 'ce_inter']:
                        losses = []
                        novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                        if self.args.joint_loss == "ce_even":
                            base_classes_idx = torch.argwhere(label < novel_class_start).flatten()
                        elif self.args.joint_loss == "ce_inter":
                            inter_classes_idx = torch.argwhere((label >= self.args.base_class) & (label < novel_class_start)).flatten()
                            base_classes_idx = torch.argwhere(label < self.args.base_class).flatten()

                            if inter_classes_idx.numel() != 0:
                                inter_loss = criterion(logits[inter_classes_idx, :], label[inter_classes_idx])
                                losses.append(inter_loss)

                        if novel_classes_idx.numel() != 0:
                            novel_loss = criterion(logits[novel_classes_idx, :], label[novel_classes_idx])
                            losses.append(novel_loss)

                        if base_classes_idx.numel() != 0:
                            base_loss = criterion(logits[base_classes_idx, :], label[base_classes_idx])
                            losses.append(base_loss)

                        loss = 0
                        for l in losses: loss += l
                        loss /= len(losses)
                    elif self.args.joint_loss == "ce_weighted":
                        novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                        base_classes_idx = torch.argwhere(label < novel_class_start).flatten()

                        w_current = (1/(session+1))     # 0.5, 0.333, 0.25
                        w_replay = 1-w_current          # 0.5, 0.667, ...

                        loss = 0
                        if novel_classes_idx.numel() != 0:
                            novel_loss = criterion(logits[novel_classes_idx, :], label[novel_classes_idx])
                            loss += w_current * novel_loss

                        if base_classes_idx.numel() != 0:
                            base_loss = criterion(logits[base_classes_idx, :], label[base_classes_idx])
                            loss += w_replay * base_loss
                    else:
                        # Original ce
                        loss = criterion(logits, label) # Note, no label smoothing here
                
                ta.add(count_acc(logits, label))
                n,b = count_acc_(logits, label, test_class, self.args)
                taNovel.add(n)
                taBase.add(b)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Model Saving
            vl, va, vaNovel, vaBase, vhm, vam, vbin, vaNN, vaBB = self.test_fc_head(self.fc, testloader, epoch, session)
            if self.args.validation_metric == "hm":
                # Validation
                if best_hm is None or vhm > best_hm:
                    best_hm = vhm
                    best_fc = deepcopy(self.fc.state_dict())
                    hm_patience_count = 0
                else:
                    hm_patience_count += 1
                lrc = scheduler.get_last_lr()[0]
                out_string = '(Joint) Sess: {}, loss {:.3f}, trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}, (train) b/n:{:.3f}/{:.3f}, (test) b/n={:.3f}/{:.3f}, lrc={:.3f}'\
                    .format(
                        session, 
                        total_loss,
                        float('%.3f' % (ta.item() * 100.0)),
                        float("%.3f" % (va * 100.0)),
                        float("%.3f" % (best_hm * 100.0)),
                        float("%.3f" % (taBase.item() * 100.0)),
                        float("%.3f" % (taNovel.item() * 100.0)),
                        float("%.3f" % (vaBase * 100.0)),
                        float("%.3f" % (vaNovel * 100.0)),
                        float(lrc))
            tqdm_gen.set_description(out_string)

            if hm_patience_count > hm_patience:
                # faster joint session
                break

            scheduler.step()
    
    self.fc.load_state_dict(best_fc, strict=True)   

def update_fc_ft_joint_gaussian(self, jointloader, testloader, class_list, session, generated):
    """
        Extract the parameters associated with the given class list.
        So only base classifier and the novel classifier
        Creating a new classifier for the current classes and the new novel classes
        Note that this new classifier is a contiguous classifier so it needs to be trained with taregts greater 
        than 60 to be offset by the number of novel classes we have seen
    """

    gaus, aug_gaus = generated
    gaussian_data, gaussian_labels = gaus
    aug_gaus_data, aug_gaus_labels = aug_gaus

    optimizer = self.get_optimizer_new(self.projector.parameters())

    if self.args.joint_schedule == "Milestone":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80],gamma=self.args.gamma)
    else:
        warmup_epochs = 10
        min_lr = 0.0
        scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=10,
                max_epochs=self.args.epochs_joint,
                warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_new,
                eta_min=min_lr
            )

    # criterion = nn.CrossEntropyLoss()
    criterion = self.select_criterion()

    best_loss = None
    best_hm = None
    best_acc = None

    best_fc = None
    label_offset = 0
    if self.args.base_only:
        label_offset = (session - 1) * self.args.way

    hm_patience = self.args.hm_patience
    hm_patience_count = 0

    # shuffle generated data
    shuffler = np.arange(gaussian_data.shape[0])
    np.random.shuffle(shuffler)
    gaussian_data = gaussian_data[shuffler]
    gaussian_labels = gaussian_labels[shuffler]
    
    # split equally
    total_iter = jointloader.__len__()
    generated_data = np.array_split(gaussian_data, total_iter)
    generated_labels = np.array_split(gaussian_labels, total_iter)

    novel_class_start = self.args.base_class + (self.args.way * (session-1))
    test_class = self.args.base_class + session * self.args.way

    with torch.enable_grad():
        tqdm_gen = tqdm(range(self.args.epochs_joint))

        # To emulate augmentation in generated data we need to mixup the generated data
        # Basically a pair of samples from a single class will be averaged to create a new instance, the choice of pair is entirely random.

        for epoch in tqdm_gen:
            total_loss = 0
            ta = Averager()
            taNovel = Averager()
            taBase = Averager()
            # for i, batch in enumerate(tqdm_gen, 1):
            for i, (data, label) in enumerate(jointloader):
                data = data.cuda()
                label = label.cuda()
                label[label >= self.args.base_class] -= label_offset

                if self.args.instance_mixup:
                    # Instance mixing novel classes
                    data, label = instance_mixup_data(data, label)
                    data, label = map(Variable, (data, label))

                # encoding = self.encode(data).detach() #<<< The encoder is essentially frozen
                encoding = self.encode(data, detach_f=True)

                # Before concatenating generated label, each item gets augmented with a random sample from generated_data
                augmented_data = gaussian_utils.augmentMultivariateData(generated_data[i], generated_labels[i], aug_gaus_data, aug_gaus_labels)

                # Pass the augmented data into the projector
                augmented_data = torch.tensor(augmented_data).to(torch.float).cuda()
                augmented_encoding = self.projector(augmented_data)

                # Concatenating the augmented gaussian data and the current novel class encodings
                encoding = torch.cat((augmented_encoding, encoding)).to(torch.float)
                label = torch.cat((torch.tensor(generated_labels[i]).cuda(), label)).to(torch.long)

                # Get the cosine similarity to the classifier
                logits = self.fc(encoding)

                if self.args.mixup_joint:
                    # loss = mixup_criterion(F.cross_entropy, logits, targets_a, targets_b, lam)
                    # loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                    pass
                else:
                    if self.args.joint_loss in ['ce_even', 'ce_inter']:
                        losses = []
                        novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                        if self.args.joint_loss == "ce_even":
                            base_classes_idx = torch.argwhere(label < novel_class_start).flatten()
                        elif self.args.joint_loss == "ce_inter":
                            # TODO: For new sessions weight everybody together
                            inter_classes_idx = torch.argwhere((label >= self.args.base_class) & (label < novel_class_start)).flatten()
                            base_classes_idx = torch.argwhere(label < self.args.base_class).flatten()

                            if inter_classes_idx.numel() != 0:
                                inter_loss = criterion(logits[inter_classes_idx, :], label[inter_classes_idx])
                                losses.append(inter_loss)

                        if novel_classes_idx.numel() != 0:
                            # Loss computed using the novel classes
                            novel_loss = self.criterion_forward(criterion, logits[novel_classes_idx, :], label[novel_classes_idx])
                            losses.append(novel_loss)

                        if base_classes_idx.numel() != 0:
                            base_loss = self.criterion_forward(criterion, logits[base_classes_idx, :], label[base_classes_idx])
                            losses.append(base_loss)

                        loss = 0
                        loss_string = f"Losses =>"
                        idx2name = {0:"novel", 1:"base"}
                        for idx, l in enumerate(losses): 
                            loss += l
                            loss_string += f" loss {idx2name[idx]}: {loss:.3f},"
                        # tqdm_gen.set_description(loss_string)
                        loss /= len(losses)
                    elif self.args.joint_loss == "ce_weighted":
                        novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                        base_classes_idx = torch.argwhere(label < novel_class_start).flatten()

                        w_current = (1/(session+1))     # 0.5, 0.333, 0.25
                        w_replay = 1-w_current          # 0.5, 0.667, 0.75

                        loss = 0
                        if novel_classes_idx.numel() != 0:
                            novel_loss = criterion(logits[novel_classes_idx, :], label[novel_classes_idx])
                            loss += w_current * novel_loss

                        if base_classes_idx.numel() != 0:
                            base_loss = criterion(logits[base_classes_idx, :], label[base_classes_idx])
                            loss += w_replay * base_loss
                    else:
                        # Original ce
                        loss = criterion(logits, label) # Note, no label smoothing here
                
                ta.add(count_acc(logits, label))

                n,b = count_acc_(logits, label, test_class, self.args)
                taNovel.add(n)
                taBase.add(b)

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt.clip)
                optimizer.step()

                total_loss += loss.item()

            # Model Saving
            vl, va, vaNovel, vaBase, vhm, vam, vbin, vaNN, vaBB = self.test_fc_head(self.fc, testloader, epoch, session)
            if self.args.validation_metric == "hm":
                # Validation
                if best_hm is None or vhm > best_hm:
                    best_hm = vhm
                    # best_fc = concat_fc.clone()
                    # best_fc = concat_fc.detach().clone()
                    best_fc = deepcopy(self.projector.state_dict())
                    hm_patience_count = 0
                else:
                    hm_patience_count += 1
                lrc = scheduler.get_last_lr()[0]
                out_string = '(Joint) Sess: {}, loss {:.3f}, trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}, (train) b/n:{:.3f}/{:.3f}, (test) b/n={:.3f}/{:.3f}, lrc={:.3f}'\
                    .format(
                        session, 
                        total_loss,
                        float('%.3f' % (ta.item() * 100.0)),
                        float("%.3f" % (va * 100.0)),
                        float("%.3f" % (best_hm * 100.0)),
                        float("%.3f" % (taBase.item() * 100.0)),
                        float("%.3f" % (taNovel.item() * 100.0)),
                        float("%.3f" % (vaBase * 100.0)),
                        float("%.3f" % (vaNovel * 100.0)),
                        float(lrc))

            tqdm_gen.set_description(out_string)

            if hm_patience_count > hm_patience:
                # faster joint session
                break

            scheduler.step()
    
    self.projector.load_state_dict(best_fc, strict=True)    

def finetune_head(self, loader, epochs=None):
    # Only for testing purposes
    # fine tuning the head given some data and labels
    optimizer = self.get_optimizer_new(self.fc.parameters())

    with torch.enable_grad():

        loss_before = 0
        for i, batch in enumerate(loader):
            data, label = [_.cuda() for _ in batch]
            encoding = self.encode(data).detach()
            logits = self.fc.get_logits(encoding)
            loss_before += F.cross_entropy(logits, label)
        loss_before /= len(loader)
        print(f"Loss at the beginning of training: {loss_before.item()}")

        tqdm_gen = tqdm(range(self.args.epochs_new))
        for epoch in tqdm_gen:
            for i, batch in enumerate(loader):
                data, label = [_.cuda() for _ in batch]
                encoding = self.encode(data).detach()
                logits = self.fc.get_logits(encoding)

                loss = F.cross_entropy(logits, label) # Technically this is base normalised cross entropy. Because denominator has base sum and targets are only for the novel classes

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tqdm_gen.set_description(f"Loss: {loss.item():.3f}")

def update_fc_gm(self,data,label,class_list):
        """
            Using the exemplars available during training to instantiate the classifier for the novel setting
        """
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            # proto=embedding.mean(0)     # Mean along the 0th axis of all the embeddings for this particular class index
            # Calculate geometric median instead
            proto = compute_geometric_median(embedding.cpu()).median.cuda()
            # proto = compute_geometric_mean(embedding)

            new_fc.append(proto)

        # Note the original protonet sums the latent vectors over the support set, and divides by the number of classes. Not by the number of data points
        if not self.args.skip_encode_norm:
            new_fc_tensor = F.normalize(torch.stack(new_fc,dim=0), dim=1)
        else:
            new_fc_tensor = torch.stack(new_fc,dim=0)

        new_fc = nn.Linear(new_fc_tensor.shape[1], new_fc_tensor.shape[0]).cuda()
        new_fc.weight.data.copy_(new_fc_tensor)
        # new_fc.weight.requires_grad = False
        return new_fc