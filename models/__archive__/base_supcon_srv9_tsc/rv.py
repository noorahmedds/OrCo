class RV_Finder():
    def __init__(self, n_inc_classes, num_features):
        self.rv = None

    def find_reseverve_vectors(self, base_fc):
        self.radius = 1.0
        self.temperature = 0.5
        
        base_prototypes = normalize(base_fc.weight.data)
        points = torch.randn(self.n_inc_classes, self.num_features).cuda()
        points = normalize(points)
        points = torch.nn.Parameter(points)
        opt = torch.optim.SGD([points], lr=5)
        
        best_angle = 0
        tqdm_gen = tqdm(range(2000))

        for _ in tqdm_gen:
            # Combining prototypes but only optimising the reserve vector
            comb = torch.cat((points, base_prototypes), axis = 0)

            # Compute the cosine similarity.
            sim = F.cosine_similarity(comb[None,:,:], comb[:,None,:], dim=-1)
            l = torch.log(torch.exp(sim/self.temperature).sum(axis = 1)).sum() / comb.shape[0]
            
            # opt.zero_grad()
            l.backward()
            opt.step()
            points.data = normalize(points.data)

            curr_angle, curr_angle_close = compute_angles(torch.cat((points, base_prototypes), axis = 0).detach())
            if curr_angle > best_angle: # best angle is the angle with which the separation is maximised
                best_angle = curr_angle

            tqdm_gen.set_description(f"Loss = {l:.5f}, Best Avg Angle (deg): {best_angle:.3f}, Average Angle rv+base [close]: {curr_angle_close:.3f}")

        # Setting Reserved vectors
        self.rv = points.data

    def find_reseverve_vectors_all(self):
        self.temperature = 1.0
        
        points = torch.randn(self.num_classes, self.num_features).cuda()
        points = normalize(points)
        points = torch.nn.Parameter(points)
        opt = torch.optim.SGD([points], lr=1)
        
        best_angle = 0
        tqdm_gen = tqdm(range(self.args.epochs_simplex))

        for _ in tqdm_gen:
            # Compute the cosine similarity.
            sim = F.cosine_similarity(points[None,:,:], points[:,None,:], dim=-1)
            l = torch.log(torch.exp(sim/self.temperature).sum(axis = 1)).sum() / points.shape[0]
            
            # opt.zero_grad()
            l.backward()
            opt.step()
            points.data = normalize(points.data)

            curr_angle, curr_angle_close = compute_angles(points.detach())
            if curr_angle > best_angle: # best angle is the angle with which the separation is maximised
                best_angle = curr_angle

            tqdm_gen.set_description(f"Loss = {l:.5f}, Best Avg Angle (deg): {best_angle:.3f}, Average Angle rv+base [close]: {curr_angle_close:.3f}")

        # Setting Reserved vectors
        self.rv = points.data
        # self.register_buffer('rv', points.data)

    def find_reseverve_vectors_two_step(self, proto):
        self.temperature = 1.0
        
        proto = normalize(proto)
        points = torch.randn(self.n_inc_classes, self.num_features).cuda()
        points = normalize(points)
        points = torch.nn.Parameter(points)
        opt = torch.optim.SGD([points], lr=1)
        best_angle = 0
        tqdm_gen = tqdm(range(1000))
        print("(Simplex search) Optimising the randn to be far away from base prototype")
        for _ in tqdm_gen:
            comb = torch.cat((proto, points), axis = 0)
            # Compute the cosine similarity.
            sim = F.cosine_similarity(comb[None,:,:], comb[:,None,:], dim=-1)
            l = torch.log(torch.exp(sim/self.temperature).sum(axis = 1)).sum() / comb.shape[0]
            l.backward()
            opt.step()
            points.data = normalize(points.data)

            curr_angle, curr_angle_close = compute_angles(points.detach())
            if curr_angle > best_angle: # best angle is the angle with which the separation is maximised
                best_angle = curr_angle
            tqdm_gen.set_description(f"Loss = {l:.5f}, Best Avg Angle (deg): {best_angle:.3f}, Average Angle rv+base [close]: {curr_angle_close:.3f}")

        # proto = torch.nn.Parameter(proto)
        # points = torch.cat((proto, points), axis = 0)
        points = torch.randn(self.num_classes, self.num_features).cuda()
        points.data = torch.cat((proto, points), axis = 0)
        points = torch.nn.Parameter(points)
        opt = torch.optim.SGD([points], lr=1)
        tqdm_gen = tqdm(range(10000))
        print("(Simplex search) Optimising everything together")
        for _ in tqdm_gen:
            # Compute the cosine similarity.
            sim = F.cosine_similarity(points[None,:,:], points[:,None,:], dim=-1)
            l = torch.log(torch.exp(sim/self.temperature).sum(axis = 1)).sum() / points.shape[0]
            l.backward()
            opt.step()
            points.data = normalize(points.data)

            curr_angle, curr_angle_close = compute_angles(points.detach())
            if curr_angle > best_angle: # best angle is the angle with which the separation is maximised
                best_angle = curr_angle
            tqdm_gen.set_description(f"Loss = {l:.5f}, Best Avg Angle (deg): {best_angle:.3f}, Average Angle rv+base [close]: {curr_angle_close:.3f}")

        # Setting Reserved vectors
        self.rv = points.data