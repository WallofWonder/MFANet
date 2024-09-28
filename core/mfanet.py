import torch
import torch.nn as nn


class GridPosition(nn.Module):
    def __init__(self, grid_num, use_gpu=True):
        nn.Module.__init__(self)
        self.grid_num = grid_num  # 16
        self.use_gpu = use_gpu

    def forward(self, batch_size):
        grid_center_x = torch.linspace(-1. + 2. / self.grid_num / 2, 1. - 2. / self.grid_num / 2,
                                       steps=self.grid_num).cuda() if self.use_gpu else torch.linspace(
            -1. + 1. / self.grid_num / 2, 1. - 1. / self.grid_num / 2, steps=self.grid_num)
        grid_center_y = torch.linspace(1. - 2. / self.grid_num / 2, -1. + 2. / self.grid_num / 2,
                                       steps=self.grid_num).cuda() if self.use_gpu else torch.linspace(
            1. - 1. / self.grid_num / 2, -1. + 1. / self.grid_num / 2, steps=self.grid_num)
        # BCHW, (b,:,h,w)->(x,y)
        grid_center_position_mat = torch.reshape(
            torch.cartesian_prod(grid_center_x, grid_center_y),  # 256,2
            (1, self.grid_num, self.grid_num, 2)  # 1,16,16,2
        ).permute(0, 3, 2, 1).contiguous()  # 1,2,16,16
        # BCN, (b,:,n)->(x,y), left to right then up to down
        grid_center_position_seq = grid_center_position_mat.reshape(1, 2, self.grid_num * self.grid_num)  # 1,2,256
        return grid_center_position_seq.repeat(batch_size, 1, 1)  # bs,2,256


class AttentionPropagation(nn.Module):
    def __init__(self, channels, head):
        nn.Module.__init__(self)
        self.head = head
        self.head_dim = channels // head
        self.query_filter, self.key_filter, self.value_filter = nn.Conv1d(channels, channels, kernel_size=1), \
            nn.Conv1d(channels, channels, kernel_size=1), \
            nn.Conv1d(channels, channels, kernel_size=1)
        self.mh_filter = nn.Conv1d(channels, channels, kernel_size=1)
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2 * channels, 2 * channels, kernel_size=1),
            nn.BatchNorm1d(2 * channels), nn.ReLU(),
            nn.Conv1d(2 * channels, channels, kernel_size=1),
        )

    def forward(self, motion1, motion2):
        # motion1(q) attend to motion(k,v)
        batch_size = motion1.shape[0]
        query, key, value = self.query_filter(motion1).view(batch_size, self.head, self.head_dim, -1), \
            self.key_filter(motion2).view(batch_size, self.head, self.head_dim, -1), \
            self.value_filter(motion2).view(batch_size, self.head, self.head_dim, -1)
        score = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', query, key) / self.head_dim ** 0.5, dim=-1)
        add_value = torch.einsum('bhnm,bhdm->bhdn', score, value).reshape(batch_size, self.head_dim * self.head, -1)
        add_value = self.mh_filter(add_value)
        motion1_new = motion1 + self.cat_filter(torch.cat([motion1, add_value], dim=1))
        return motion1_new


class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv1d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.InstanceNorm1d(channels, eps=1e-3),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, out_channels, kernel_size=1),
            nn.InstanceNorm1d(out_channels, eps=1e-3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class Pool(nn.Module):
    def __init__(self, channels, head, k, drop_p=0.):
        nn.Module.__init__(self)
        self.k = k

        self.init_filter = PointCN(channels)
        self.drop = nn.Dropout(p=drop_p) if drop_p > 0 else nn.Identity()
        self.proj = nn.Linear(channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.pool = AttentionPropagation(channels, head)

    def top_k_graph(self, scores, x, k):
        # scores: BN x: BCN
        x = self.init_filter(x)
        num_nodes = x.shape[-1]
        num_sampled_nodes = int(k * num_nodes)
        values, idx = torch.topk(scores, num_sampled_nodes, dim=-1)
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
            values = values.unsqueeze(0)
        idx_gather = idx.unsqueeze(1).repeat(1, x.shape[1], 1)  # BK->BCK
        x_new = torch.gather(x, 2, idx_gather)  # BCK
        values = values.unsqueeze(1)
        x_new = torch.mul(x_new, values.repeat(1, x.shape[1], 1))
        return x_new, idx

    def forward(self, x):
        # x: BCN
        Z = self.drop(x).permute(0, 2, 1).contiguous()  # BNC
        weights = self.proj(Z).squeeze()  # BN
        scores = self.sigmoid(weights)  # BN
        x_new, idx = self.top_k_graph(scores, x, self.k)  # x_new: BCK
        x_new = self.pool(x_new, x)  # BCK
        return x_new


class ADJ(nn.Module):
    def __init__(self, channels, head, up_sample=False):
        nn.Module.__init__(self)
        self.att = AttentionPropagation(channels, head)
        self.up_sample = up_sample

    def forward(self, x, x_ori):
        # x: BCK, x_ori: BCN
        if self.up_sample:
            x_new = self.att(x, x_ori)
        else:
            x_new = self.att(x_ori, x)  # BCN
        return x_new

    class PointCN(nn.Module):
        def __init__(self, channels, out_channels=None):
            nn.Module.__init__(self)
            if not out_channels:
                out_channels = channels
            self.shot_cut = None
            if out_channels != channels:
                self.shot_cut = nn.Conv1d(channels, out_channels, kernel_size=1)
            self.conv = nn.Sequential(
                nn.InstanceNorm1d(channels, eps=1e-3),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels, out_channels, kernel_size=1),
                nn.InstanceNorm1d(out_channels, eps=1e-3),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, kernel_size=1),
            )

        def forward(self, x):
            out = self.conv(x)
            if self.shot_cut:
                out = out + self.shot_cut(x)
            else:
                out = out + x
            return out


class PositionEncoder(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.position_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, channels, kernel_size=1)
        )

    def forward(self, x):
        return self.position_encoder(x)


class InitProject(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.init_project = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, channels, kernel_size=1)
        )

    def forward(self, x):
        return self.init_project(x)


class InlinerPredictor(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.inlier_pre = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=1), nn.InstanceNorm1d(64, eps=1e-3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 16, kernel_size=1), nn.InstanceNorm1d(16, eps=1e-3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 4, kernel_size=1), nn.InstanceNorm1d(4, eps=1e-3), nn.BatchNorm1d(4), nn.ReLU(),
            nn.Conv1d(4, 1, kernel_size=1)
        )

    def forward(self, d):
        # BCN -> B1N
        return self.inlier_pre(d)


class LayerBlock(nn.Module):
    def __init__(self, channels, head, grid_nums, sample_rates, up_sample=False):
        nn.Module.__init__(self)
        # assert len(sample_rates) == len(grid_nums) - 1
        self.grid_nums = grid_nums
        self.sample_rates = sample_rates
        self.pools = nn.ModuleList()
        self.adj = nn.ModuleList()
        self.grid_centers = []
        self.grid_pos_embeds = nn.ModuleList()

        for i in range(len(grid_nums)):
            self.grid_centers.append(GridPosition(grid_num=grid_nums[i], use_gpu=True))
            self.grid_pos_embeds.append(PositionEncoder(channels))
            if i != len(grid_nums) - 1:
                self.adj.append(ADJ(channels=channels, head=head, up_sample=up_sample))

        for i in range(len(sample_rates)):
            self.pools.append(Pool(channels=channels, head=head, k=sample_rates[i]))

        self.att_bot = AttentionPropagation(channels=channels, head=head)
        self.dealign = AttentionPropagation(channels, head)
        self.inlier_pre = InlinerPredictor(channels)

    def forward(self, xs, d):
        # xs: Correspondences bs,1,n,4
        # d: Motions bs,128,n
        batch_size = xs.shape[0]
        grid_pos_embeds = []
        for i in range(len(self.grid_nums)):
            grid_pos_embeds.append(self.grid_pos_embeds[i](self.grid_centers[i](batch_size)))
        # Filter motions
        d_ds = d
        for i in range(len(self.sample_rates)):
            # d_list.append(d_ds)
            d_ds = self.pools[i](d_ds)
        # estimate the motion field
        grid_pos_embeds[0] = self.att_bot(grid_pos_embeds[0], d_ds)
        # adjust the motion field
        for i in range(len(self.grid_nums) - 1):
            grid_pos_embeds[i + 1] = self.adj[i](grid_pos_embeds[i + 1], grid_pos_embeds[i])

        d_new = self.dealign(d, grid_pos_embeds[-1])
        # BCN -> B1N -> BN
        logits = torch.squeeze(self.inlier_pre(d_new - d), 1)
        e_hat = weighted_8points(xs, logits)
        return d_new, logits, e_hat


class MFANet(nn.Module):
    def __init__(self, config, use_gpu=True):
        nn.Module.__init__(self)
        self.layer_num = config.layer_num

        self.pos_embed = PositionEncoder(config.net_channels)
        self.init_project = InitProject(config.net_channels)
        self.layer_blocks = nn.Sequential(
            *[LayerBlock(config.net_channels, config.head, config.grid_nums, config.sample_rates, config.up_sample) for
              _ in range(self.layer_num)]
        )

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        # B1NC -> BCN
        input = data['xs'].transpose(1, 3).contiguous().squeeze(3)  # bs,4,n
        x1, x2 = input[:, :2, :], input[:, 2:, :]  # bs,2,n
        motion = x2 - x1  # 运动向量 bs,2,n

        pos = x1  # B2N

        pos_embed = self.pos_embed(pos)  # bs,128,n

        d = self.init_project(motion) + pos_embed  # bs,128,n

        res_logits, res_e_hat = [], []
        for i in range(self.layer_num):
            d, logits, e_hat = self.layer_blocks[i](data['xs'], d)  # BCN
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)

    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


if __name__ == '__main__':  # Just for debugging
    from config import get_config

    config, unparsed = get_config()
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    x = torch.randn(16, 1, 2000, 4).cuda()
    # y = torch.randn(32, 2000).cuda()
    data = {'xs': x}

    net = MFANet(config).cuda()
    res_logits, res_e_hat = net(data)
    print("done")
