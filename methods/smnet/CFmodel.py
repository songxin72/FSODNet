import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet
import einops
import numpy as np
mode = 'bilinear'  # 'nearest' #
from torch.nn.parameter import Parameter
import scipy.stats as st
import copy


def up_conv(cin, cout, up=True):
    yield nn.Conv2d(cin, cout, 3, padding=1)
    yield nn.GroupNorm(cout // 2, cout)
    yield nn.ReLU(inplace=True)
    if up:
        yield nn.Upsample(scale_factor=2, mode='bilinear')


def local_conv(cin, cout):
    yield nn.Conv2d(cin, cout * 2, 3, padding=1)
    yield nn.GroupNorm(cout, cout * 2)
    yield nn.ReLU(inplace=True)
    yield nn.Upsample(scale_factor=2, mode='bilinear')
    yield nn.Conv2d(cout * 2, cout, 3, padding=1)
    yield nn.GroupNorm(cout // 2, cout)
    yield nn.ReLU(inplace=True)
    yield nn.Upsample(scale_factor=2, mode='bilinear')


class RAM(nn.Module):
    def __init__(self, config, feat, tar_feat):
        super(RAM, self).__init__()
        # self.conv2 = nn.Sequential(*list(up_conv(feat[1], tar_feat)))
        # self.conv1 = nn.Sequential(*list(up_conv(feat[0], tar_feat, False)))
        self.gconv = nn.Sequential(*list(up_conv(tar_feat, tar_feat, False)))
        # self.conv0 = nn.Sequential(*list(up_conv(tar_feat * 3, tar_feat, False)))
        self.res_conv1 = nn.Conv2d(tar_feat, tar_feat, 3, padding=1)
        self.res_conv2 = nn.Conv2d(tar_feat, tar_feat, 3, padding=1)

        self.fuse = nn.Conv2d(tar_feat * 3, tar_feat, 3, padding=1)

    def forward(self, xs, glob_x):
        glob_x0 = nn.functional.interpolate(self.gconv(glob_x), size=xs[0].size()[2:], mode=mode)

        loc_x1 = xs[0]
        res_x1 = torch.sigmoid(self.res_conv1(loc_x1 - glob_x0))  #
        loc_x2 = nn.functional.interpolate(xs[1], size=xs[0].size()[2:], mode=mode)
        res_x2 = torch.sigmoid(self.res_conv2(loc_x2 - glob_x0))  #
        loc_x = self.fuse(torch.cat([loc_x1 * res_x1, loc_x2 * res_x2, glob_x0], dim=1))

        return loc_x, res_x1, res_x2


class decoder_pn(nn.Module):
    def __init__(self, config, encoder, feat):
        super(decoder_pn, self).__init__()

        self.adapter0 = nn.Sequential(*list(up_conv(feat[0], feat[0], False)))
        self.adapter1 = nn.Sequential(*list(up_conv(feat[1], feat[0], False)))
        self.adapter2 = nn.Sequential(*list(up_conv(feat[2], feat[0], False)))
        self.adapter3 = nn.Sequential(*list(up_conv(feat[3], feat[0], False)))
        self.adapter4 = nn.Sequential(*list(up_conv(feat[4], feat[0], False)))
        self.feat = feat[1]
        self.region = RAM(config, feat[2:4], feat[0])
        self.local = RAM(config, feat[0:2], feat[0])

        self.gb_conv = nn.Sequential(*list(local_conv(feat[0], feat[0])))

    def forward(self, xs, h, x_size):
        xs[4] = h[0].unsqueeze(0)
        xs[0] = self.adapter0(xs[0])
        xs[1] = self.adapter1(xs[1])
        xs[2] = self.adapter2(xs[2])
        xs[3] = self.adapter3(xs[3])
        xs[4] = self.adapter4(xs[4])

        glob_x = xs[4]
        reg_x, r3, r4 = self.region(xs[2:4], glob_x)

        glob_x = self.gb_conv(glob_x)
        loc_x, r1, r2 = self.local(xs[0:2], glob_x)

        reg_x = nn.functional.interpolate(reg_x, size=xs[0].size()[2:], mode=mode)
        pred = torch.sum(loc_x * reg_x, dim=1, keepdim=True)
        pred = nn.functional.interpolate(pred, size=x_size, mode='bilinear')

        back = h[1].unsqueeze(0)
        back = self.adapter4(back)
        glob_x = back
        reg_x, r3, r4 = self.region(xs[2:4], glob_x)
        glob_x = self.gb_conv(glob_x)
        loc_x, r1, r2 = self.local(xs[0:2], glob_x)
        reg_x = nn.functional.interpolate(reg_x, size=xs[0].size()[2:], mode=mode)
        pred2 = torch.sum(loc_x * reg_x, dim=1, keepdim=True)
        pred2 = nn.functional.interpolate(pred2, size=x_size, mode='bilinear')

        OutDict = {}
        OutDict['sal'] = [pred, ]
        OutDict['final'] = pred

        return OutDict, pred, pred2


class decoder(nn.Module):
    def __init__(self, config, encoder, feat):
        super(decoder, self).__init__()

        self.adapter0 = nn.Sequential(*list(up_conv(feat[0], feat[0], False)))
        self.adapter1 = nn.Sequential(*list(up_conv(feat[1], feat[0], False)))
        self.adapter2 = nn.Sequential(*list(up_conv(feat[2], feat[0], False)))
        self.adapter3 = nn.Sequential(*list(up_conv(feat[3], feat[0], False)))
        self.adapter4 = nn.Sequential(*list(up_conv(feat[4], feat[0], False)))
        self.feat = feat[1]
        self.region = RAM(config, feat[2:4], feat[0])
        self.local = RAM(config, feat[0:2], feat[0])

        self.gb_conv = nn.Sequential(*list(local_conv(feat[0], feat[0])))

    def forward(self, xs, h, x_size):
        xs[4] = h[0].unsqueeze(0)
        xs[0] = self.adapter0(xs[0])
        xs[1] = self.adapter1(xs[1])
        xs[2] = self.adapter2(xs[2])
        xs[3] = self.adapter3(xs[3])
        xs[4] = self.adapter4(xs[4])

        glob_x = xs[4]
        reg_x, r3, r4 = self.region(xs[2:4], glob_x)

        glob_x = self.gb_conv(glob_x)
        loc_x, r1, r2 = self.local(xs[0:2], glob_x)

        reg_x = nn.functional.interpolate(reg_x, size=xs[0].size()[2:], mode=mode)
        pred = torch.sum(loc_x * reg_x, dim=1, keepdim=True)
        pred = nn.functional.interpolate(pred, size=x_size, mode='bilinear')

        back = h[1].unsqueeze(0)
        back = self.adapter4(back)
        glob_x = back
        reg_x, r3, r4 = self.region(xs[2:4], glob_x)
        glob_x = self.gb_conv(glob_x)
        loc_x, r1, r2 = self.local(xs[0:2], glob_x)
        reg_x = nn.functional.interpolate(reg_x, size=xs[0].size()[2:], mode=mode)
        pred2 = torch.sum(loc_x * reg_x, dim=1, keepdim=True)
        pred2 = nn.functional.interpolate(pred2, size=x_size, mode='bilinear')

        OutDict = {}
        OutDict['sal'] = [pred, ]
        OutDict['final'] = pred

        return OutDict, pred, pred2


def build_grid(resolution):
    device = 'cpu'  #'cuda' if torch.cuda.is_available() else 'cpu'
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.tensor(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)


class SoftPositionEmbed(nn.Module):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.

        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
        """
        super(SoftPositionEmbed, self).__init__()
        self.proj = nn.Linear(4, hidden_size)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        return inputs + self.proj(self.grid)


def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid and collapse slot dimension."""
    # `slots` has shape: [batch_size, num_slots, slot_size].
    slots = torch.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
    grid = einops.repeat(slots, 'b_n i j d -> b_n (tilei i) (tilej j) d', tilei=resolution[0], tilej=resolution[1])
    # `grid` has shape: [batch_size*num_slots, height, width, slot_size].
    return grid


def spatial_flatten(x):
    return torch.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[-1]])


def unstack_and_split(x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask."""
    unstacked = einops.rearrange(x, '(b s) c h w -> b s c h w', b=batch_size)
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=2)
    return channels, masks


class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_slots, encoder_dims, iters=3, hidden_dim=128, eps=1e-8):
        """Builds the Slot Attention module.
        Args:
            iters: Number of iterations.
            num_slots: Number of slots.
            encoder_dims: Dimensionality of slot feature vectors.
            hidden_dim: Hidden layer size of MLP.
            eps: Offset for attention coefficients before normalization.
        """
        super(SlotAttention, self).__init__()

        self.eps = eps
        self.iters = iters
        self.num_slots = num_slots
        self.scale = encoder_dims ** -0.5
        self.device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)
        self.slots_embedding = nn.Embedding(num_slots, encoder_dims)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(encoder_dims, encoder_dims)
        self.project_k = nn.Linear(encoder_dims, encoder_dims)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)

        # Slot update functions.
        self.gru = nn.GRUCell(encoder_dims, encoder_dims)

        hidden_dim = max(encoder_dims, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, encoder_dims)
        )

    def forward(self, inputs, num_slots=None):
        # inputs has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_input(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        slots = self.slots_embedding(torch.arange(0, n_s).expand(b, n_s).to(self.device))

        # Multiple rounds of attention.
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.

            updates = torch.einsum('bjd,bij->bid', v, attn)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


def gkern(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)


class HA(nn.Module):
    # holistic attention module
    def __init__(self):
        super(HA, self).__init__()
        gaussian_kernel = np.float32(gkern(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention, x):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)  # HA注意力操作
        soft_attention = min_max_norm(soft_attention)

        Soft_Att= soft_attention.max(attention)
        zero = torch.zeros_like(Soft_Att)
        one = torch.ones_like(Soft_Att)

        Soft_Att = torch.tensor(torch.where(Soft_Att > 0.05, one, Soft_Att))
        Soft_Att = torch.tensor(torch.where(Soft_Att <=0.05, zero, Soft_Att))

        Depth_pos = torch.mul(x, Soft_Att)  # 像素级相乘
        Depth_neg = torch.mul(x, 1 - Soft_Att)

        return Depth_pos, Depth_neg


class Residual_Block(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
    ):
        super(Residual_Block, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)


    def forward(self, x) :
        x = self.conv(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class MSModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(MSModule, self).__init__()
        self.s_conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())
        self.s_conv_2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())
        self.m_conv_1 = nn.Sequential(nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())
        self.m_conv_2 = nn.Sequential(nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())

        self.s_conv_3 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())

        self.m_conv_3 = nn.Sequential(nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())

        self.s_residual_block = Residual_Block(out_channels, out_channels)
        self.m_residual_block = Residual_Block(out_channels*4, out_channels)

    def forward(self, img_feature, label_feature1, label_feature2, label_feature3):

        label_feature1 = self.s_conv_1(label_feature1)
        label_feature2 = self.s_conv_2(label_feature2)
        label_feature3 = self.s_conv_3(label_feature3)

        m = torch.cat((self.s_residual_block(img_feature), label_feature1, label_feature2, label_feature3), dim=1)
        m = self.m_residual_block(m)
        s1 = self.m_conv_1(torch.cat((m, label_feature1), dim=1))

        s2 = self.m_conv_2(torch.cat((m, label_feature2), dim=1))
        s3 = self.m_conv_3(torch.cat((m, label_feature3), dim=1))
        return s1, s2, s3


class MSModule2(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(MSModule2, self).__init__()
        self.s_conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())
        self.s_conv_2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())
        self.m_conv_1 = nn.Sequential(nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())
        self.m_conv_2 = nn.Sequential(nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())


        self.s_residual_block = Residual_Block(out_channels, out_channels)
        self.m_residual_block = Residual_Block(out_channels*3, out_channels)

    def forward(self, img_feature, label_feature1, label_feature2):

        label_feature1 = self.s_conv_1(label_feature1)
        label_feature2 = self.s_conv_2(label_feature2)

        m = torch.cat((self.s_residual_block(img_feature), label_feature1, label_feature2), dim=1)
        m = self.m_residual_block(m)
        s1 = self.m_conv_1(torch.cat((m, label_feature1), dim=1))

        s2 = self.m_conv_2(torch.cat((m, label_feature2), dim=1))

        return s1, s2


class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()

        self.encoder = encoder
        self.decoder = decoder(config, encoder, feat)
        self.decoder2 = decoder_pn(config, encoder, feat)
        self.encoder_dims = 128

        self.slot_attention = SlotAttention(
            iters=5,
            num_slots=2,
            encoder_dims=self.encoder_dims,
            hidden_dim=self.encoder_dims)
        # self.decoder_pos = SoftPositionEmbed(128, self.decoder_initial_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_dims, self.encoder_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dims, self.encoder_dims)
        )
        self.layer_norm = nn.LayerNorm(self.encoder_dims)

        self.slot_attention2 = SlotAttention(
            iters=5,
            num_slots=2,
            encoder_dims=self.encoder_dims,
            hidden_dim=self.encoder_dims)
        self.mlp2 = nn.Sequential(
            nn.Linear(self.encoder_dims, self.encoder_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dims, self.encoder_dims)
        )
        self.layer_norm2 = nn.LayerNorm(self.encoder_dims)

        self.HA = HA()
        # self.HA = HA()

        self.ms1 = MSModule2(1, 16, stride=1)
        self.ms2 = MSModule2(16, 16, stride=1)
        self.ms3 = MSModule2(16, 16, stride=1)
        self.ms4 = MSModule2(16, 16, stride=1)
        self.ms5 = MSModule2(16, 16, stride=1)

        self.ms6 = MSModule2(1, 16, stride=1)
        self.ms7 = MSModule2(16, 16, stride=1)
        self.ms8 = MSModule2(16, 16, stride=1)
        self.ms9 = MSModule2(16, 16, stride=1)
        self.ms10 = MSModule2(16, 16, stride=1)

        self.uc1 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1), nn.ReLU())
        self.uc2 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1), nn.ReLU())
        self.uc3 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1), nn.ReLU())

        self.uc4 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1), nn.ReLU())
        self.uc5 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1), nn.ReLU())
        self.uc6 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1), nn.ReLU())

    def forward(self, x, x1, label1, label2, label4):
        x_size = x.size()[2:]
        fused_image, vis_rec, inf_rec, xs, xi = self.encoder(x, x1)  # 输出的格式：[batch, 128, 576, 768]

        z = einops.rearrange(xs[4], 'b c h w -> b h w c')
        encoder_pos2 = SoftPositionEmbed(self.encoder_dims, (xs[4].shape[2], xs[4].shape[3]))  #.cuda()
        z = encoder_pos2(z)  # Position embedding.  格式：[b, h, w, c]
        z = spatial_flatten(z)  # Flatten spatial dimensions (treat image as set).
        z = self.mlp2(self.layer_norm2(z))  # Feedforward network on set. 得到特征图 格式[batch, f1_dim, f2_dim]
        # Slot Attention module.
        slots = self.slot_attention2(z)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # Spatial broadcast decoder.
        xz = spatial_broadcast(slots, (xs[4].shape[2], xs[4].shape[3]))  # 格式[2, 442368, 128, 128]
        # `x` has shape: [batch_size*num_slots, height_init, width_init, slot_size].
        decoder_pos2 = SoftPositionEmbed(self.encoder_dims, (xs[4].shape[2], xs[4].shape[3]))  #.cuda()
        xz = decoder_pos2(xz)  # 将slot输入到编码器中
        h = einops.rearrange(xz, 'b_n h w c -> b_n c h w')
        v_out, v_pos, v_neg = self.decoder(xs, h, x_size)

        z = einops.rearrange(xi[4], 'b c h w -> b h w c')
        encoder_pos = SoftPositionEmbed(self.encoder_dims, (xi[4].shape[2], xi[4].shape[3]))  #.cuda()
        z = encoder_pos(z)  # Position embedding.  格式：[b, h, w, c]
        z = spatial_flatten(z)  # Flatten spatial dimensions (treat image as set).
        z = self.mlp(self.layer_norm(z))  # Feedforward network on set. 得到特征图 格式[batch, f1_dim, f2_dim]
        # Slot Attention module.
        slots = self.slot_attention(z)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # Spatial broadcast decoder.
        xz = spatial_broadcast(slots, (xi[4].shape[2], xi[4].shape[3]))  # 格式[2, 442368, 128, 128]
        # `x` has shape: [batch_size*num_slots, height_init, width_init, slot_size].
        decoder_pos = SoftPositionEmbed(self.encoder_dims, (xi[4].shape[2], xi[4].shape[3]))  #.cuda()
        xz = decoder_pos(xz)  # 将slot输入到编码器中
        h = einops.rearrange(xz, 'b_n h w c -> b_n c h w')
        i_out, i_pos, i_neg = self.decoder(xi, h, x_size)

        if (label1 is not None) and (label2 is not None):
            l_1, l_2 = self.ms1(F.interpolate(xs[0], size=x_size, mode="bilinear", align_corners=True),
                           label1, label2)
            l_1, l_2 = self.ms2(F.interpolate(xs[1], size=x_size, mode="bilinear", align_corners=True),
                           l_1, l_2)
            l_1, l_2 = self.ms3(F.interpolate(xs[2], size=x_size, mode="bilinear", align_corners=True),
                           l_1, l_2)
            l_1, l_2 = self.ms4(F.interpolate(xs[3], size=x_size, mode="bilinear", align_corners=True),
                           l_1, l_2)
            l_1, l_2 = self.ms5(F.interpolate(xs[4], size=x_size, mode="bilinear", align_corners=True),
                           l_1, l_2)
            l_1 = self.uc1(F.interpolate(l_1, scale_factor=1, mode="bilinear", align_corners=True))
            l_2 = self.uc2(F.interpolate(l_2, scale_factor=1, mode="bilinear", align_corners=True))
            # l_3 = self.uc3(F.interpolate(l_3, scale_factor=1, mode="bilinear", align_corners=True))

            l_4, l_5 = self.ms6(F.interpolate(xi[0], size=x_size, mode="bilinear", align_corners=True),
                           label4, label2)
            l_4, l_5 = self.ms7(F.interpolate(xi[1], size=x_size, mode="bilinear", align_corners=True),
                           l_4, l_5)
            l_4, l_5 = self.ms8(F.interpolate(xi[2], size=x_size, mode="bilinear", align_corners=True),
                           l_4, l_5)
            l_4, l_5 = self.ms9(F.interpolate(xi[3], size=x_size, mode="bilinear", align_corners=True),
                           l_4, l_5)
            l_4, l_5 = self.ms10(F.interpolate(xi[4], size=x_size, mode="bilinear", align_corners=True),
                           l_4, l_5)
            l_4 = self.uc4(F.interpolate(l_4, scale_factor=1, mode="bilinear", align_corners=True))
            l_5 = self.uc5(F.interpolate(l_5, scale_factor=1, mode="bilinear", align_corners=True))
            # l_6 = self.uc6(F.interpolate(l_6, scale_factor=1, mode="bilinear", align_corners=True))
            return i_out, v_out, v_pos, v_neg, i_pos, i_neg, l_1, l_2, l_4, l_5

        return i_out, v_out, v_pos, v_neg, i_pos, i_neg

