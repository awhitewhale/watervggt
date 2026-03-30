import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from PIL import Image
from adamp import AdamP
from dataset_all import TestData
from attention import NonLocalSparseAttention
from deform_conv import DCN_layer
import torch
import random
from math import log10
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)


# recommend
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        # m.weight.data.normal_(0, 0.02)
        # m.bias.data.zero_()
        # nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal(m.weight.data, mode='fan_out')
        # nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_psnr(J, gt):
    mse = F.mse_loss(J, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def create_emamodel(net, ema=True):
    if ema:
        for param in net.parameters():
            param.detach_()
    return net


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)
    recoverd = recoverd.transpose(0, 2, 3, 1)  
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(recoverd.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=1, multichannel=True)

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]

class SFT_layer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(SFT_layer, self).__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )

    def forward(self, x, inter):
        '''
        :param x: degradation representation: B * C
        :param inter: degradation intermediate representation map: B * C * H * W
        '''
        gamma = self.conv_gamma(inter)
        beta = self.conv_beta(inter)

        return x * gamma + beta


class IGM(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size):
        super(IGM, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.dcn = DCN_layer(self.channels_in, self.channels_out, kernel_size,
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sft = SFT_layer(self.channels_in, self.channels_out)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :inter: degradation map: B * C * H * W
        '''
        dcn_out = self.dcn(x, inter)
        sft_out = self.sft(x, inter)
        out = dcn_out + sft_out
        out = x + out

        return out


class GetGradientNopadding(nn.Module):
    def __init__(self):
        super(GetGradientNopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, inp_feat):
        x_list = []
        for i in range(inp_feat.shape[1]):
            x_i = inp_feat[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        res = torch.cat(x_list, dim=1)

        return res


class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels * chan_factor), 1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        return self.bot(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels // chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        )

    def forward(self, x):
        return self.bot(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class ContextBlock(nn.Module):

    def __init__(self, n_feat, activation, bias=True):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            activation,
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        input_x = input_x.view(batch, channel, height * width)
        input_x = input_x.unsqueeze(1)
        context_mask = self.conv_mask(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.unsqueeze(3)
        context = torch.matmul(input_x, context_mask)
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        context = self.modeling(x)
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term
        return x


# Residual Context Block (RCB)
class RCB(nn.Module):
    def __init__(self, n_feat, act, bias=True):
        super(RCB, self).__init__()

        self.act = act
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias),
            self.act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        self.gcnet = ContextBlock(n_feat, self.act, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.act(self.gcnet(res))
        res = x + res
        return res


# Attention Feature Fusion (AFF)
class AFF(nn.Module):
    def __init__(self, channels, activation, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            activation,
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            activation,
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = torch.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class AtrousBlock(nn.Module):
    def __init__(self, mid_channels, kernel_size, stride, activation, atrous=[1, 2, 3, 4]):
        super(AtrousBlock, self).__init__()
        self.atrous_layers = []
        for i in range(4):
            self.atrous_layers.append(
                nn.Conv2d(mid_channels, mid_channels // 2, kernel_size, stride, dilation=atrous[i],
                          padding=atrous[i]))
        self.atrous_layers = nn.Sequential(*self.atrous_layers)
        self.conv = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0)
        self.act = activation
        self.att = AFF(mid_channels, self.act)

    def forward(self, data):
        x1 = self.act(self.atrous_layers[0](data))
        x2 = self.act(self.atrous_layers[1](data))
        x3 = self.act(self.atrous_layers[2](data))
        x4 = self.act(self.atrous_layers[3](data))

        x_total = self.act(self.conv(torch.cat((x1, x2, x3, x4), 1)))
        output = self.att(data, x_total)
        return output


class wcv_pre(nn.Module):
    def __init__(self, n_feat=32, height=256, width=256, n_RCB=2, chan_factor=2, bias=True):
        super(wcv_pre, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width
        self.act = nn.LeakyReLU(0.1, True)
        atrous = [1, 2, 3, 4]

        rcb_top = [RCB(int(n_feat * chan_factor ** 0), self.act, bias=bias) for _ in range(n_RCB)]
        self.dau_top = nn.Sequential(*rcb_top)
        rcb_mid = [RCB(int(n_feat * chan_factor ** 1), self.act, bias=bias) for _ in range(n_RCB)]
        self.dau_mid = nn.Sequential(*rcb_mid)
        rcb_bot = [RCB(int(n_feat * chan_factor ** 2), self.act, bias=bias) for _ in range(n_RCB)]
        self.dau_bot = nn.Sequential(*rcb_bot)
        self.atb_top = AtrousBlock(int(n_feat * chan_factor ** 0), 3, 1, self.act, atrous)
        self.atb_mid = AtrousBlock(int(n_feat * chan_factor ** 1), 3, 1, self.act, atrous)
        self.atb_bot = AtrousBlock(int(n_feat * chan_factor ** 2), 3, 1, self.act, atrous)
        self.nl_top = NonLocalSparseAttention(channels=int(n_feat * chan_factor ** 0))
        self.nl_mid = NonLocalSparseAttention(channels=int(n_feat * chan_factor ** 1))
        self.nl_bot = NonLocalSparseAttention(channels=int(n_feat * chan_factor ** 2))

        self.down2 = DownSample(int((chan_factor ** 0) * n_feat), 2, chan_factor)
        self.down4 = nn.Sequential(
            DownSample(int((chan_factor ** 0) * n_feat), 2, chan_factor),
            DownSample(int((chan_factor ** 1) * n_feat), 2, chan_factor)
        )

        self.up21_1 = UpSample(int((chan_factor ** 1) * n_feat), 2, chan_factor)
        self.up21_2 = UpSample(int((chan_factor ** 1) * n_feat), 2, chan_factor)
        self.up32_1 = UpSample(int((chan_factor ** 2) * n_feat), 2, chan_factor)
        self.up32_2 = UpSample(int((chan_factor ** 2) * n_feat), 2, chan_factor)

        self.conv_in = nn.Conv2d(3, n_feat, kernel_size=3, padding=1, bias=bias)
        self.conv_mid = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
        self.conv_out = nn.Conv2d(n_feat, 3, kernel_size=3, padding=1, bias=bias)
        self.grad_out = nn.Conv2d(n_feat, 3, kernel_size=1, padding=0, bias=bias)

        # only two inputs for AFF
        self.aff_top = AFF(int(n_feat * chan_factor ** 0), self.act)
        self.aff_mid = AFF(int(n_feat * chan_factor ** 1), self.act)
        self.aff_final = AFF(n_feat, self.act)
        self.igb_layer = IGM(n_feat, n_feat, 3)

        self.get_gradient = GetGradientNopadding()

        self.b_concat_1 = nn.Conv2d(2 * n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
        self.b_block_1 = RCB(2 * n_feat, self.act, bias=bias)

        self.b_concat_2 = nn.Conv2d(2 * n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
        self.b_block_2 = RCB(2 * n_feat, self.act, bias=bias)
        self.b_fea_conv = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)

    def forward(self, x, la):
        x_top = x.clone()
        x_top_la = self.conv_in(la)
        x_grad = self.get_gradient(x)
        x_top = self.conv_in(x_top)
        x_mid = self.down2(x_top)
        x_bot = self.down4(x_top)

        x_top1 = self.dau_top(self.igb_layer(self.atb_top(x_top), x_top_la))
        x_mid1 = self.dau_mid(self.atb_mid(x_mid))
        x_bot1 = self.dau_bot(self.atb_bot(x_bot))

        x_mid1 = self.aff_mid(x_mid1, self.up32_1(x_bot1))
        x_top1 = self.aff_top(x_top1, self.up21_1(x_mid1))

        x_top2 = self.dau_top(self.igb_layer(self.atb_top(x_top1), x_top_la))
        x_mid2 = self.dau_mid(self.nl_mid(x_mid1))
        x_bot2 = self.dau_bot(self.nl_bot(x_bot1))

        x_mid2 = self.aff_mid(x_mid2, self.up32_2(x_bot2))
        x_top2 = self.aff_top(x_top2, self.up21_2(x_mid2))

        mid_out = self.conv_mid(x_top2)
        mid_out = mid_out + x_top

        x_b_fea = self.conv_in(x_grad)
        x_cat_1 = torch.cat([x_b_fea, x_top1], dim=1)

        x_cat_1 = self.b_block_1(x_cat_1)
        x_cat_1 = self.b_concat_1(x_cat_1)

        x_cat_2 = torch.cat([x_cat_1, x_top2], dim=1)

        x_cat_2 = self.b_block_2(x_cat_2)
        x_cat_2 = self.b_concat_2(x_cat_2)

        grad_out = x_cat_2 + x_b_fea
        res_grad = self.grad_out(grad_out)
        out = self.aff_final(mid_out, grad_out)
        result = self.conv_out(out)

        return result, res_grad

def wcv_init(input_root = '/testdata'):
    bz = 1
    save_path = '/testsavedir'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    checkpoint = torch.load("/pre/wcv_pre.pth")
    Mydata_ = TestData(input_root)
    data_load = data.DataLoader(Mydata_, batch_size=bz)
    model = wcv_pre().cuda()
    model = nn.DataParallel(model, device_ids=[0,1])
    optimizer = AdamP(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    model.eval()
    for data_idx, data_ in enumerate(data_load):
        print(Mydata_.A_paths[data_idx].split('/')[-1])
        data_input, data_la = data_

        data_input = Variable(data_input).cuda()
        data_la = Variable(data_la).cuda()
        print(data_idx)
        with torch.no_grad():

            wcv, _ = model(data_input, data_la)
            name = Mydata_.A_paths[data_idx].split('/')[-1]
            temp_res = np.transpose(wcv[0, :].cpu().detach().numpy(), (1, 2, 0))
            temp_res[temp_res > 1] = 1
            temp_res[temp_res < 0] = 0
            temp_res = (temp_res*255).astype(np.uint8)
            temp_res = Image.fromarray(temp_res)
            temp_res.save('%s/%s' % (save_path, name))
    return wcv, save_path
