import math
import torch
from torch import nn
from conformer import Conformer
from cfg_parse import cfg
from thop import profile


class DSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DSConv2d, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(kernel_size // 2, kernel_size // 2),
            groups=in_channels
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1)
        )

    def forward(self, input_x):
        out = self.depth_conv(input_x)
        out_final = self.pointwise_conv(out)
        return out_final


class ReneTrialBlock(nn.Module):
    def __init__(self, cfg, in_channels):
        super(ReneTrialBlock, self).__init__()
        self.cfg = cfg
        self.left_flow = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            DSConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(5, 5), padding=(5 // 2, 5 // 2))
        )
        self.right_flow = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(5, 5), padding=(5 // 2, 5 // 2)),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            DSConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1)),
        )
        self.layer = nn.Linear(cfg['rnn_hid_dim'] * 2, self.cfg['output_dim'])

    def forward(self, input_data):
        feature_size = int(math.sqrt(cfg['rnn_hid_dim'] * 2))
        input_feature = input_data.reshape(input_data.size(0), cfg['rtb_data_channels'], feature_size, feature_size)
        out = self.left_flow(input_feature) + self.right_flow(input_feature) + input_feature
        out_final = self.layer(out.view(input_data.size(0), -1))
        return out_final


class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg

        self.conformer = Conformer(num_classes=cfg['rnn_hid_dim'], input_dim=cfg['whiper_dim'],
                                   encoder_dim=cfg['encoder_dim'], num_encoder_layers=cfg['num_encoder_layers'],
                                   num_attention_heads=cfg['num_attention_heads'],
                                   input_dropout_p=cfg['input_dropout'],
                                   feed_forward_dropout_p=cfg['feed_forward_dropout'],
                                   attention_dropout_p=cfg['attention_dropout'], conv_dropout_p=cfg['conv_dropout'])

        self.gru = nn.GRU(input_size=self.cfg['rnn_hid_dim'],
                          hidden_size=self.cfg['rnn_hid_dim'],
                          num_layers=self.cfg['rnn_layers'],
                          bidirectional=self.cfg['bidirect'])
        
        self.rene = ReneTrialBlock(self.cfg, cfg['rtb_data_channels'])

    def forward(self, x, input_length):  # x :  [batch, n_frames, dim]
        x = self.conformer(x, input_length)[0]  # out: [batch, n_frames, channels]
        x = x.permute(1, 0, 2)  # out: [n_frames, batch, channels]
        rnn_out, hn = self.gru(x)  # out: [n_frames, batch, chs]
        out = self.rene(rnn_out[-1])  # out: [batch, task_out_dim]
        return out


if __name__ == '__main__':
    x = torch.rand(size=(16, 1500, 384))  # [batch,n_frames,dim]
    input_lengths = torch.LongTensor(1500 * 16)  # n_frames * batch_size
    model = Model(cfg)
    flops, params = profile(model, inputs=(x, input_lengths))
    x = model(x, input_lengths)
    print('Flops:', flops)
    print('Params:', params)
    print(x.size())
    print(x)  
