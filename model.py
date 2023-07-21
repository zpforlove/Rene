import torch
from torch import nn
import torch.nn.functional as F
from conformer import Conformer
from cfg_parse import cfg


# Define Mish Activation function
def mish(x):
    return x * torch.tanh(F.softplus(x))


class FCNet(nn.Module):
    """
    FCNet: Classify the final result, with 2 linear layers
    """

    def __init__(self, cfg):
        super(FCNet, self).__init__()
        self.cfg = cfg
        in_dim = self.cfg['rnn_hid_dim'] * (2 if self.cfg['bidirect'] else 1)
        self.layer_0 = nn.Linear(in_dim, self.cfg['fc_layer_dim'])
        self.layer_1 = nn.Linear(self.cfg['fc_layer_dim'], self.cfg['output_dim'])
        self.drop = nn.Dropout(self.cfg['fc_layer_drop'])

    def forward(self, x):  # x: [bn, rnn_hid_dim*2]
        x = self.layer_0(x)
        x = mish(x)
        x = self.drop(x)
        return self.layer_1(x)  # [bn, output_dim]


class Model(nn.Module):
    """
    Model Building Class
    """

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

        self.fc = FCNet(self.cfg)

    def forward(self, x, input_length):  # x :  [batch, n_frames, dim]
        x = self.conformer(x, input_length)[0]  # out: [batch, n_frames, channels]
        x = x.permute(1, 0, 2)  # out: [n_frames, batch, channels]
        rnn_out, hn = self.gru(x)  # out: [n_frames, batch, chs]
        out = self.fc(rnn_out[-1])  # out: [batch, task_out_dim]
        return out


if __name__ == '__main__':
    x = torch.rand(size=(16, 1500, 384))  # [batch,n_frames,dim]
    input_lengths = torch.LongTensor(1500 * 16)  # n_frames * batch_size
    model = Model(cfg)
    x = model(x, input_lengths)
    print(x.size())
    print(x)  # [16, 15]: log prob -> softmax
