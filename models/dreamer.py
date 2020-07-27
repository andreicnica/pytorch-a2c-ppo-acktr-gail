import numpy as np
import torch
import torch.nn as nn

from models.model import NNBase, init, init_null


class HeadDecoder(nn.Module):
    def __init__(self, action_size, feature_size, action_hidden_size=200, action_layers=3, dist='one_hot',
                 activation=nn.ELU, min_std=1e-4, init_std=5, mean_scale=5):
        super().__init__()
        self.action_size = action_size
        self.feature_size = feature_size
        self.hidden_size = action_hidden_size
        self.layers = action_layers
        self.dist = dist
        self.activation = activation
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.feedforward_model = self.build_model()
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)

    def build_model(self):
        model = [nn.Linear(self.feature_size, self.hidden_size)]
        model += [self.activation()]
        for i in range(1, self.layers):
            model += [nn.Linear(self.hidden_size, self.hidden_size)]
            model += [self.activation()]
        if self.dist == 'tanh_normal':
            model += [nn.Linear(self.hidden_size, self.action_size * 2)]
        elif self.dist == 'one_hot' or self.dist == 'relaxed_one_hot':
            model += [nn.Linear(self.hidden_size, self.action_size)]
        elif self.dist == 'no_head':
            pass
        else:
            raise NotImplementedError(f'{self.dist} not implemented')
        return nn.Sequential(*model)

    def forward(self, state_features):
        x = self.feedforward_model(state_features)
        return x


class ObservationEncoder(nn.Module):
    def __init__(self, depth=32, stride=2, shape=(4, 84, 84), activation=nn.ReLU):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(shape[0], 1 * depth, 4, stride),
            activation(),
            nn.Conv2d(1 * depth, 2 * depth, 4, stride),
            activation(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride),
            activation(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride),
            activation(),
        )
        self.shape = shape
        self.stride = stride
        self.depth = depth

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        embed = torch.reshape(embed, (*batch_shape, -1))
        return embed

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.shape[1:], 0, 4, self.stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, 4, self.stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, 4, self.stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, 4, self.stride)
        embed_size = 8 * self.depth * np.prod(conv4_shape).item()
        return embed_size


class DreamerNet(NNBase):
    def __init__(self, cfg, obs_space, action_space):
        num_inputs = obs_space[0]
        recurrent = cfg.recurrent
        hidden_size = getattr(cfg, "hidden_size", 200)
        use_init = cfg.use_init

        super(DreamerNet, self).__init__(recurrent, hidden_size, hidden_size)
        self._hidden_size = hidden_size

        self.main = ObservationEncoder()

        out_conv_size = self.main(torch.rand((1,) + obs_space)).size()
        out_feat_size = int(np.prod(out_conv_size))

        # Dreamer RSSMTransition - without GRU CELL
        self.fc = nn.Sequential(
            nn.Linear(out_feat_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        if use_init:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0))
        else:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0))

        self.critic_linear = HeadDecoder(1, hidden_size)
        self.policy = HeadDecoder(1, hidden_size, dist="no_head")

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)
        x = self.fc(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), self.policy(x), rnn_hxs


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)


def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))
