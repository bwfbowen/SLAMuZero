from typing import List 

import functools
import tensorflow as tf 
import sonnet as snt 
from acme.tf import networks


def min_max_normalize2d(s):
    s_min = tf.reduce_min(s, axis=[1, 2], keepdims=True)
    s_max = tf.reduce_max(s, axis=[1, 2], keepdims=True)
    s_scale = s_max - s_min
    s_scale = tf.where(tf.less(s_scale, 1e-5), s_scale + 1e-5, s_scale)
    s_normed = (s - s_min) / s_scale
    return s_normed


class AvgPool2D(snt.Module):
    def __init__(self, window_shape, strides, padding, name="avg_pool"):
        super().__init__(name=name)
        self.window_shape = window_shape
        self.strides = strides
        self.padding = padding

    def __call__(self, inputs):
        return tf.nn.avg_pool2d(inputs, ksize=self.window_shape, strides=self.strides, padding=self.padding)
    

class BatchReshape(snt.Module):
    def __init__(self, output_shape, name='reshape'):
        super().__init__(name=name)    
        self._output_shape = output_shape

    def __call__(self, x):
        batch_size = tf.shape(x)[0]
        target_shape = tf.concat([[batch_size], self._output_shape], 0)
        return tf.reshape(x, target_shape)


class ResidualConvBlockV1(snt.Module):
    """A v1 residual convolutional block."""
    def __init__(self, channels, stride, use_projection, name='residual_conv_block_v1'):
        super(ResidualConvBlockV1, self).__init__(name=name)
        self._use_projection = use_projection
        if use_projection:
            self._proj_conv = snt.Conv2D(output_channels=channels, kernel_shape=3, stride=stride, padding='SAME', with_bias=False)
            self._proj_ln = snt.LayerNorm(axis=[-3, -2, -1], create_scale=True, create_offset=True)
        self._conv_0 = snt.Conv2D(output_channels=channels, kernel_shape=3, stride=stride, padding='SAME', with_bias=False)
        self._ln_0 = snt.LayerNorm(axis=[-3, -2, -1], create_scale=True, create_offset=True)
        self._conv_1 = snt.Conv2D(output_channels=channels, kernel_shape=3, stride=1, padding='SAME', with_bias=False)
        self._ln_1 = snt.LayerNorm(axis=[-3, -2, -1], create_scale=True, create_offset=True)

    def __call__(self, x):
        shortcut = x

        if self._use_projection:
            shortcut = self._proj_conv(shortcut)
            shortcut = self._proj_ln(shortcut)

        out = self._conv_0(x)
        out = self._ln_0(out)
        out = tf.nn.relu(out)
        out = self._conv_1(out)
        out = self._ln_1(out)

        return tf.nn.relu(shortcut + out)


class ResNetRepresentation(snt.Module):
    def __init__(self, 
                 frame_height: int = 128, 
                 frame_width: int = 128,
                 num_channels: int = 3,
                 input_channels: int = 32, 
                 name='representation'):
        super().__init__(name=name)
        self._num_channels = num_channels
        self._frame_height = frame_height
        self._frame_width = frame_width
        self._num_elements = self._num_channels * self._frame_height * self._frame_width

        self.repr_func = snt.Sequential([
            snt.Conv2D(output_channels=input_channels, kernel_shape=3, stride=2, padding='SAME', with_bias=False),
            tf.nn.relu,
            *[ResidualConvBlockV1(channels=input_channels, stride=1, use_projection=True) for _ in range(2)],
            snt.Conv2D(output_channels=input_channels * 2, kernel_shape=3, stride=2, padding='SAME', with_bias=False),
            tf.nn.relu,
            *[ResidualConvBlockV1(channels=input_channels * 2, stride=1, use_projection=True) for _ in range(3)],
            AvgPool2D(window_shape=3, strides=2, padding='SAME'),
            *[ResidualConvBlockV1(channels=input_channels * 2, stride=1, use_projection=True) for _ in range(3)],
            AvgPool2D(window_shape=3, strides=2, padding='SAME'),
        ])

    def __call__(self, obs):
        rgb_obs = tf.reshape(obs[:, :self._num_elements], (-1, self._frame_height, self._frame_width, self._num_channels))
        s = self.repr_func(rgb_obs)
        s = min_max_normalize2d(s)
        return s


class DecEvaluation(snt.Module):
    def __init__(self, 
                 num_actions: int,
                 output_sizes: List[int] = [64, 64],
                 name='evaluation'
        ):
        super().__init__(name=name)
        self.deconv = snt.Sequential([
            snt.Flatten(),
            snt.Linear(1024),
            tf.nn.relu,
            snt.Linear(4096),
            tf.nn.relu,
            BatchReshape((8, 8, 64)),
            snt.Conv2DTranspose(output_channels=32, kernel_shape=4, stride=2, padding='SAME'),
            tf.nn.relu,
            snt.Conv2DTranspose(output_channels=16, kernel_shape=4, stride=2, padding='SAME'),
            tf.nn.relu,
            snt.Conv2DTranspose(output_channels=2, kernel_shape=4, stride=2, padding='SAME'),
        ])
        self.decpos = snt.Sequential([
            snt.Flatten(),
            snt.Linear(64),
            tf.nn.relu,
            snt.Linear(256),
            tf.nn.relu,
            snt.Linear(256),
            tf.nn.relu,
            snt.Linear(3),
        ])
        self.flatten = snt.Flatten()
        self.hidden = snt.nets.MLP(output_sizes)
        self.pv = networks.PolicyValueHead(num_actions)

    def decode(self, s):
        ds = self.decode_map(s)
        dpos = self.decode_pos(ds)
        return ds, dpos 
    
    def decode_map(self, s):
        ds = tf.sigmoid(self.deconv(s))
        return ds 
    
    def decode_pos(self, ds):
        dpos = self.decpos(ds)
        return dpos
    
    def __call__(self, s):
        ds, dpos = self.decode(s)
        x = self.flatten(ds)
        x = tf.concat([x, dpos], axis=1)
        x = self.hidden(x)
        logits, value = self.pv(x)
        return logits, value 
    

class ResNetEvaluation(snt.Module):
    def __init__(self, 
                 num_actions: int, 
                 output_sizes: List[int] = [1024, 4096],
                 name='evaluation'):
        super().__init__(name=name)
        
        self.flatten = snt.Flatten()
        self.hidden = snt.nets.MLP(output_sizes)

        output_channels = 64
        self.pi_func = snt.Sequential([
            BatchReshape((8, 8, 64)),
            snt.Conv2D(output_channels=output_channels, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
            tf.nn.relu,
            snt.Flatten(),
            snt.Linear(output_channels),
            tf.nn.relu,
            snt.Linear(num_actions)
        ])
        
        self.v_func = snt.Sequential([
            BatchReshape((8, 8, 64)),
            snt.Conv2D(output_channels=output_channels, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
            tf.nn.relu,
            snt.Conv2D(output_channels=output_channels, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
            tf.nn.relu,
            snt.Flatten(),
            snt.Linear(output_channels),
            tf.nn.relu,
            snt.Linear(1)
        ])

    def __call__(self, s):
        s = self.flatten(s)
        s = self.hidden(s)
        value = self.v_func(s)
        logits = self.pi_func(s)
        return logits, value


if __name__ == '__main__':
    import numpy as np 
    arr = np.ones((1,128*128*3+45))
    r = ResNetRepresentation()
    o = r(arr)
    print(o.shape)
    