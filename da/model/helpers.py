#!/usr/bin/env python
# *-* encoding: utf-8 *-*

from keras.layers import Cropping2D
import keras.models
import numpy as np
from typing import List, Tuple


def Cropping2DLike(fro, to):
    fro_height, fro_width = fro.get_shape()[1].value, fro.get_shape()[2].value
    to_height, to_width = to.get_shape()[1].value, to.get_shape()[2].value
    top = (fro_height - to_height) // 2
    bottom = fro_height - to_height - top
    left = (fro_width - to_width) // 2
    right = fro_width - to_width - left
    return Cropping2D(cropping=((top, bottom), (left, right)))


class UnetEvaluator:
    def __init__(self, mod1: keras.models.Model, inp: np.ndarray = None):
        self._mod1 = mod1
        self._inp: np.ndarray
        self._data_shape: Tuple[int, ...]
        self._inp_list: List[List[np.ndarray]]
        self._n: np.ndarray

        self._output_dim = np.array(mod1.output_shape[1:3])
        self._input_dim = np.array(mod1.input_shape[1:3])
        self._n_classes = mod1.output_shape[3]
        self._overlap = np.floor((self._input_dim - self._output_dim) / 2).astype(int)

        if inp is not None:
            self.set_data(inp)

    def set_data(self, inp: np.ndarray):

        if len(inp.shape) == 4:
            inp = np.squeeze(inp, axis=-1)

        self._inp = inp
        self._data_shape = inp.shape

        inp_flip = np.flip(inp, axis=0)
        inp_augment = np.concatenate((inp_flip[self._data_shape[0]-self._overlap[0]:],
                                      inp,
                                      inp_flip[:self._input_dim[0]]), axis=0)
        inp_flip = np.flip(inp_augment, axis=1)
        inp_augment = np.concatenate((inp_flip[:, self._data_shape[1]-self._overlap[1]:],
                                      inp_augment,
                                      inp_flip[:, :self._input_dim[1]]), axis=1)
        self._n = np.ceil(self._data_shape[0:2] / self._output_dim).astype(int)

        inp2 = np.ndarray((np.prod(self._n),) + tuple(self._input_dim) + (1,))
        self._inp_list = list()
        for i in range(self._data_shape[2]):
            inp1 = inp_augment[:, :, i]
            j = 0
            for x in range(self._n[0]):
                for y in range(self._n[1]):
                    inp2[j] = np.expand_dims(inp1[x * self._output_dim[0]:x * self._output_dim[0] + self._input_dim[0],
                                             y * self._output_dim[1]:y * self._output_dim[1] + self._input_dim[1]],
                                             axis=-1)
                    j += 1
            self._inp_list.append(inp2.copy())

    def __call__(self):
        out = np.ndarray(self._data_shape + (self._n_classes,))
        out1 = np.ndarray(tuple((self._output_dim * self._n).astype(int)) + (self._n_classes,))
        for i in range(self._data_shape[2]):
            outp = self._mod1.predict(self._inp_list[i], steps=1)
            j = 0
            for x in range(self._n[0]):
                for y in range(self._n[1]):
                    out1[x * self._output_dim[0]:(x + 1) * self._output_dim[0],
                         y * self._output_dim[1]:(y + 1) * self._output_dim[1]] = outp[j]
                    j += 1
            out[:, :, i] = out1[:self._data_shape[0], :self._data_shape[1]]
        return out


def apply_unet(mod1: keras.models.Model, inp: np.ndarray):
    data_shape = inp.shape
    output_dim = np.array([x.value for x in mod1.output.get_shape()[1:3]])
    input_dim = np.array([x.value for x in mod1.input.get_shape()[1:3]])
    n_classes = mod1.output.get_shape()[3].value
    overlap = np.floor((input_dim - output_dim) / 2).astype(int)
    out = np.ndarray(data_shape + (n_classes,))

    # def make_list(x):
    #     return [int(y) for y in x]
    # output_dim = make_list(output_dim)
    # input_dim = make_list(input_dim)
    # overlap = make_list(overlap)

    inp_flip = np.flip(inp, axis=0)
    inp_augment = np.concatenate((inp_flip[data_shape[0]-overlap[0]:], inp, inp_flip[:input_dim[0]]), axis=0)
    inp_flip = np.flip(inp_augment, axis=1)
    inp_augment = np.concatenate((inp_flip[:, data_shape[1]-overlap[1]:], inp_augment, inp_flip[:, :input_dim[1]]), axis=1)
    n = np.ceil(data_shape[0:2] / output_dim).astype(int)

    out1 = np.ndarray(tuple((output_dim * n).astype(int)) + (n_classes,))
    inp2 = np.ndarray((np.prod(n),) + tuple(input_dim) + (1,))

    for i in range(data_shape[2]):
        inp1 = inp_augment[:, :, i]
        j = 0
        for x in range(n[0]):
            for y in range(n[1]):
                inp2[j] = np.expand_dims(inp1[x * output_dim[0]:x * output_dim[0] + input_dim[0],
                                         y * output_dim[1]:y * output_dim[1] + input_dim[1]], axis=-1)
                j += 1
        outp = mod1.predict(inp2, steps=1)
        j = 0
        for x in range(n[0]):
            for y in range(n[1]):
                out1[x * output_dim[0]:(x + 1) * output_dim[0],
                y * output_dim[1]:(y + 1) * output_dim[1]] = outp[j]
                j += 1
        out[:, :, i] = out1[:data_shape[0], :data_shape[1]]

    return out
