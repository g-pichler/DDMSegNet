#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import keras
from typing import Union, Tuple
import numpy as np
from keras.utils import to_categorical

from gpkeras import util
from gpkeras import trainer

from da import parameters


class TestSequence(keras.utils.Sequence):
    def __init__(self,
                 input_dim: Union[int, Tuple[int, ...]],
                 output_dim: Tuple[int, ...],
                 n_inputs: int,
                 n_outputs: int,
                 n_batches: int = 4*32,
                 ):
        self.input_dim = (input_dim,) * 2 if isinstance(input_dim, int) else input_dim
        self.output_dim = output_dim
        self.n_batches = n_batches
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs

    def __len__(self):
        return self.n_batches

    def __getitem__(self, item):
        if not 0 <= item < self.n_batches:
            raise IndexError("Index out of bounds")

        ones_input = np.ones(self.input_dim + (1,))
        ones_output = np.ones(self.output_dim + (1,))
        zeros_input = np.zeros(self.input_dim + (1,))
        # zeros_output = np.zeros((self.batch_size,) + self.output_dim + (1,))

        data = [ones_input]
        for _ in range(self.n_inputs - 1):
            data.append(zeros_input)
        labels = [to_categorical(ones_output, 2)] * self.n_outputs

        return data, labels


def BWTestset_2D(args: parameters.TrainingParameters, model: keras.Model):
    input_dim = args.input_dim
    n_outputs = 1
    # n_inputs = len(model.inputs)
    output_dim = tuple([x.value for x in model.outputs[0].get_shape()[1:3]])

    sqs = [TestSequence(input_dim=input_dim,
                        output_dim=output_dim,
                        # n_batches=args.n_batches,
                        n_outputs=n_outputs,
                        n_inputs=n_inputs) for n_inputs in (1, 2)]
    return sqs, sqs
