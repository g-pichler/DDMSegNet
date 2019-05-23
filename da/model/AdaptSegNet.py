#!/usr/bin/env python
# *-* encoding: utf-8 *-*

from .DDM import Unet, unet_raw
from keras.models import Model
from keras.layers import Input, core, Dropout
from keras.layers.merge import Concatenate
from keras.layers import Conv2D, MaxPool2D, UpSampling2D
from keras.layers import LeakyReLU, ReLU
from keras.layers import ZeroPadding2D, Cropping2D
from keras.layers import Lambda, Activation
import keras.losses
import keras.backend as K
from da.parameters import TrainingParameters



# TARGET_LABEL = 1
# SOURCE_LABEL = 0


def discriminator(input_dim,
                  ndf=64,
                  alpha=0.2,
                  output_name='discr',
                  trainable=True,
                  strides=(2, 2, 2, 2, 2),
                  kernel_size=4,
                  padding="same",
                  dense_layers=()):
    inp = Input(input_dim)

    x = Conv2D(ndf, kernel_size=kernel_size, strides=strides[0], padding=padding, trainable=trainable)(inp)
    x = ReLU(negative_slope=alpha)(x)
    x = Conv2D(ndf * 2, kernel_size=kernel_size, strides=strides[1], padding=padding, trainable=trainable)(x)
    x = ReLU(negative_slope=alpha)(x)
    x = Conv2D(ndf * 4, kernel_size=kernel_size, strides=strides[2], padding=padding, trainable=trainable)(x)
    x = ReLU(negative_slope=alpha)(x)
    x = Conv2D(ndf * 8, kernel_size=kernel_size, strides=strides[3], padding=padding, trainable=trainable)(x)
    x = ReLU(negative_slope=alpha)(x)
    x = Conv2D(1, name=output_name, kernel_size=kernel_size, strides=strides[4], padding=padding, trainable=trainable)(
        x)

    for k in dense_layers:
        # Adding fully connected layers at the end
        x = ReLU(negative_slope=alpha)(x)
        x = core.Dense(k, trainable=trainable)(x)

    # Sigmoid layer is included in bce loss

    return Model(inputs=inp, outputs=x)


def AdaptSegNet(args: TrainingParameters,
                trainable=True):

    input_dim = (args.input_dim,)*2 if isinstance(args.input_dim, int) else tuple(args.input_dim)

    # make the networks
    segnet = unet_raw(input_dim=input_dim,
                      n_classes=args.n_classes,
                      trainable=trainable,
                      add_1conv=args.unet_1x1conv,
                      padding=args.unet_padding,
                      dropout=args.dropout,
                      )

    discr = discriminator(input_dim=segnet.output_shape[1:],
                          trainable=trainable,
                          )

    return segnet, discr
