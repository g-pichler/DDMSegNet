from keras.models import Model
from keras.layers import Input, core, Dropout
from keras.layers.merge import Concatenate
from keras.layers import Conv2D, MaxPool2D, UpSampling2D
from keras.layers import Lambda
import keras.losses
import gpkeras.losses
from da.parameters import TrainingParameters

import tensorflow as tf
from .helpers import Cropping2DLike


def resize(width, height):
    def resize1(inp):
        return tf.image.resize_nearest_neighbor(inp, [width, height])

    return Lambda(resize1)


#
# def ResizeLike(ref_tensor):
#     def resize_like(inp):
#         return tf.image.resize_nearest_neighbor(inp, [ref_tensor.get_shape()[1].value, ref_tensor.get_shape()[2].value])
#     return Lambda(resize_like)


def unet_raw(input_dim,
             n_classes,
             n_channels=1,
             conv_init=64,
             conv_iter=4,
             kernel_size=3,
             dropout=None,
             activation="relu",
             padding="valid",
             output_name='softmax',
             trainable=True,
             add_1conv=False):

    if dropout <= 0.0:
        dropout = None

    input_dim = (input_dim,)*2 if isinstance(input_dim, int) else tuple(input_dim)

    inp = Input(input_dim + (n_channels,))
    level = list()
    cur = Conv2D(conv_init,
                 kernel_size=kernel_size,
                 strides=1,
                 activation=activation,
                 padding=padding,
                 trainable=trainable)(inp)
    cur = Conv2D(conv_init,
                 kernel_size=kernel_size,
                 strides=1,
                 activation=activation,
                 padding=padding,
                 trainable=trainable)(cur)

    # Down
    for i in range(1, conv_iter):
        level.append(cur)
        cur = MaxPool2D(pool_size=(2, 2))(cur)
        if dropout is not None:
            cur = Dropout(dropout, trainable=trainable)(cur)
        cur = Conv2D(conv_init * (2 ** i),
                     kernel_size=kernel_size,
                     strides=1,
                     activation=activation,
                     padding=padding,
                     trainable=trainable)(cur)
        cur = Conv2D(conv_init * (2 ** i),
                     kernel_size=kernel_size,
                     strides=1,
                     activation=activation,
                     padding=padding,
                     trainable=trainable)(cur)

    # Up
    for i in range(conv_iter - 2, -1, -1):
        cur = UpSampling2D(size=(2, 2))(cur)
        if add_1conv:
            cur = Conv2D(conv_init * (2 ** i),
                         kernel_size=1,
                         trainable=trainable)(cur)
        if padding == 'valid':
            lv = Cropping2DLike(level[i], cur)(level[i])
        else:
            lv = level[i]
        cur = Concatenate(axis=3)([cur, lv])
        cur = Conv2D(conv_init * (2 ** i),
                     kernel_size=kernel_size,
                     strides=1,
                     activation=activation,
                     padding=padding,
                     trainable=trainable)(cur)
        cur = Conv2D(conv_init * (2 ** i),
                     kernel_size=kernel_size,
                     strides=1,
                     activation=activation,
                     padding=padding,
                     trainable=trainable)(cur)

    cur = Conv2D(n_classes,
                 kernel_size=(1, 1),
                 strides=1,
                 activation=activation,
                 padding=padding,
                 trainable=trainable)(cur)
    cur = core.Activation('softmax',
                          name=output_name,
                          trainable=trainable)(cur)

    model = Model(inputs=inp,
                  outputs=cur)

    return model


def Unet(args: TrainingParameters):
    input_dim = (args.input_dim,) * 2 if isinstance(args.input_dim, int) else tuple(args.input_dim)

    model = unet_raw(n_classes=args.n_classes,
                     input_dim=input_dim,
                     dropout=args.dropout,
                     add_1conv=args.unet_1x1conv,
                     padding=args.unet_padding,
                     )

    return model


def DDM(args: TrainingParameters):
    input_dim = (args.input_dim,) * 2 if isinstance(args.input_dim, int) else tuple(args.input_dim)

    unet_model = Unet(args)

    inp = [Input(input_dim + (1,)) for _ in range(3)]
    outp = [unet_model(i) for i in inp]

    for mod in (keras.losses, gpkeras.losses):
        loss_fkt = getattr(mod, args.unsupervised_loss_function, None)
        if loss_fkt is not None:
            break

    assert loss_fkt is not None

    def unsupervised_loss(tensors):
        x1 = tensors[0]
        x2 = tensors[1]
        return loss_fkt(x1, x2)

    outp0 = Lambda(lambda x: x, name='supervised_softmax')(outp[0])
    outp1 = Lambda(unsupervised_loss, name="unsupervised_loss")(outp[1:])

    return Model(inputs=inp, outputs=[outp0, outp1])
