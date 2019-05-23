#!/usr/bin/env python
# *-* encoding: utf-8 *-*

from .ddm import DDMTrainer
import keras
from typing import List
from gpkeras import dataloader
import gpkeras
from keras.layers import Lambda, Input, Average
from keras.models import Model
import keras.backend as K
import keras.optimizers
from pprint import pformat
import json
import logging
import os.path as osp

logger = logging.getLogger(__name__)


class AdaptSegNetTrainer(DDMTrainer):
    training_sequences_batched: List[keras.utils.Sequence]
    testing_sequences_batched: List[keras.utils.Sequence]
    model_worker: keras.Model = None
    model_adversary: keras.Model = None
    opt_worker: keras.optimizers.Optimizer = None
    opt_adversary: keras.optimizers.Optimizer = None
    worker: keras.Model = None
    adversary: keras.Model = None
    worker_frozen: keras.Model = None
    adversary_frozen: keras.Model = None

    def __init__(self, args):
        super().__init__(args=args)

        self.register(self.prepare_model, gpkeras.trainer.Stages.model)
        self.unregister(self.load_weights)

    def load_weights(self):
        we_are_resuming = False
        if self._args.resume:
            assert self._args.state_file
            try:
                with open(self._args.state_file, "r") as fp:
                    self._args.initial_epoch, self._args.load_weights = json.load(fp)
                    self._args.load_weights = osp.join(osp.dirname(self._args.state_file),
                                                       osp.basename(self._args.load_weights))

            except FileNotFoundError:
                logger.warning("State file {} not found, not loading old state".format(self._args.state_file))
            else:
                we_are_resuming = True
                logger.info(
                    "Resuming epoch {} from {}".format(self._args.initial_epoch, self._args.load_weights))
        if self._args.load_weights:
            logger.info("Loading weights from {}".format(self._args.load_weights))
            if not we_are_resuming:
                logger.debug('We are not resuming, so only restoring worker weights')
                self.model1.load_weights(self._args.load_weights, by_name=True)
            else:
                self.model.load_weights(self._args.load_weights)

    def setup_optimizer(self):

        o = getattr(keras.optimizers, self._args.optimizer)
        self.opt_worker = o(lr=self._args.lr)

        o = getattr(keras.optimizers, self._args.adaptsegnet_adversary_optimizer)
        self.opt_adversary = o(lr=self._args.adaptsegnet_adversary_lr)

    def setup_compile(self, initial_epoch=None):
        for mod in (keras.losses, gpkeras.losses):
            supervised_loss_fkt = getattr(mod, self._args.supervised_loss_function, None)
            if supervised_loss_fkt is not None:
                break
        assert supervised_loss_fkt is not None

        if initial_epoch is None:
            initial_epoch = self._args.initial_epoch

        if initial_epoch >= self._args.n_pretrain:
            unsupervised_loss_fkt = gpkeras.losses.prediction_loss
            lbd = self._args.unsupervised_lambda

            self.model_adversary.compile(self.opt_adversary,
                                         loss={"discr_bce": gpkeras.losses.prediction_loss},
                                         )

        else:
            unsupervised_loss_fkt = gpkeras.losses.constant_zero
            lbd = 0.0

        self.model_worker.compile(self.opt_worker,
                                  loss={'supervised_softmax': supervised_loss_fkt,
                                        'segnet_bce': unsupervised_loss_fkt,
                                        },
                                  loss_weights={'supervised_softmax': 1.0,
                                                'segnet_bce': lbd
                                                },
                                  metrics={"supervised_softmax": [gpkeras.metrics.pixel_accuracy]},
                                  )

        logger.info("Worker Model Summary:")
        self.model_worker.summary(print_fn=logger.info)

        logger.info("Adversary Model Summary:")
        self.model_adversary.summary(print_fn=logger.info)

    def train(self):

        self.register(self.save_checkpoint, gpkeras.trainer.Stages.cleanup)

        worker_steps = self._args.adaptsegnet_segnet_steps
        adversary_steps = self._args.adaptsegnet_adversary_steps

        initial_epoch = self._args.initial_epoch
        if initial_epoch < self._args.n_pretrain:
            self.model_worker.fit_generator(self.training_sequences_batched[0],
                                            epochs=self._args.n_pretrain,
                                            validation_data=self.testing_sequences_batched[0],
                                            callbacks=self.keras_callbacks,
                                            verbose=2,
                                            initial_epoch=initial_epoch)

            self.worker_frozen.set_weights(self.worker.get_weights())
            initial_epoch = self._args.n_pretrain
            self.setup_compile(initial_epoch=initial_epoch)

        n_len = (len(self.training_sequences_batched[0]), len(self.training_sequences_batched[1]))
        n_batches = min([n_len[0] // worker_steps, n_len[1] // adversary_steps])

        for callback in self.keras_callbacks:
            callback.set_model(self.model)
        for callback in self.keras_callbacks:
            callback.on_train_begin(None)
        for i_epoch in range(initial_epoch, self._args.n_epoch):
            logs = dict()
            logger.info("Epoch {}/{}".format(i_epoch + 1, self._args.n_epoch))
            for callback in self.keras_callbacks:
                callback.on_epoch_begin(epoch=i_epoch, logs=logs)
            i_worker, i_adversary = 0, 0
            for i_batch in range(n_batches):
                #sys.stdout.write('\rBatch {}/{}'.format(i_batch + 1, n_batches))
                for callback in self.keras_callbacks:
                    callback.on_batch_begin(batch=i_batch, logs=logs)
                # logger.debug("Training Worker")
                for _ in range(worker_steps):
                    x_train, y_train, sample_weight, *_ = self.training_sequences_batched[0][i_worker] + (None,)
                    self.model_worker.train_on_batch(x=x_train,
                                                     y=y_train,
                                                     sample_weight=sample_weight,
                                                     )
                    # self.model_worker.fit_generator(self.training_sequences_batched[0], verbose=0)
                    i_worker += 1

                self.worker_frozen.set_weights(self.worker.get_weights())

                # logger.debug("Training Adversary")
                for _ in range(adversary_steps):
                    x_train, y_train, sample_weight, *_ = self.training_sequences_batched[1][i_adversary] + (None,)
                    self.model_adversary.train_on_batch(x=x_train,
                                                        y=y_train,
                                                        sample_weight=sample_weight,
                                                        )
                    # self.model_adversary.fit_generator(self.training_sequences_batched[1], verbose=0)
                    i_adversary += 1

                self.adversary_frozen.set_weights(self.adversary.get_weights())
                for callback in self.keras_callbacks:
                    callback.on_batch_end(batch=i_batch, logs=logs)

            #sys.stdout.write("\r")

            if i_epoch == 0 or (i_epoch + 1) % self._args.statistics_interval == 0:
                # Losses
                losses = self.model_worker.evaluate_generator(self.training_sequences_batched[0],
                                                              verbose=0)
                losses = (losses,) if not isinstance(losses, list) else losses
                for k, v in zip(self.model_worker.metrics_names, losses):
                    logs[k] = v

                # Validation
                losses = self.model_worker.evaluate_generator(self.testing_sequences_batched[0],
                                                              verbose=0)
                losses = (losses,) if not isinstance(losses, list) else losses
                for k, v in zip(self.model_worker.metrics_names, losses):
                    logs["val_" + k] = v

                # Adversary
                losses = self.model_adversary.evaluate_generator(self.training_sequences_batched[1],
                                                                 verbose=0)
                losses = (losses,) if not isinstance(losses, list) else losses
                for k, v in zip(self.model_adversary.metrics_names, losses):
                    logs["adversary_" + k] = v

                # Adversary validation
                losses = self.model_adversary.evaluate_generator(self.testing_sequences_batched[1],
                                                                 verbose=0)
                losses = (losses,) if not isinstance(losses, list) else losses
                for k, v in zip(self.model_adversary.metrics_names, losses):
                    logs["adversary_val_" + k] = v

                logger.info("Performance:\n{}".format(pformat(logs, width=1)))
            else:
                logger.debug('Skipping performance evaluation')

            for callback in self.keras_callbacks:
                callback.on_epoch_end(epoch=i_epoch,
                                      logs=logs)

            for sequence in self.training_sequences_batched + self.testing_sequences_batched:
                sequence.on_epoch_end()

    def prepare_data(self):
        super(AdaptSegNetTrainer, self).prepare_data()

        self.training_sequences_batched = \
            [self.training_sequence_batched,
             dataloader.BatchGenerator(self.dataset.train_unlabeled, batchsize=self._args.batch_size)
             ]

        self.testing_sequences_batched = \
            [self.testing_sequence_batched,
             dataloader.BatchGenerator(self.dataset.test_unlabeled, batchsize=self._args.batch_size)
             ]

    def get_model1(self):
        m = getattr(self.model_namespace, self._args.model)

        self.worker, self.adversary = self.model
        self.worker_frozen, self.adversary_frozen = m(self._args, trainable=False)

        self.model1 = self.worker

    def prepare_model(self):

        inp = [Input(self.worker.input_shape[1:]), Input(self.adversary.input_shape[1:])]
        outp = [self.worker(inp[0]), self.adversary(inp[1])]
        self.model = Model(inputs=inp, outputs=outp)

        # Load weights and set them equal
        self.load_weights()
        self.worker_frozen.set_weights(self.worker.get_weights())
        self.adversary_frozen.set_weights(self.adversary.get_weights())

        inp = [Input(self.worker.input_shape[1:]) for _ in range(3)]
        outp = list()
        outp0 = self.worker(inp[0])
        outp.append(Lambda(lambda x: x, name="supervised_softmax")(outp0))
        outp0 = self.adversary_frozen(self.worker(inp[2]))
        outp.append(Lambda(lambda x: keras.losses.binary_crossentropy(0 * K.ones_like(x), x),
                           name="segnet_bce")(outp0))
        self.model_worker = Model(inputs=inp, outputs=outp)

        inp = [Input(self.worker.input_shape[1:]) for _ in range(2)]
        outp = list()
        outp0 = self.adversary(self.worker_frozen(inp[0]))
        outp0 = Lambda(lambda x: keras.losses.binary_crossentropy(K.zeros_like(x), x),
                       name="bce_source")(outp0)
        outp1 = self.adversary(self.worker_frozen(inp[1]))
        outp1 = Lambda(lambda x: keras.losses.binary_crossentropy(K.ones_like(x), x),
                       name="bce_target")(outp1)
        outp.append(Average(name="discr_bce")([outp0, outp1]))

        self.model_adversary = Model(inputs=inp, outputs=outp)
