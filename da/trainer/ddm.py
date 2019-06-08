#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import gpkeras
import keras.models
import keras.utils
import os.path as osp
import nibabel as nib
from da.model import UnetEvaluator
import da
import da.parameters
import da.datasets
from gpkeras import dataloader
import logging
import numpy as np
import gpkeras.losses
import keras.losses
from gpkeras import callbacks
from typing import Tuple, List, Dict, DefaultDict
from da.trainer import preprocessing
from collections import defaultdict
import da.datasets.dataset

logger = logging.getLogger(__name__)


class MyEvaluator:
    def __init__(self,
                 model1: keras.Model,
                 data_list:  List[Dict[str, str]],
                 args: da.parameters.TrainingParameters,
                 prefix: str = '',
                 ):
        self.n_classes = args.n_classes
        self._interval = args.statistics_interval
        self.evaluators: DefaultDict[str, List[UnetEvaluator]] = defaultdict(list)
        self.labels: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
        self._prefix = prefix if len(prefix) == 0 or prefix[-1] == '_' else '{}_'.format(prefix)

        for name, cls in zip(('SRC', 'TARG'), (args.source_class, args.target_class)):
            for test_labeled_item in data_list:
                f_data = osp.join(args.dataset_directory,
                                  test_labeled_item[cls])
                fname_label = test_labeled_item['Labels_{}'.format(cls)] if \
                    'Labels_{}'.format(cls) in test_labeled_item else test_labeled_item['Labels']
                f_label = osp.join(args.dataset_directory, fname_label)

                label = preprocessing.preprocess_labels(
                    np.asarray(nib.load(f_label).dataobj),
                    args=args)
                self.labels[name].append(label)

                data = preprocessing.preprocess_data(nib.load(f_data).get_fdata(), args=args)
                evaluator = UnetEvaluator(model1, data)
                self.evaluators[name].append(evaluator)

    def __call__(self, epoch, logs):
        if epoch != 0 and (epoch + 1) % self._interval != 0:
            logger.debug('Skipping DC computation')
            # Delete the DC logs
            for k in self.evaluators.keys():
                for key in ["{prefix}{k}_DC_{i}".format(prefix=self._prefix, k=k, i=i) for i in range(self.n_classes)]:
                    if key in logs:
                        del logs[key]
                if "{prefix}{k}_DC".format(prefix=self._prefix, k=k) in logs:
                    del logs["{prefix}{k}_DC".format(prefix=self._prefix, k=k)]
            return
        for k, eval_list in self.evaluators.items():
            dice_coeff = np.zeros((self.n_classes,))
            for e, label in zip(eval_list, self.labels[k]):
                dice_coeff += gpkeras.stuff.dice_coeff(label, e())
            dice_coeff /= len(eval_list)

            mean_dc = 0.0
            for i, dc in enumerate(dice_coeff):
                logs["{prefix}{k}_DC_{i}".format(prefix=self._prefix, k=k, i=i)] = dc
                if i > 0:
                    # zero is background
                    mean_dc += dc
            mean_dc /= len(dice_coeff) - 1
            logs["{prefix}{k}_DC".format(prefix=self._prefix, k=k)] = mean_dc
            dice_str = "[" + " ".join(["{:.3f}".format(x) for x in dice_coeff]) + "]"
            logger.info("{prefix}DICE_{k:6.6}: {dice_str}, avg: {mean_dc:.3f}".format(prefix=self._prefix,
                                                                                      k=k,
                                                                                      dice_str=dice_str,
                                                                                      mean_dc=mean_dc))


class DDMTrainer(gpkeras.trainer.KerasTrainer):
    model1: keras.models.Model
    _args: da.parameters.TrainingParameters
    model_namespace = da.model
    dataset_namespace = da.datasets.dataset
    dataset: da.datasets.dataset.NiftiDataset_2D
    testing_sequence_batched: keras.utils.Sequence
    training_sequence_batched: keras.utils.Sequence

    def __init__(self, args):
        super().__init__(args=args)
        self.register(self.get_model1, gpkeras.trainer.Stages.model)
        self.register(self.prepare_data, gpkeras.trainer.Stages.data)
        self.register(self.train, gpkeras.trainer.Stages.training)
        self.register(self.setup_callback_testing_evaluator, gpkeras.trainer.Stages.callbacks)
        self.register(self.setup_callback_evaluation_evaluator, gpkeras.trainer.Stages.callbacks)

    def setup_callback_testing_evaluator(self):
        if self._args.statistics_interval > 0 and self.dataset.test_labeled_list:
            self.keras_callbacks.insert(
                0,
                keras.callbacks.LambdaCallback(
                    on_epoch_end=MyEvaluator(model1=self.model1,
                                             data_list=self.dataset.test_labeled_list,
                                             args=self._args,
                                             )))

    def setup_callback_evaluation_evaluator(self):
        if self._args.statistics_interval > 0 and self.dataset.evaluation_list:
            self.keras_callbacks.insert(
                0,
                keras.callbacks.LambdaCallback(
                    on_epoch_end=MyEvaluator(model1=self.model1,
                                             data_list=self.dataset.evaluation_list,
                                             prefix='EVAL',
                                             args=self._args,
                                             )))

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
        else:
            unsupervised_loss_fkt = gpkeras.losses.constant_zero
            lbd = 0.0

        self.model.compile(
            optimizer=self.optimizer,
            loss={'supervised_softmax': supervised_loss_fkt,
                  'unsupervised_loss': unsupervised_loss_fkt},
            loss_weights={'supervised_softmax': 1.0,
                          'unsupervised_loss': lbd},
            metrics={"supervised_softmax": [gpkeras.metrics.pixel_accuracy]}
        )

    def get_model1(self):
        inp1 = self.model.inputs[0]
        out1 = self.model.outputs[0]
        self.model1 = keras.Model(inputs=inp1, outputs=out1)

    def train(self):
        self.register(self.save_checkpoint, gpkeras.trainer.Stages.cleanup)

        initial_epoch = self._args.initial_epoch
        while initial_epoch < self._args.n_epoch:

            train_to = self._args.n_pretrain if \
                initial_epoch < self._args.n_pretrain else \
                self._args.n_epoch

            self.model.fit_generator(self.training_sequence_batched,
                                     epochs=train_to,
                                     validation_data=self.testing_sequence_batched,
                                     callbacks=self.keras_callbacks,
                                     verbose=2,
                                     initial_epoch=initial_epoch,
                                     # max_queue_size=self._args.max_queue_size,
                                     # workers=self._args.workers,
                                     # use_multiprocessing=self._args.use_multiprocessing,
                                     )
            initial_epoch = train_to
            if initial_epoch < self._args.n_epoch:
                self.setup_compile(initial_epoch=initial_epoch)

    def prepare_data(self):
        training_sequence = dataloader.ProductSequence((self.dataset.train_lableled, self.dataset.train_unlabeled))
        testing_sequence = dataloader.ProductSequence((self.dataset.test_labeled, self.dataset.test_unlabeled))

        self.training_sequence_batched = dataloader.BatchGenerator(
            training_sequence, batchsize=self._args.batch_size)
        self.testing_sequence_batched = dataloader.BatchGenerator(
            testing_sequence, batchsize=self._args.batch_size)
