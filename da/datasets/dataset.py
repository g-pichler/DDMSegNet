#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import keras
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from gpkeras import dataloader, preprocessing
import da.parameters
import logging
from . import settings

logger = logging.getLogger(__name__)


class NiftiDataset_2D:
    label_merging: Dict[str, Tuple[Tuple[int, ...], ...]]
    training_list: List[Dict[str, str]]
    testing_list: List[Dict[str, str]]
    frequencies: List[float]
    train_labeled: dataloader.NiftiSequence_2D
    train_unlabeled: dataloader.NiftiSequence_2D
    test_labeled: dataloader.NiftiSequence_2D
    test_unlabeled: dataloader.NiftiSequence_2D
    train_labeled_list: List[Dict[str, str]]
    train_unlabeled_list: List[Dict[str, str]]
    test_labeled_list: List[Dict[str, str]]
    test_unlabeled_list: List[Dict[str, str]]

    def __init__(self,
                 args: da.parameters.TrainingParameters,
                 model: keras.models.Model,
                 ):
        dataset: settings.Dataset = next((x for x in settings.datasets if x.Name == args.dataset_name))
        dataset.base_directory = args.dataset_directory
        self.label_merging = dataset.label_merging
        self.training_list: List[Dict[str, str]] = dataset.lists['training']
        self.testing_list: List[Dict[str, str]] = dataset.lists['testing']
        self.frequencies: List[float] = dataset.lists['frequencies']

        # n_outputs = len(model.outputs)
        # n_inputs = len(model.inputs)
        output_dim = tuple([x.value for x in model.outputs[0].get_shape()[1:3]])
        input_dim = (args.input_dim,) * 2 if isinstance(args.input_dim, int) else args.input_dim

        # check that we have a consistent number of classes
        assert len(set([len(x) for x in self.label_merging.values()])) == 1
        n_classes = len(self.label_merging['training'])

        def label_transform(labels):
            labels = preprocessing.crop_array_center(labels, dims=output_dim)
            labels = preprocessing.change_labels(labels, new_labels=self.label_merging['training'])
            labels = keras.utils.to_categorical(labels, num_classes=n_classes)
            return labels

        def data_transform(x):
            return np.expand_dims(np.squeeze(x), -1)

        def dummy_label(x):
            return np.ndarray(output_dim)

        # GM_DC = partial(gpkeras.metrics.dice_coeff, label=1)
        # WM_DC = partial(gpkeras.metrics.dice_coeff, label=2)
        # CSF_DC = partial(gpkeras.metrics.dice_coeff, label=3)
        # GM_DC.__name__ = "GM_DC"
        # WM_DC.__name__ = "WM_DC"
        # CSF_DC.__name__ = "CSF_DC"

        worker_training_indices = list(range(len(self.training_list)))
        worker_testing_indices = args.testing_indices
        for i in args.testing_indices:
            del worker_training_indices[worker_training_indices.index(i)]

        assert all([i not in worker_training_indices for i in worker_testing_indices])

        frequencies_np = np.array(self.frequencies)
        p_weights = 1 / frequencies_np
        p_weights = p_weights / n_classes

        weight_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None

        if args.weight_training_samples:
            logger.debug("Weighting training samples according to {!s}".format(p_weights))

            def weight_function(x, y):
                y = preprocessing.change_labels(y[0], new_labels=self.label_merging['training'])
                myweight = np.zeros((n_classes,))
                for v, c in zip(*np.unique(y, return_counts=True)):
                    myweight[v] = c
                myweight /= np.prod(y.shape)
                return np.sum(myweight * p_weights)
        else:
            logger.debug("Not weighting training samples")

        data_list: List[Tuple[str, ...]] = list()
        label_list: List[Tuple[str, ...]] = list()
        self.train_lableled_list = [self.training_list[i] for i in worker_training_indices]
        for entry in self.train_lableled_list:
            data_list.append((entry[args.source_class],))
            label_list.append((entry["Labels"],))

        self.train_lableled = dataloader.NiftiSequence_2D(data_list=data_list,
                                                          label_list=label_list,
                                                          basedir=dataset.base_directory,
                                                          data_transforms=(data_transform,),
                                                          label_transforms=(label_transform,),
                                                          dim=input_dim,
                                                          delta=output_dim,
                                                          normalize=args.normalize,
                                                          weight_function=weight_function,
                                                          avoid_ram=args.avoid_ram
                                                          )
        data_list = list()
        label_list = list()
        if args.dataset_shuffle:
            if args.dataset_name == 'iseg':
                # The last iSEG training sample has different dimensions, so we have to ignore it
                self.train_unlabeled_list = self.testing_list[:-1]
            else:
                self.train_unlabeled_list = self.testing_list
            train_unlabeled_list2 = self.train_unlabeled_list[1:] + self.train_unlabeled_list[:1]
        else:
            self.train_unlabeled_list = self.testing_list
            train_unlabeled_list2 = self.train_unlabeled_list

        for entry, entry2 in zip(self.testing_list, train_unlabeled_list2):
            data_list.append((entry[args.source_class], entry2[args.target_class]))
            label_list.append((entry[args.source_class],))

        self.train_unlabeled = dataloader.NiftiSequence_2D(data_list=data_list,
                                                           label_list=label_list,
                                                           basedir=dataset.base_directory,
                                                           data_transforms=(data_transform, data_transform),
                                                           label_transforms=(dummy_label,),
                                                           dim=input_dim,
                                                           delta=output_dim,
                                                           normalize=args.normalize,
                                                           avoid_ram=args.avoid_ram,
                                                           )

        data_list = list()
        label_list = list()
        self.test_labeled_list = [self.training_list[i] for i in worker_testing_indices]

        for entry in self.test_labeled_list:
            data_list.append((entry[args.target_class],))
            label_list.append((entry["Labels"],))

        self.test_labeled = dataloader.NiftiSequence_2D(data_list=data_list,
                                                        label_list=label_list,
                                                        basedir=dataset.base_directory,
                                                        data_transforms=(data_transform,),
                                                        label_transforms=(label_transform,),
                                                        dim=input_dim,
                                                        delta=output_dim,
                                                        normalize=args.normalize,
                                                        weight_function=weight_function,
                                                        avoid_ram=args.avoid_ram,
                                                        )
        data_list = list()
        label_list = list()
        self.test_unlabeled_list = self.test_labeled_list
        for entry in self.test_unlabeled_list:
            data_list.append((entry[args.source_class], entry[args.target_class]))
            label_list.append((entry[args.source_class],))

        self.test_unlabeled = dataloader.NiftiSequence_2D(data_list=data_list,
                                                          label_list=label_list,
                                                          basedir=dataset.base_directory,
                                                          data_transforms=(data_transform, data_transform),
                                                          label_transforms=(dummy_label,),
                                                          dim=input_dim,
                                                          delta=output_dim,
                                                          normalize=args.normalize,
                                                          avoid_ram=args.avoid_ram,
                                                          )
