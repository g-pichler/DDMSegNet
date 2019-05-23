#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import gpkeras.util
import da.parameters
import numpy as np
import da.datasets.settings
from gpkeras import preprocessing


def preprocess_labels(labels: np.ndarray,
                      args: da.parameters.TrainingParameters,
                      type: str = 'training'):
    dataset = next((x for x in da.datasets.settings.datasets if x.Name == args.dataset_name))
    label_merging = dataset.label_merging
    labels = preprocessing.change_labels(labels, new_labels=label_merging[type])
    return labels


def preprocess_data(data: np.ndarray,
                    args: da.parameters.TrainingParameters):
    if args.normalize:
        data = gpkeras.util.normalize(data)
    return data
