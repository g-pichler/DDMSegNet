#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import os.path as osp
import json
import logging
import os
from gpkeras.stuff import get_label_frequencies_nii
import sys
from da.datasets.datasetup import get_dataset

logger = logging.getLogger(__name__)

dataset = 'iseg'

if __name__ != '__main__':
    logger.error('This module is not supposed to be imported.')
    sys.exit(1)
else:

    dataset = get_dataset(dataset=dataset)
    os.chdir(dataset.base_directory)
    lists = dict()

    protos = dataset.Classes + ('Labels', )
    fname = {proto: 'label' if proto == 'Labels' else proto for proto in protos}
    training_list = list()
    for subject_id in range(1, 11):
        training_data = {proto: 'subject-{subject_id!s}-{proto}.img'.format(subject_id=subject_id, proto=fname[proto])
                         for proto in protos}
        for f in training_data.values():
            assert osp.exists(f)
        training_list.append(training_data)

    lists['training'] = training_list

    protos = dataset.Classes
    testing_list = list()
    for subject_id in range(11, 24):
        testing_data = {proto: 'subject-{subject_id!s}-{proto}.img'.format(subject_id=subject_id, proto=proto)
                        for proto in protos}
        for f in testing_data.values():

            assert osp.exists(f)
        testing_list.append(testing_data)

    lists['testing'] = testing_list

    label_list = list()
    for i, entry in enumerate(training_list):
        label_list.append(entry["Labels"])
    freqs = list(get_label_frequencies_nii(label_list,
                                           label_merging=dataset.label_merging['training'],)
                 )
    lists['frequencies'] = freqs

    with open(dataset.lists_file, 'w') as fp:
        json.dump(lists, fp, indent=1)
