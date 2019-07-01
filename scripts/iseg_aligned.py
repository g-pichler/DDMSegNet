#!/usr/bin/env python
# *-* encoding: utf-8 *-*


from da.datasets.datasetup import do_aligned, do_aligned_disjoint, get_dataset

import sys
import logging

logger = logging.getLogger(__name__)


dataset = 'iseg'
alignment_class = 'T1'

if __name__ != '__main__':
    logger.error('This module is not supposed to be imported.')
    sys.exit(1)
else:
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    d = get_dataset(dataset)
    testing_list = d.lists['testing'][:-1]  # The last image has different dimensions, so we exclude it
    do_aligned(dataset, testing_list, alignment_class)
    do_aligned_disjoint('{}_{}'.format(dataset, 'aligned'))
