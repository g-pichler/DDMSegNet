#!/usr/bin/env python
# *-* encoding: utf-8 *-*

from da.datasets.datasetup import do_aligned
import sys
import logging


logger = logging.getLogger(__name__)

dataset = 'mrbrains13'
alignment_class = 'T1'

if __name__ != '__main__':
    logger.error('This module is not supposed to be imported.')
    sys.exit(1)
else:
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    do_aligned(dataset, alignment_class=alignment_class)
