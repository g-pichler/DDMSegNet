#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import os.path as osp
from typing import Dict, Tuple, Union, List, Optional
from dataclasses import dataclass
import os
import json
from itertools import chain


@dataclass
class Dataset:
    Name: str
    Classes: Tuple[str, ...]
    label_merging: Dict[str, Tuple[Tuple[int, ...], ...]]
    base_directory: Optional[str] = None
    lists_file: Optional[str] = None
    lists: Optional[Dict[str, Union[List[Dict[str, str]], List[float]]]] = None

    def __post_init__(self):
        if self.lists_file is None:
            lists_dir = osp.join(osp.dirname(osp.realpath(__file__)), "lists")
            os.makedirs(lists_dir, exist_ok=True)
            self.lists_file = osp.join(lists_dir, "{}.json".format(self.Name))

        if self.lists is None:
            try:
                with open(self.lists_file, 'r') as fp:
                    self.lists = json.load(fp)
            except FileNotFoundError:
                pass


datasets: Tuple[Dataset, ...] = (
    (Dataset('mrbrains13',
             ('T1', 'T2_FLAIR', 'T1_IR', 'T1_1mm'),
             {
                 'training': ((0, 7, 8), (1, 2), (3, 4), (5, 6)),
                 'testing': ((0, 7, 8), (1, 2), (3, 4), (5, 6)),
             },
             ),
     Dataset('mrbrains13_aligned',
             ('A_T1', 'B_T1', 'A_T2_FLAIR', 'B_T2_FLAIR', 'A_T1_IR', 'B_T1_IR', 'A_T1_1mm', 'B_T1_1mm'),
             {
                 'training': ((0, 7, 8), (1, 2), (3, 4), (5, 6)),
                 'testing': ((0, 7, 8), (1, 2), (3, 4), (5, 6)),
             },
             ),
    Dataset('mrbrains13_aligned_disjoint',
             ('A_T1', 'B_T1', 'A_T2_FLAIR', 'B_T2_FLAIR', 'A_T1_IR', 'B_T1_IR', 'A_T1_1mm', 'B_T1_1mm'),
             {
                 'training': ((0, 7, 8), (1, 2), (3, 4), (5, 6)),
                 'testing': ((0, 7, 8), (1, 2), (3, 4), (5, 6)),
             },
             ),
     Dataset('iseg',
             ('T1', 'T2'),
             {
                 'training': ((0,), (150,), (250,), (10,)),
                 'testing': ((0,), (150,), (250,), (10,)),
             },
             ),
     Dataset('iseg_aligned',
             ('A_T1', 'B_T1', 'A_T2', 'B_T2'),
             {
                 'training': ((0,), (150,), (250,), (10,)),
                 'testing': ((0,), (150,), (250,), (10,)),
             },
             ),
     Dataset('iseg_aligned_disjoint',
             ('A_T1', 'B_T1', 'A_T2', 'B_T2'),
             {
                 'training': ((0,), (150,), (250,), (10,)),
                 'testing': ((0,), (150,), (250,), (10,)),
             },
             ),
     )
)
