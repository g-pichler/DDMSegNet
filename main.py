#!/usr/bin/env python
# *-* encoding: utf-8 *-*

from da.parameters import TrainingParameters, get_argument_parser

import da.trainer as trainer
import gpkeras
from typing import List

import logging
logger = logging.getLogger(__name__)


def get_args(env_args: List[str] = None) -> gpkeras.parameters.TrainingParameters:

    parser = get_argument_parser()
    args = parser.parse_args(env_args)
    param = TrainingParameters(args)

    logger.info(str(param))

    return param


def main(args: gpkeras.parameters.TrainingParameters):

    t = getattr(trainer, args.trainer)(args)
    return t()


if __name__ == '__main__':
    main(get_args())
