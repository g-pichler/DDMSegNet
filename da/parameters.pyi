from gpkeras import parameters
from typing import Union, Tuple, List
import argparse

class TrainingParameters(parameters.TrainingParameters):
    input_dim: Union[int, Tuple[int, ...]]
    n_classes: int
    normalize: bool
    statistics_interval: int
    source_class: str
    target_class: str
    testing_indices: Tuple[int, ...]
    weight_training_samples: bool
    dropout: float
    dataset_shuffle: bool
    avoid_ram: bool
    unsupervised_loss_function: str
    supervised_loss_function: str
    unsupervised_lambda: float
    n_pretrain: int
    dataset_directory: str
    dataset_name: str
    adaptsegnet_adversary_optimizer: str
    adaptsegnet_adversary_lr: float
    adaptsegnet_adversary_steps: int
    adaptsegnet_segnet_steps: int
    unet_padding: str
    unet_1x1conv: bool
    adaptsegnet_adversary_strides: List[int]
    adaptsegnet_adversary_alpha: float

    def __init__(self, args: argparse.Namespace):
        pass


def get_argument_parser(description: str = None) -> argparse.ArgumentParser:
    pass
