#!/usr/bin/env python
# *-* encoding: utf-8 *-*

from gpkeras import parameters
import argparse


def get_argument_parser(description='Training Pnarameters'):
    parser = parameters.get_argument_parser(description=description)
    parser.add_argument("--input_dim", nargs='+', type=int,
                        default=[128, 128], help="Input dimension")
    parser.add_argument('--n_classes', type=int, default=4,
                        help='Number of classes for segmentation')
    parser.add_argument("--no-normalize", action="store_false", default=True, dest="normalize",
                        help="Should we normalize the input?")
    parser.add_argument('--statistics_interval', type=int, default=1,
                        help='Compute statistics every # epochs')
    parser.add_argument('--source_class', type=str, default='T1',
                        help='Source class for training')
    parser.add_argument('--target_class', type=str, default='T2',
                        help='Target class for training')
    parser.add_argument('--dropout', type=float, default=-1.0,
                        help='Dropout for the network')
    parser.add_argument("--testing_indices", nargs='+', type=int,
                        default=(0,), help="Indices for testing")
    parser.add_argument("--no-weight-samples", action="store_false", default=True,
                        dest='weight_training_samples', help="Weight the training samples by occurrence of classes")
    parser.add_argument("--dataset_shuffle", action="store_true", default=False,
                        help="Shuffle the dataset for the target class")
    parser.add_argument("--avoid_ram", action="store_true", default=False,
                        help="Avoid storing pre-processed images in memory")
    parser.add_argument("--unsupervised_loss_function", type=str, default='kullback_leibler_divergence',
                        help="loss for the unsupervised term")
    parser.add_argument("--supervised_loss_function", type=str, default='categorical_crossentropy',
                        help="loss for the unsupervised term")
    parser.add_argument('--unsupervised_lambda', type=float, default=0.1,
                        help='Lambda for unsupervised loss')
    parser.add_argument('--n_pretrain', type=int, default=200,
                        help='No. of epochs to pretrain with lambda=0.0')
    parser.add_argument("--dataset_directory", type=str,
                        default="./data/{args.dataset}", help="The dataset base directory")
    parser.add_argument("--dataset_name", type=str, default="mrbrains13",
                        help="Name of the dataset")
    parser.add_argument("--adaptsegnet_adversary_optimizer", type=str,
                        default="adam", help="Name of the dataset")
    parser.add_argument('--adaptsegnet_adversary_lr', type=float, default=1e-4,
                        help='Learning rate for the adversary')
    parser.add_argument('--adaptsegnet_segnet_steps', type=int, default=1,
                        help='Number of batches for the segmentation network '
                             'sees before the adversary is trained again')
    parser.add_argument('--adaptsegnet_adversary_steps', type=int, default=1,
                        help='Number of batches for the adversary '
                             'sees before the segnet is trained again')
    parser.add_argument('--unet_padding', type=str, default='valid',
                        choices=('valid', 'same'), help='Padding for the U-Net')
    parser.add_argument('--unet_1x1conv', action='store_true', default=False,
                        help='Do not add a 1x1 conv layer to the U-Net')
    parser.add_argument('--adaptsegnet_adversary_strides', type=int, default=(2, 2, 1, 1, 2),
                        nargs='+', help='Strides for the adversary network')
    parser.add_argument('--adaptsegnet_adversary_alpha', type=float, default=0.0,
                        help='Negative slope for LeakyRelu in adversary network')
    return parser


class TrainingParameters(parameters.TrainingParameters):
    def __init__(self, args: argparse.Namespace):
        # Adding new parameters
        self.params += ['input_dim',
                        'n_classes',
                        'normalize',
                        'statistics_interval',
                        'source_class',
                        'target_class',
                        'dropout',
                        'testing_indices',
                        'weight_training_samples',
                        'dataset_shuffle',
                        'avoid_ram',
                        'unsupervised_loss_function',
                        'supervised_loss_function',
                        'unsupervised_lambda',
                        'n_pretrain',
                        'dataset_directory',
                        'dataset_name',
                        'adaptsegnet_adversary_optimizer',
                        'adaptsegnet_adversary_lr',
                        'adaptsegnet_adversary_steps',
                        'adaptsegnet_segnet_steps',
                        'unet_padding',
                        'unet_1x1conv',
                        'adaptsegnet_adversary_strides',
                        'adaptsegnet_adversary_alpha',
                        ]

        self.formatted_params += [
            'dataset_directory',
        ]

        super().__init__(args)

    def _check_dataset(self):
        assert self.dataset in ('iseg', 'mrbrains13')
