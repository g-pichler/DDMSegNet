#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import argparse
import os.path as osp
import os
from . import settings
import logging
import numpy as np
import json
from da.datasets import settings
import sys
from itertools import chain
from pathlib import Path
import nibabel as nib
import logging
import os
import gpkeras

import SimpleITK as sitk

logger = logging.getLogger(__name__)


def get_dataset(dataset, arguments=None) -> settings.Dataset:
    parser = argparse.ArgumentParser(description='Dataset Parameters')
    parser.add_argument("dataset_directory", type=str,
                        help="The dataset base directory")
    args = parser.parse_args(arguments)
    dataset = next((x for x in settings.datasets if x.Name == dataset), None)
    dataset.base_directory = args.dataset_directory
    return dataset


def do_aligned(dataset, testing_list=None,
               alignment_class='T1',
               registration_algorithm=None,
               suffix='aligned'):
    if registration_algorithm is None:
        def registration_algorithm(src_img):
            return sitk.AffineTransform(3)

    new_dataset = '{}_{}'.format(dataset, suffix)
    dataset = get_dataset(dataset=dataset)
    os.chdir(dataset.base_directory)

    if testing_list is None:
        testing_list = dataset.lists['testing']
    testing_list2 = testing_list[1:] + testing_list[:1]

    new_classes = tuple(chain(*[['{}_{}'.format(x, cls) for x in ('A', 'B')] for cls in dataset.Classes]))
    new_base_directory = Path(suffix)
    os.makedirs(str(new_base_directory), exist_ok=True)

    new_dataset = settings.Dataset(new_dataset,
                                   Classes=new_classes,
                                   label_merging=dataset.label_merging,
                                   base_directory=str(new_base_directory),
                                   lists=dataset.lists.copy())

    new_dataset.lists['testing'] = list()
    new_dataset.lists['training'] = list()
    for x in dataset.lists['training']:
        entry = dict()
        for cls, v in x.items():
            if cls != 'Labels':
                for z in ('A', 'B'):
                    entry['{}_{}'.format(z, cls)] = v
            else:
                entry[cls] = v
        new_dataset.lists['training'].append(entry)

    n_imgs = len(testing_list)

    # transform_v1 = None
    # src_sitk_img = None
    # first_transform = None
    # first_src_sitk_img = None
    for i in range(n_imgs):
        logger.info("Doing {}/{} ...".format(i + 1, n_imgs))
        src_path = testing_list[i][alignment_class]
        targ_path = testing_list2[i][alignment_class]

        src_img: nib.Nifti1Image = nib.load(src_path)
        targ_img: nib.Nifti1Image = nib.load(targ_path)
        # np_arrs = [np.asarray(v.dataobj) for v in imgs]

        src_fdata = src_img.get_fdata()
        targ_fdata = targ_img.get_fdata()

        src_fdata = np.squeeze(src_fdata)
        targ_fdata = np.squeeze(targ_fdata)

        src_fdata = gpkeras.util.normalize(src_fdata)
        targ_fdata = gpkeras.util.normalize(targ_fdata)

        src_sitk_img: sitk.Image = sitk.GetImageFromArray(src_fdata)
        targ_sitk_img: sitk.Image = sitk.GetImageFromArray(targ_fdata)

        src_sitk_img = sitk.Cast(src_sitk_img, sitk.sitkFloat32)
        targ_sitk_img = sitk.Cast(targ_sitk_img, sitk.sitkFloat32)

        # Transforming !
        logger.debug('Doing affine registration...')
        transform_v0 = sitk.CenteredTransformInitializer(src_sitk_img, targ_sitk_img,
                                                         sitk.Euler3DTransform(),
                                                         sitk.CenteredTransformInitializerFilter.GEOMETRY)
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        # registration_method.SetMetricAsMeanSquares()

        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0,
                                                          numberOfIterations=1000,
                                                          convergenceWindowSize=35,
                                                          )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Set the initial moving and optimized transforms.
        # optimized_transform = sitk.Similarity3DTransform()
        opt_transform = registration_algorithm(src_sitk_img)
        registration_method.SetMovingInitialTransform(transform_v0)
        registration_method.SetInitialTransform(opt_transform)

        registration_method.Execute(src_sitk_img, targ_sitk_img)

        logger.info('Final metric value: {0}'.format(registration_method.GetMetricValue()))
        logger.info(
            'Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

        # Need to compose the transformations after registration.
        # final_transform_v11 = sitk.Transform(optimized_transform)
        # final_transform_v11.AddTransform(initial_transform)

        transform_v1 = sitk.Transform(transform_v0)
        transform_v1.AddTransform(opt_transform)

        logger.debug('Transforming ...')

        list_entry = dict()
        for cls in dataset.Classes:
            path = testing_list2[i][cls]
            img: nib.Nifti1Image = nib.load(path)
            fdata = np.asarray(img.dataobj)
            if len(fdata.shape) == 4:
                assert fdata.shape[3] == 1
                expand_dim = True
            else:
                expand_dim = False
            fdata = np.squeeze(fdata)
            sitk_img: sitk.Image = sitk.GetImageFromArray(fdata)
            img_resampled: sitk.Image = sitk.Resample(sitk_img, src_sitk_img, transform_v1,
                                                      sitk.sitkLinear, 0, sitk_img.GetPixelIDValue())
            np_out = sitk.GetArrayFromImage(img_resampled)
            if expand_dim:
                np_out = np.expand_dims(np_out, axis=-1)
            nifti_out = nib.Nifti1Image(np_out, img.affine, img.header)

            list_entry['{}_{}'.format('A', cls)] = testing_list[i][cls]
            out_path = new_base_directory / '{!s}_{}.nii.gz'.format(i + 1, cls)
            list_entry['{}_{}'.format('B', cls)] = str(out_path)
            nib.save(nifti_out, str(out_path))
        new_dataset.lists['testing'].append(list_entry)

    with open(new_dataset.lists_file, 'w') as fp:
        json.dump(new_dataset.lists, fp, indent=1)


def do_aligned_disjoint(dataset, suffix='disjoint'):
    base_dataset = next((x for x in settings.datasets if x.Name == dataset), None)
    new_dataset = '{}_{}'.format(dataset, suffix)
    dataset = get_dataset(dataset=dataset)

    new_dataset = settings.Dataset(new_dataset,
                                   Classes=dataset.Classes,
                                   label_merging=dataset.label_merging,
                                   base_directory=dataset.base_directory,
                                   lists=dataset.lists.copy())
    new_dataset.lists['testing'] = list()

    flip = True
    for item in base_dataset.lists['testing'][:-1]:
        if flip:
            new_dataset.lists['testing'].append(item)
        flip = not flip

    with open(new_dataset.lists_file, 'w') as fp:
        json.dump(new_dataset.lists, fp, indent=1)
