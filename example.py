import os
import argparse

from depthmodels.createmodel import create_model
from example.example import run

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                prog = 'Simple Example of Metric Scaling and Extrinsic Calibration of Monocular Neural Network-Derived 3D Point Clouds in Railway Applications',
                description = 'This is a simple example of usage of the official implementation. Leave models to NONE and the preprocessed data included in this repository is used')

    parser.add_argument(
        '--depthmodel', 
        type=str, 
        default=None, 
        help='Name of depthmodel (None is preprocessed)', 
        choices=[None, 'depthpro', 'unidepth', 'adabins', 'depthanything', 'midas']
    )

    parser.add_argument(
        '--depthmodelrepo', 
        type=str, 
        default=None, 
        help='If you defined a depthmodel (--depthmodel) this needs to be the path to their repository. '
    )

    parser.add_argument(
        '--segmentation', 
        type=str, 
        default='preprocessed', 
        help='Method of segmentation', 
        choices=['preprocessed', 'internimage', 'osdar23gt']
    )

    parser.add_argument(
        '--internimage_chekpoint', 
        type=str, 
        default=None, 
        help='If you choose internimage (--segmentation) this needs to be the path to pretrained weights (iter_160000.pth)'
    )

    parser.add_argument(
        '--sparsify', 
        type=float, 
        default=0.0, 
        help='This is a sparsify factor for uniformely exclude points before calibrating. This speed up the calibration but may lead to poorer results'
    )


    args = parser.parse_args()

    if args.depthmodel is not None:
        depthmodel = create_model(args)
    else:
        depthmodel = None


    if args.segmentation == 'preprocessed':
        segmodel = None
    
    elif args.segmentation == 'internimage':

        from segmodels.internimage import model_internimage
        config = os.path.join('config', 'upernet_internimage_t_512x1024_160k_rs19.py')
        internimage = model_internimage(config, args.internimage_chekpoint)

    elif args.segmentation == 'osdar23gt':
        segmodel = 'osdar23gt'

    run(depthmodel, segmodel, args.sparsify)