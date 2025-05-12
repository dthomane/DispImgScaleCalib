import argparse
import os

from utils.osdar23 import calib_scenes
from calibration.data import CalibConstants
from evaluation.osdar23 import evaluateOsdar23Scene

from depthmodels.createmodel import create_model

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                prog = 'Evaluate Osdar23 Dataset Metric Scaling and Extrinsic Calibration of Monocular Neural Network-Derived 3D Point Clouds in Railway Applications')

    parser.add_argument(
        'dataset', 
        type=str, 
        help='Root of the dataset structure', 
    )

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
        '--optimizer', 
        type=str, 
        default='BF', 
        help='Method for Angle optimization BF: Brute Force, or LM: Levenberg Marquardt', 
        choices=['BF', 'LM']
    )

    args = parser.parse_args()

    if args.depthmodel is not None:
        depthmodel = create_model(args)
    else:
        depthmodel = None



    for scene in calib_scenes:

        if args.depthmodel is None:
            depthmodeName = 'Unidepth'
        else:
            depthmodeName = args.depthmodel

        if args.segmentation == 'preprocessed':
            segmodelName = 'internimage'
        else:
            segmodelName = args.segmentation
        
        evalConst = CalibConstants()
        evalConst.max_dist = 50
        evalConst.min_disp = 20

        evaluateOsdar23Scene(
            args.dataset, 
            scene, 
            depthmodel=depthmodel,
            evalConst=evalConst,
            depthmodelName=depthmodeName, 
            segmodelName=segmodelName, 
            optimizer=args.optimizer
        )