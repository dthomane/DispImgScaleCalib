import argparse
import os

from utils.osdar23 import calib_scenes
from calibration.osdar23 import calibrateOsdar23Scene
from calibration.data import CalibConstants

from depthmodels.createmodel import create_model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                prog = 'Calibrate Osdar23 Dataset Metric Scaling and Extrinsic Calibration of Monocular Neural Network-Derived 3D Point Clouds in Railway Applications')

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


    if args.segmentation == 'preprocessed':
        segmodel = None
    
    elif args.segmentation == 'internimage':

        from segmodels.internimage import model_internimage
        config = os.path.join('config', 'upernet_internimage_t_512x1024_160k_rs19.py')
        internimage = model_internimage(config, args.internimage_chekpoint)

    elif args.segmentation == 'osdar23gt':
        segmodel = 'osdar23gt'

    
    for scene in calib_scenes:

        if args.depthmodel is None:
            depthmodeName = 'Unidepth'
        else:
            depthmodeName = args.depthmodel

        if args.segmentation == 'preprocessed':
            segmodelName = 'internimage'
        else:
            segmodelName = args.segmentation
        
        
        resultfolder = os.path.join(args.dataset, 'calib_results', depthmodeName, scene, segmodelName, args.optimizer)
        if not os.path.exists(resultfolder):
            print(f'Creating Folder for calib results: {resultfolder}')
            os.makedirs(resultfolder)

        calibConst = CalibConstants()
        calibConst.max_dist = 20
        calibConst.min_disp = 15

        calibrateOsdar23Scene(
            args.dataset,
            scene,
            resultfolder,
            calib_const=calibConst,
            depthmodel=depthmodel,
            segmodel=segmodel,
            sparsify=args.sparsify,
            optimizer=args.optimizer
        )