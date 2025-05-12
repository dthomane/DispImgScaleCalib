import os
import cv2
import open3d as o3d
import numpy as np

import copy

from utils.osdar23 import osdar23extrinsics
from utils.averageCalibration import averageCalibration

from depthmodels.utils import to3D_depth
from evaluation.evaluation import visibleLidarPoints, chamferDist

from calibration.data import CalibConstants, CalibParams


def evaluateOsdar23Scene(
        root, 
        scene, 
        evalConst = CalibConstants(), 
        depthmodel=None, 
        depthmodelName='', 
        segmodelName='', 
        optimizer='BF', 
        evaluationModes = ['scale', 'scaleAndExtrinsic', 'scaleAndAvExtrinsic']
    ):

    imgfolder = os.path.join(root, 'dataset', scene, 'rgb_highres_center')
    lidarfolder = os.path.join(root, 'dataset', scene, 'lidar')
    calibfolder = os.path.join(root, 'calib_results', depthmodelName, scene, segmodelName, optimizer)
    resultfolder = os.path.join(root, 'eval_results', depthmodelName, scene, segmodelName, optimizer)

    for evaluationMode in evaluationModes:
        resfolder = os.path.join(resultfolder, evaluationMode)
        if not os.path.exists(resfolder):
            print(f'Creating Folder for eval results: {resfolder}')
            os.makedirs(resfolder)

    if  "scaleAndAvExtrinsic" in evaluationModes:
        Rav, Tav = averageCalibration(calibfolder)

   
    imgfiles = sorted(os.listdir(imgfolder))
    lidfiles = sorted(os.listdir(lidarfolder))

    if len(imgfiles) != len(lidfiles):
        print("FATAL ERROR: Need as much Lidarfiles as imagefiles")
        assert False


    for imgfile, lidfile in zip(imgfiles, lidfiles):
        imgname = os.path.splitext(imgfile)[0]
        lidname = os.path.splitext(lidfile)[0]

        n_img = int(imgname.split('_')[0])
        n_lid = int(lidname.split('_')[0])

        if n_img != n_lid:  
            print(f'FATAL ERROR: Files numbers do not match: Img: {n_img} to Lidar: {n_lid}, skipping...')
            continue

        if depthmodel is None:
            depthfile = os.path.join(root, 'preprocessed', 'UniDepth', scene, f'{imgname}.tif')
            depthimg = cv2.imread(depthfile, cv2.IMREAD_UNCHANGED)

        else:
            imgfile = os.path.join(imgfolder, imgfile)
            origimg = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)

            depthimg = depthmodel.predict(origimg)


        calibration = CalibParams()
        calibration.load(os.path.join(calibfolder, f'{imgname}.json'))

        gt_extr = osdar23extrinsics()

        pcd = o3d.io.read_point_cloud(os.path.join(lidarfolder, f'{lidname}.pcd'))
        lid_pnts = np.array(pcd.points)

        vsbl_lidar_pnts = visibleLidarPoints(lid_pnts, evalConst, gt_extr.R, gt_extr.T, gt_extr.R0)
        vsbl_lidar_pnts = gt_extr.R.dot(gt_extr.R0.dot(vsbl_lidar_pnts.T)).T + gt_extr.T
        

        to3D = to3D_depth(evalConst)
        image_3D = to3D(depthimg, scale=calibration.s)

        depths = np.sqrt((image_3D**2).sum(axis=2))
        mask_map = depths < evalConst.max_dist
        img_pnts = image_3D[mask_map]

        for evaluationMode in evaluationModes:
            if evaluationMode == "scale":
                Re = gt_extr.R
                Te = gt_extr.T

            elif evaluationMode == "scaleAndExtrinsic":
                Re = np.array(calibration.R)
                Te = np.array(calibration.T)
                Re = Re.dot(np.linalg.inv(gt_extr.R0))

            elif evaluationMode == "scaleAndAvExtrinsic":
                Re = Rav
                Te = Tav
                Re = Re.dot(np.linalg.inv(gt_extr.R0))
            
            else:
                print("ERROR: Unknown evaluation method")
                assert False
            
            loc_pnts = copy.deepcopy(img_pnts)
            
            loc_pnts = Re.dot(gt_extr.R0.dot(loc_pnts.T)).T + Te

            lid_m, lid_s, img_m, img_s = chamferDist(vsbl_lidar_pnts, loc_pnts)

                
            with open(os.path.join(resultfolder, evaluationMode, f'{imgname}.csv'), 'w') as file:
                file.write("mean lidar, std lidar, mean img, std img\n")
                file.write(f'{lid_m}, {lid_s}, {img_m}, {img_s}')

            cd = 0.5*(lid_m + img_m)
            print(f'Evaluated image: {imgname}, mode: {evaluationMode}, chamfer distance: {cd}')

