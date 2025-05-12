import os
import cv2
import numpy as np
import open3d as o3d

from utils.osdar23 import osdar23extrinsics

from calibration.data import CalibConstants
from segmodels.utils import LBL_RAIL_RAISED
from segmodels.osdar23gt import osdar23gt

from depthmodels.utils import to3D_depth

from calibration.calibration import initCalibration, correctCalibration
from evaluation.evaluation import visibleLidarPoints, chamferDist

def run(depthmodel=None, segmodel=None, sparsify = 0.0):

    print(f'Starting calibration example')

    ## We use an image from the scene 14_signals_station_14.3 as example here
    this_dir = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.join(this_dir, '14_signals_station_14.3')

    ## We do not need the raw image file as we use preprocessed depth and segmentation here
    if depthmodel is not None or segmodel is not None:
        imgfile = os.path.join(example_dir, 'rgb_highres_center', '017_1631450781.800000025.png')
        origimg = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)


    if depthmodel is None:
        ## Load depth
        depthfile = os.path.join(example_dir, 'preprocessed', 'unidepth', '017_1631450781.800000025.tif')
        depthimg = cv2.imread(depthfile, cv2.IMREAD_UNCHANGED)

    else:
        depthimg = depthmodel.predict(origimg)
        
    
    if segmodel is None:
        ## Load segmentation
        segfile = os.path.join(example_dir, 'preprocessed', 'internimage', '017_1631450781.800000025.png')
        segimg = cv2.imread(segfile, cv2.IMREAD_UNCHANGED)
        segimg = cv2.resize(segimg, (4112, 2504), interpolation = cv2.INTER_LINEAR)
        
    elif segmodel == 'osdar23gt':
        segimg = osdar23gt(this_dir, '14_signals_station_14.3').predict('017_1631450781.800000025.png')
    
    else:
        segimg = segmodel.predict(origimg)


    ## Calib constats have default intrinsics and width/height from osdar23 dataset
    ## We initialize 3D reprojection helper class
    calib_const = CalibConstants()
    to3D = to3D_depth(calib_const)

    ## in Railsem19, rails are class rail-raised (17)
    seg_msk = segimg == LBL_RAIL_RAISED
    msk = np.logical_and(to3D.maskFromDepthMap(depthimg), seg_msk)

    points = to3D(depthimg)
    points = points[msk]

    ## Uniformely exclude points speed up the calibration but may lead to poorer results
    if sparsify > 0.0:
        sparsemask = np.random.rand(len(points)) > sparsify
        points = points[sparsemask]

    ## Main calibration procedure
    T, R = initCalibration(points)
    s, T, R = correctCalibration(points, T, R, False)

    print(f'Done calibration. scale-factor is {s}')
    print(f'Start evaluation')  
    
    ## Eval
    gt_extr = osdar23extrinsics()

    ## Compute visible lidar points
    pcd = o3d.io.read_point_cloud(os.path.join(example_dir, 'lidar', '017_1631450781.799982000.pcd'))
    lid_pnts = np.array(pcd.points)

    vsbl_lidar_pnts = visibleLidarPoints(lid_pnts, calib_const, gt_extr.R, gt_extr.T, gt_extr.R0)
    vsbl_lidar_pnts = gt_extr.R.dot(gt_extr.R0.dot(vsbl_lidar_pnts.T)).T + T


    ## Reproject raw unscaled
    image_raw_3D = to3D(depthimg)

    depths = np.sqrt((image_raw_3D**2).sum(axis=2))
    mask_map = depths < calib_const.max_dist

    img_pnts_raw = image_raw_3D[mask_map]
    img_pnts_raw = gt_extr.R.dot(gt_extr.R0.dot(img_pnts_raw.T)).T + T
    
    raw_lid_m, _, raw_img_m, _ = chamferDist(vsbl_lidar_pnts, img_pnts_raw)
    cd_raw = 0.5*(raw_lid_m + raw_img_m)
    

    ## Reproject scaled
    image_3D = to3D(depthimg, scale=s)

    depths = np.sqrt((image_3D**2).sum(axis=2))
    mask_map = depths < calib_const.max_dist

    img_pnts = image_3D[mask_map]
    img_pnts = gt_extr.R.dot(gt_extr.R0.dot(img_pnts.T)).T + T

    lid_m, _, img_m, _ = chamferDist(vsbl_lidar_pnts, img_pnts)
    cd = 0.5*(lid_m + img_m)


    print(f'Done evaluation: Chamfer Distance raw {cd_raw} and calibrated: {cd}')  
    print('Showing results in open3d: green is lidar, blue is calibrated reprojection, red is raw reprojection')

    ## 3D-Visualisation
    viewer = o3d.visualization.VisualizerWithKeyCallback()
    viewer.create_window()

    lidar_o3d = o3d.geometry.PointCloud()
    lidar_o3d.points = o3d.utility.Vector3dVector(vsbl_lidar_pnts)
    lidar_o3d.paint_uniform_color([0,1,0])

    img_raw = o3d.geometry.PointCloud()
    img_raw.points = o3d.utility.Vector3dVector(img_pnts_raw)
    img_raw.paint_uniform_color([1,0,0])

    img_cal = o3d.geometry.PointCloud()
    img_cal.points = o3d.utility.Vector3dVector(img_pnts)
    img_cal.paint_uniform_color([0,0,1])

    frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame()

    viewer.add_geometry(frame_world)
    viewer.add_geometry(lidar_o3d)
    viewer.add_geometry(img_cal)
    viewer.add_geometry(img_raw)

    viewer.run()
    viewer.destroy_window()