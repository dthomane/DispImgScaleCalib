import os
import cv2
import numpy as np

from segmodels.utils import LBL_RAIL_RAISED

from depthmodels.utils import to3D_depth
from calibration.data import CalibConstants, CalibParams

from calibration.calibration import initCalibration, correctCalibration
from segmodels.osdar23gt import osdar23gt


def calibrateOsdar23Scene(root, scene, calib_const = CalibConstants(), depthmodel=None, segmodel=None, optimizer='BF', sparsify=0.0):

    resultfolder = os.path.join(root, 'calib_results', 'UniDepth', scene, 'internimage', optimizer)
    
    if not os.path.exists(resultfolder):
        print(f'Creating Folder for calib results: {resultfolder}')
        os.makedirs(resultfolder)


    imgfolder = os.path.join(root, 'dataset', scene, 'rgb_highres_center')
    imgfiles = sorted(os.listdir(imgfolder))


    for imgfile in imgfiles:
        imgname = imgfile.split('.')[0]

        if depthmodel is not None or segmodel is not None:
            imgfile = os.path.join(imgfolder, imgfile)
            origimg = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)


        if depthmodel is None:         
            depthfile = os.path.join(root, 'preprocessed', 'UniDepth', scene, f'{imgname}.tif')
            depthimg = cv2.imread(depthfile, cv2.IMREAD_UNCHANGED)

        else:
            depthimg = depthmodel.predict(origimg)
            
        
        if segmodel is None:
            ## Load segmentation
            segfile = os.path.join(root, 'preprocessed', 'internimage', f'{imgname}.png')
            segimg = cv2.imread(segfile, cv2.IMREAD_UNCHANGED)
            segimg = cv2.resize(segimg, (4112, 2504), interpolation = cv2.INTER_LINEAR)
            
        elif segmodel == 'osdar23gt':
            segimg = osdar23gt(os.path.join(root, 'dataset'), scene).predict(imgfile)
        
        else:
            segimg = segmodel.predict(origimg)



        found_result = False
        cur_accuracy_thresh = calib_const.accuracy_thresh
        to3D.resetThrehold()
        
        while not found_result:

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
            try:
                T, R = initCalibration(points)
                s, T, R = correctCalibration(points, T, R, False, optimizer)
            except:
                adjPossible = to3D.adjustDistanceThreshold()

                if not adjPossible:
                    print("ERROR: No result found for image: ", depthfile)
                    break

                print(f'Lower distance threshold due to no result. New Threshold: {to3D.threshold()}')
                continue

            found_result = True
            break

            
        if found_result:
            # Save calibration
            resultfile = os.path.join(resultfolder, os.path.splitext(depthfile)[0] + ".json")

            calibdata = CalibParams()
            calibdata.T = T.tolist()
            calibdata.R = R.tolist()
            calibdata.s = s
            calibdata.f = calib_const.f
            calibdata.cx = calib_const.cx
            calibdata.cy = calib_const.cy
            calibdata.acc = cur_accuracy_thresh
            calibdata.min_disp = calib_const.min_disp
            calibdata.max_dist = calib_const.max_dist 
            calibdata.save(resultfile)

            print(f'Saved calibration to: {resultfile}')
