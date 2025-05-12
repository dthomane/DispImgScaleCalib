import os
import numpy as np

from calibration.data import CalibParams

def averageCalibration(pathToCalibFolder):

    files = os.listdir(pathToCalibFolder)

    cnt = 0.0
    Tav = np.zeros(3)
    Rav = np.zeros((3,3))

    for file in files:

        ## Read Files
        calib_data = CalibParams()
        calib_data.load(os.path.join(pathToCalibFolder, file))

        ## Cam general coords to datset specific
        R = np.array(calib_data.R)
        T = np.array(calib_data.T)

        Tav += T
        Rav += R
        cnt += 1.0

        #dR = R_g.dot(R.transpose())
        #dPhi = np.arccos((np.trace(dR)-1)/2)

    Rav /= cnt
    Tav /= cnt

    return Rav, Tav
