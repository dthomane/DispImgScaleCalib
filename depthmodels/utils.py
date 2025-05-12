import numpy as np
import cv2

class to3D_depth:
    def __init__(self, calib_const):
        self.cx = calib_const.cx
        self.cy = calib_const.cy
        self.f = calib_const.f
        self.max_dist = calib_const.max_dist

        self.standard_dist = self.max_dist

        self.lower_dist_if_no_res = calib_const.lower_dist_if_no_res
        self.min_dist_if_no_res = calib_const.min_dist_if_no_res

    def __call__(self, img, scale=1.0):
        x_over_z = (self.cx - np.arange(img.shape[1])) / self.f
        y_over_z = (self.cy - np.arange(img.shape[0])) / self.f

        z = img / np.sqrt(1. + x_over_z[None,:]**2 + y_over_z[:,None]**2)
        x = x_over_z[None,:] * z
        y = y_over_z[:,None] * z

        return np.concatenate((-scale*x[:,:,None],-scale*y[:,:,None],scale*z[:,:,None]), axis=2)

    def maskFromDepthMap(self, depth_map):
        return depth_map < self.max_dist

    def adjustDistanceThreshold(self):
        self.max_dist = self.max_dist - self.lower_dist_if_no_res

        if self.max_dist < self.min_dist_if_no_res:
            return False
        
        return True

    def resetThrehold(self):
        self.max_dist = self.standard_dist

    def threshold(self):
        return self.max_dist


class to3D_points:
    def __init__(self, calib_const):
        self.max_dist = calib_const.max_dist
        self.standard_dist = self.max_dist

        self.lower_dist_if_no_res = calib_const.lower_dist_if_no_res
        self.min_dist_if_no_res = calib_const.min_dist_if_no_res

    def __call__(self, img, scale=1.0):
        return img*scale

    def maskFromDepthMap(self, points):
        depth_map = np.sqrt((points**2).sum(axis=2))
        return depth_map < self.max_dist

    def adjustDistanceThreshold(self):
        self.max_dist = self.max_dist - self.lower_dist_if_no_res

        if self.max_dist < self.min_dist_if_no_res:
            return False
        
        return True

    def resetThrehold(self):
        self.max_dist = self.standard_dist

    def threshold(self):
        return self.max_dist



class to3D_disp:
    def __init__(self, calib_const):
        self.cx = calib_const.cx
        self.cy = calib_const.cy
        self.f = calib_const.f
        self.min_disp = calib_const.min_disp

        self.standard_disp = self.min_disp

        self.higher_disp_if_no_res = calib_const.higher_disp_if_no_res
        self.max_disp_if_no_res = calib_const.max_disp_if_no_res

    def __call__(self, img, scale=1.0):
        ## Project image to 3D
        Q = np.array(([1.0, 0.0, 0.0, -self.cx],
                    [0.0, 1.0, 0.0, -self.cy],
                    [0.0, 0.0, 0.0, self.f],
                    [0.0, 0.0, 1/scale, 0.0]), dtype=np.float32)

        return cv2.reprojectImageTo3D(img, Q, handleMissingValues=True)

    def maskFromDepthMap(self, depth_map):
        return depth_map > self.min_disp

    def adjustDistanceThreshold(self):
        self.min_disp += self.higher_disp_if_no_res

        if self.min_disp > self.max_disp_if_no_res:
            return False
        
        return True

    def resetThrehold(self):
        self.min_disp = self.standard_disp

    def threshold(self):
        return self.min_disp