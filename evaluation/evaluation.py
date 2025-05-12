import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN


def computeVisibillity(pnts, f, cx, cy, width, height):
    
    visibillity = np.full(len(pnts), False)
    img_coords= np.full((len(pnts), 2), [0,0])

    for pnt, i in zip(pnts, range(len(pnts))):
        u = pnt[0]/pnt[2]
        v = pnt[1]/pnt[2]

        u = f*u + cx
        v = f*v + cy

        img_coords[i, :] = [v, u]

        if u >= 0 and v >= 0 and u < width and v < height:
            visibillity[i] = True

    return visibillity, img_coords


def chamferDist(lidar_pnts, img_pnts):
   
    tree = KDTree(img_pnts)
    lid_dists = tree.query(lidar_pnts, workers=-1)[0]

    tree = KDTree(lidar_pnts)
    img_dists = tree.query(img_pnts, workers=-1)[0]

    return lid_dists.mean(), lid_dists.std(ddof=1), img_dists.mean(), img_dists.std(ddof=1)



def computeErrorStats(errfunc, multiDistr=True):
    
    if multiDistr:
        db = DBSCAN(eps=0.01, min_samples=100).fit(errfunc.reshape(-1, 1))
        labels = db.labels_
        n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        labels = np.zeros(len(errfunc))
        n_cluster = 1

    distributions = np.empty((n_cluster, 2))

    for i in range(n_cluster):
        pnts = errfunc[np.where(labels==i)]
        distributions[i,0] = pnts.mean()
        distributions[i,1] = pnts.std(ddof=1)

    return distributions, labels


def visibleLidarPoints(lidar_pnts, calib_const, R, T, R0=np.eye(3)):
    
    Rinv = np.linalg.inv(R)
    Tinv = -1.0* Rinv.dot(T)
    
    lidar_pnts_in_cam_coords = np.linalg.inv(R0).dot((Rinv.dot(lidar_pnts.T).T + Tinv).T).T


    lidar_pnts_in_cam_coords = lidar_pnts_in_cam_coords[np.where(lidar_pnts_in_cam_coords[:,2] > 0)]

    lidar_pnts_in_cam_coords = lidar_pnts_in_cam_coords[np.where(np.linalg.norm(lidar_pnts_in_cam_coords, axis=1) <= calib_const.max_dist)]

    vis, _ = computeVisibillity(lidar_pnts_in_cam_coords, calib_const.f, calib_const.cx, calib_const.cy, calib_const.width, calib_const.height)

    lidar_pnts_in_cam_coords = lidar_pnts_in_cam_coords[np.where(vis)]

    return lidar_pnts_in_cam_coords


def lidarEvaluation(img_pnts, lidar_pnts, calib_const, R, T, R0=np.eye(3)):
    
    lidar_pnts = visibleLidarPoints(lidar_pnts, calib_const, R, T, R0)

    lidar_pnts = R.dot(R0.dot(lidar_pnts.T)).T + T
    img_pnts = R.dot(R0.dot(img_pnts.T)).T + T

    lid_m, lid_s, img_m, img_s = chamferDist(lidar_pnts, img_pnts)
    return lid_m, lid_s, img_m, img_s