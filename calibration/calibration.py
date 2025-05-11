import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import DBSCAN
from scipy import stats as st
from scipy.optimize import least_squares as ls

import timeit

## Function compute initial Rotation and Translation
## Parameter: pnts: np.array with points on the rails
## Returns: T, R
def initCalibration(pnts):
    pnts_mean = pnts.mean(axis=0)

    cov = np.cov((pnts-pnts_mean).T)
    ev, eig = np.linalg.eig(cov)

    ez = eig.T[np.argmin(ev)]

    if (-pnts_mean).dot(ez) < 0:
        ez = -ez
    ez = ez/np.linalg.norm(ez)

    ex_cam = np.array([1,0,0])
    ex = np.cross(ez, ex_cam)
    ex = ex/np.linalg.norm(ex)

    ey = np.cross(ez, ex)
    ey = ey/np.linalg.norm(ey)

    return pnts_mean, np.array([ex, ey, ez])

def projectToZPlane(pnts, T, R):
    
    ex = R[0]
    ey = R[1]
    ez = R[2]

    d = T.dot(ez)
    
    pnts_2d = []
    
    for i in range(len(pnts)):
        z = pnts[i] - T
        k = (d - ez[0]*z[0] - ez[1]*z[1]-ez[2]*z[2])/(ez[0]**2+ez[1]**2+ez[2]**2)

        z1 = z[0] + k*ez[0]
        z2 = z[1] + k*ez[1]
        z3 = z[2] + k*ez[2]

        xn = ex[0]*z1 + ex[1]*z2+ex[2]*z3
        yn = ey[0]*z1 + ey[1]*z2+ey[2]*z3

        pnts_2d.append([xn, yn])

    return np.array(pnts_2d)


def optimize2D(pnts_2d, optimizer = "BF", min_angle = -np.pi/5, max_angle = np.pi/5):
    
    def optFunc(x):

        theta = x[0]

        R2d = np.array([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta),  np.cos(theta)]])

        pnts_2d_rot = pnts_2d.dot(R2d)

        iqr = st.iqr(pnts_2d_rot[:,1])
        n = len(pnts_2d_rot[:,1])
        max = pnts_2d_rot[:,1].max()
        min = pnts_2d_rot[:,1].min()
        
        bin_width = 2*iqr*(n**(-1/3))
        n_bins = (max-min)/bin_width

        bins = np.arange(min, max, bin_width)
        hist_y, _ = np.histogram(pnts_2d_rot[:,1], bins=bins, density=True)            

        n_bins = len(hist_y)
        expected_bin_height = 1.0/(n_bins*bin_width)
        xSq_y = ((hist_y - expected_bin_height)**2/expected_bin_height).sum()

        return np.array([n_bins / xSq_y])
    

    if optimizer == "BF":

        angles = np.arange(min_angle, max_angle, 0.001)
        errfunc  = np.array([angles, np.zeros(len(angles))])
        
        for theta, i in zip(errfunc[0], range(len(errfunc[0]))):
            errfunc[1,i] = optFunc(np.array([theta]))[0]#[0]
            
        opt_angle = errfunc[0, np.argmin(errfunc[1])]

    else:
        if optimizer == "LM":

            res = ls(optFunc, np.array([0]), method='lm')
            opt_angle = res.x[0]

        else:
            print("ERROR: Invalid Optimizer")

    return opt_angle
    

def correctCalibration(pnts, T, R, plot = True, optimizer = "BF", eps = 0.01, min_samples=500):
    #plot = True
    ex = R[0]
    ey = R[1]
    ez = R[2]

    # Create Line of sight as intersection of Planes with normal vectors ex_cam and ez_world
    # Line has form: q + t*w
    exc = np.array([1.0,0.0,0.0])
    ezw = R.T.dot(np.array([0.0,0.0,1.0]))
    ezw = ezw/np.linalg.norm(ezw)
    
    eyc_ezw = exc.dot(ezw)
    
    w = np.cross(exc,ezw)
    w = w/np.linalg.norm(w)
    q = -(ezw.dot(T)*(eyc_ezw))/(1-(eyc_ezw)**2)*exc+(ezw.dot(T))/(1-(eyc_ezw)**2)*ezw

    los_3d = np.zeros((2,3))
    los_3d[0] = q + ((pnts[:,2].min()-q[2])/w[2]) * w
    los_3d[1] = q + ((pnts[:,2].max()-q[2])/w[2]) * w
    
    # Project all points to working z-Plane
    pnts_2d = projectToZPlane(pnts, T, R)
    los_2d = projectToZPlane(los_3d, T, R)

    if plot:
        fig, ax = plt.subplots(2, 2)

        ax[0,0].plot(pnts_2d[:,0], pnts_2d[:,1], 'bo')
        ax[0,0].plot(los_2d[:,0], los_2d[:,1], 'r-')
        
    # Optimize the Angle and correct Normal Vectors
    opt_angle = optimize2D(pnts_2d, optimizer)
   
    ex = ex*np.cos(opt_angle) + np.cross(ez, ex)*np.sin(opt_angle) + ez.dot(ez.dot(ex)*(1-np.cos(opt_angle)))
    ex = ex/np.linalg.norm(ex)

    ey = np.cross(ez, ex)
    ey = ey/np.linalg.norm(ey)
    

    R2d = np.array([[np.cos(opt_angle), -np.sin(opt_angle)], 
                    [np.sin(opt_angle),  np.cos(opt_angle)]])

    pnts_2d = pnts_2d.dot(R2d)

    if plot:
        ax[0,1].plot(pnts_2d[:,0], pnts_2d[:,1], 'bo')
        ax[0,1].plot(los_2d[:,0], los_2d[:,1], 'r-')

    # Find Rail Clusters and centers
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pnts_2d[:,1].reshape(-1, 1))
    labels = db.labels_

    n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
    distributions = np.empty((n_cluster, 2))

    for i in range(n_cluster):
        pnts = pnts_2d[:,1][np.where(labels==i)]
        distributions[i,0] = pnts.mean()
        distributions[i,1] = pnts.std(ddof=1)


    if plot:
        n_bins = 50
        edge = pnts_2d[:,1].max()
        bin_width = edge/n_bins
        bins = np.arange(-edge, edge+bin_width, bin_width)

        hist_y, _ = np.histogram(pnts_2d[:,1], bins=bins, density=True)        
        bin_centers = (bins[1:] + bins[:-1])/2

        ax[1,0].stem(bin_centers, hist_y)

        x = np.linspace(-3, 3, 1000)
        for i in range(n_cluster):
            ax[1,0].plot(x, st.norm.pdf(x, distributions[i, 0], distributions[i,1]))
        plt.savefig("plot_2D_interm.png")


    d2los = np.array([distributions[:,0] - los_2d[:,1].mean(), range(len(distributions[:,0]))])# los_mean

    neg = d2los[:,np.where(d2los[0,:] < 0)]
    pos = d2los[:,np.where(d2los[0,:] > 0)]

    idx_left = int(neg[1,0,[neg[0,0,:].argmax()]])
    idx_right = int(pos[1,0,[pos[0,0,:].argmin()]])

    right_rail = distributions[idx_left,0]
    left_rail = distributions[idx_right,0]


    scale = 1.435/(left_rail - right_rail)
    y_shift = right_rail + (left_rail-right_rail)/2


    T += np.array([ex, ey, ez]).T.dot(np.array([0, y_shift, 0]))
    T *= scale

    if plot:
        pnts_l = pnts_2d[np.where(labels == idx_left)]
        pnts_r = pnts_2d[np.where(labels == idx_right)]
        pnts_n = pnts_2d[np.where(np.logical_and(labels != idx_left, labels != idx_right))]
        
        ax[1,1].plot(pnts_l[:,0], pnts_l[:,1], 'ro')
        ax[1,1].plot(pnts_r[:,0], pnts_r[:,1], 'ro')
        ax[1,1].plot(pnts_n[:,0], pnts_n[:,1], 'bo')

    print(f'Found Scale: {scale}')

    if plot:
        plt.savefig("plot_2D.png")

    return scale, T, np.array([ex, ey, ez])