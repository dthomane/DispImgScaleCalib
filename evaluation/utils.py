import numpy as np
import scipy.stats as st

def overlayColoredErrors(ax, img, lidar_pnts, errfunc, max_err = 10.0):

    error_norm = np.minimum(np.abs(errfunc)/max_err, np.ones(len(errfunc)))
    error_colors = np.array([error_norm, (1-error_norm), np.zeros(len(error_norm))]).transpose()

    ax.imshow(img)
    ax.scatter(lidar_pnts[:,1], lidar_pnts[:,0], c=error_colors)

def overlayDistributions(ax, img, lidar_pnts, lbls):

    e = list(set(lbls))
    colors = np.full((len(e), 3), [0,0,0])

    for j in range(len(e)):
        if j == 0:
            colors[j] = [1,0,0]
        else:
            if j == 1:
                colors[j] = [0,1,0]
            else:
                if j == 2: 
                    colors[j] = [0,0,1]
                else:
                    colors[j] = np.random.rand(3)

    ax.imshow(img)

    for i in range(len(e)):
        ax.scatter(lidar_pnts[np.where(lbls == e[i]),1], lidar_pnts[np.where(lbls == e[i]),0],
                    color=colors[i], label=str(e[i]))
    ax.legend()


def plotStats(ax, stats, min=0, max=2, colors=None):
    
    x = np.linspace(min, max, 100)

    if colors is None:
        colors = ['red'] * len(stats)

    for i in range(len(stats)):
        ax.plot(x, st.norm.pdf(x, stats[i, 0], stats[i,1]), label=str(i), color=colors[i])
        print("Mean: ", stats[i, 0], "Std: ", stats[i, 1])
    ax.legend()