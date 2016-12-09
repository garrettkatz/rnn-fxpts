"""
Convenience wrappers around matplotlib plotting functions.
Points are handled in matrix columns rather than separate arguments for separate coordinates.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot(ax, X, *args, **kwargs):
    """
    Plot 2d or 3d points in numpy.array X
    ax should be the matplotlib.Axes (or Axes3D) on which to plot
    X[:,p] should be the p^{th} point to plot
    args and kwargs should be as in matplotlib.Axes.plot
    """
    if X.shape[0]==2:
        ax.plot(X[0,:],X[1,:], *args, **kwargs)
    elif X.shape[0]==3:
        #ax = plt.gca(projection="3d")
        ax.plot(X[0,:],X[1,:],X[2,:],*args, **kwargs)

def scatter(ax, X, *args, **kwargs):
    """
    Scatter-plot 2d or 3d points in numpy.array X
    ax should be the matplotlib.Axes (or Axes3D) on which to plot
    X[:,p] should be the p^{th} point to plot
    args and kwargs should be as in matplotlib.Axes.plot
    """
    if X.shape[0]==2:
        ax.scatter(X[0,:],X[1,:], *args, **kwargs)
    elif X.shape[0]==3:
        #ax = plt.gca(projection="3d")
        ax.scatter(X[0,:],X[1,:],X[2,:],*args, **kwargs)

def text(ax, X, strs, *args, **kwargs):
    """
    Plot text at 2d or 3d points in numpy.array X
    ax should be the matplotlib.Axes (or Axes3D) on which to plot
    X[:,p] should be the p^{th} point at which to plot
    strs[p] should be the p^{th} string to plot
    args and kwargs should be as in matplotlib.Axes.plot
    """
    if X.shape[0]==2:
        ax.text(X[0,:],X[1,:], strs, *args, **kwargs)
    elif X.shape[0]==3:
        #ax = plt.gca(projection="3d")
        ax.text(X[0,:],X[1,:],X[2,:], strs, *args, **kwargs)

def quiver(ax, X, U, *args, **kwargs):
    """
    Plot 2d or 3d vector field in numpy.arrays X and U.
    ax should be the matplotlib.Axes (or Axes3D) on which to plot
    X[:,p] should be the base point for the p^{th} vector
    U[:,p] should be the p^{th} vector to plot
    args and kwargs should be as in matplotlib.Axes.plot
    """
    if X.shape[0]==2:
        ax.quiver(X[0,:],X[1,:],U[0,:],U[1,:], *args, **kwargs)
    elif X.shape[0]==3:
        #ax = plt.gca(projection="3d")
        ax.quiver(X[0,:],X[1,:],X[2,:],U[0,:],U[1,:],U[2,:],*args, **kwargs)

def plotNd(X, lims, *args):
    """
    Plot Nd points in numpy.array X
    Every two dimensions are shown on a separate subplot
    The last dimension is omitted when N odd
    X[:,p] should be the p^{th} point to plot
    lims[n,0] and lims[n,1] are low and high plot limits for the n^{th} dimension
    args should be as in matplotlib.Axes.plot
    """
    num_subplots = int(X.shape[0]/2);
    num_rows = np.floor(np.sqrt(num_subplots))
    num_cols = np.ceil(num_subplots/num_rows)
    for subplot in range(num_subplots):
        ax = plt.subplot(num_rows, num_cols, subplot+1)
        ax.plot(X[2*subplot,:], X[2*subplot+1,:], *args)
        ax.set_xlim(lims[0,:])
        ax.set_ylim(lims[1,:])

def set_lims(ax, lims):
    """
    Set all 2d or 3d plot limits at once.
    ax is the matplotlib.Axes (or Axes3D) on which to plot
    lims[0,:] are xlims, etc.
    """
    ax.set_xlim(lims[0,:])
    ax.set_ylim(lims[1,:])
    if len(lims)>2:
        ax.set_zlim(lims[2,:])

def lattice(mins, maxes, samp):
    """
    Samples Nd points on a regularly spaced grid
    mins[i], maxes[i] are the grid extents in the i^{th} dimension
    samp is the number of points to sample in each dimension
    Returns numpy.array G, where
      G[:,n] is the n^{th} grid point sampled
    """
    G = np.mgrid[tuple(slice(mins[d],maxes[d],(samp*1j)) for d in range(len(mins)))]
    G = np.array([g.flatten() for g in G])
    return G

def plot_trisurf(ax, X, *args, **kwargs):
    """
    Plots points in numpy.array X as a surface.
    ax is the matplotlib.Axes3D on which to plot
    X[:,p] is the p^{th} point
    X[2,:] is shown as a surface over X[1,:] and X[2,:]
    args and kwargs should be as in matplotlib.Axes3D.plot_trisurf
    """
    ax.plot_trisurf(X[0,:],X[1,:],X[2,:],*args, **kwargs)
