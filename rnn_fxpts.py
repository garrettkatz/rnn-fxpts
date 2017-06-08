"""
Library for fixed point location in recurrent neural networks using directional fibers.
"""
import os
import sys
import time
import pickle as pkl
import itertools as it
import numpy as np
import scipy.optimize as spo
import plotter as ptr
import matplotlib as mpl
import matplotlib.pyplot as plt

def hardwrite(f,data):
    """
    Force file write to disk
    """
    if f.name == os.devnull: return
    f.write(data)
    f.flush()
    os.fsync(f)

def eps(x):
    """
    Returns the machine precision at x.
    I.e., the distance to the nearest distinct finite-precision number.
    Applies coordinate-wise if x is a numpy.array.
    """
    return np.fabs(np.spacing(x))

def mldivide(A, B):
    """
    Returns x, where x solves Ax = B. (A\B in MATLAB)
    """
    return np.linalg.lstsq(A,B)[0]

def solve(A, B):
    """
    Returns x, where x solves Ax = B.
    Assumes A is invertible.
    """
    return np.linalg.solve(A,B)

def mrdivide(B,A):
    """
    Returns x, where x solves B = xA. (B/A in MATLAB)
    """
    return np.linalg.lstsq(A.T, B.T)[0].T

def get_connected_components(V, neighbors=None):
    """
    Find all connected components in an adjacency graph.
    Assumes the nodes of the adjacency graph are points in Euclidean space.
    V should be a numpy.array, where V[:,p] is the p^{th} node.
    Returns a numpy.array components, where
      components[p]==components[q] iff V[:,p], V[:,q] are connected.
    neighbors should be a function handle that returns a boolean numpy.array.
    The boolean array should satisfy
      neighbors(X, y)[q] == True iff X[:,p], y are neighbors in the graph.
    """
    # Default neighbor criteria
    if neighbors==None:
        neighbors = lambda X, y: (np.fabs(X-y) < 10*eps(y)).all(axis=0)

    # Initialize each point in isolated component
    components = np.arange(V.shape[1])
    # Merge components one point at a time
    for p in range(V.shape[1]):
        # Index neighbors to current point
        n = neighbors(V[:,:p+1], V[:,[p]])
        # Merge components containing neighbors
        if len(components[:p+1][n]) == 0:
            print('comp',components)
            print('comp[:p+1]',components[:p+1])
            print('n',n)
            print('comp[:p+1][n]',components[:p+1][n])
            print('|V|,p',V.shape,p)
        components[:p+1][n] = components[:p+1][n].min()

    return components

def get_unique_points(V, neighbors=None):
    """
    This is a helper function, consider get_unique_fxpts instead.
    Extract "unique" points from a set of duplicates.
    V and neighbors should be as in get_connected_components.
    Neighboring points are considered identical.
    One unique representative is selected from each connected component in the neighbor graph.
    returns V_unique, components, where
      V_unique[:,p] is the p^{th} unique representative, and
      components is as in get_connected_components.
    """
    # Get connected components of neighbor graph
    components = get_connected_components(V, neighbors)

    # Extract representatives from each component
    _, idx = np.unique(components, return_index=True)
    V_unique = V[:,idx]

    return V_unique, components

def get_unique_points_recursively(V, neighbors=None, base=2):
    """
    This is a helper function, consider get_unique_fxpts instead.
    V and neighbors should be as in get_connected_components.
    This is a recursive variant of get_unique_points with better performance.
    The base case of the recursion is determined by the base parameter.
    Returns V_unique as in get_unique_points.
    """
    if V.shape[1] <= base:
        V_unique, _ = get_unique_points(V, neighbors=neighbors)
    else:
        split = int(V.shape[1]/2)
        V_split = [get_unique_points_recursively(V[:,:split], neighbors=neighbors),
                   get_unique_points_recursively(V[:,split:], neighbors=neighbors)]
        V_unique, _ = get_unique_points(np.concatenate(V_split,axis=1), neighbors=neighbors)
    return V_unique

def get_unique_fxpts(W, fxV):
    """
    Extracts "unique" fixed points from a set of duplicates.
    W is the weight matrix (a numpy.array).
    fxV[:,p] is the p^{th} (potentially duplicate) fixed point.
    Returns the unique fixed points in fxV_unique, where
      fxV_unique[:,q] is the q^{th} unique fixed point.
    """
    neighbors = lambda X, y: identical_fixed_points(W, X, y)[0]
    fxV_unique = get_unique_points_recursively(fxV, neighbors=neighbors)
    return fxV_unique

def estimate_forward_error(W, V):
    """
    Estimates the numerical forward error in numpy.tanh(W.dot(V))-V.
    Returns the numpy.array margin, where
      margin[i,j] == the forward error bound on (numpy.tanh(W.dot(V))-V)[i,j].
    """
    e_sigma = 5
    N = V.shape[0]
    W_abs = np.fabs(W)
    V_eps = eps(V)
    tWV_eps = eps(np.tanh(np.dot(W,V)))
    margin = np.dot(W_abs, V_eps)
    margin += N*eps(np.dot(W_abs, np.fabs(V)))
    margin += e_sigma * tWV_eps
    margin += V_eps
    margin += np.maximum(tWV_eps, V_eps)
    return margin

def fixed_within_eps(W, V):
    """
    Detects points that are certainly not fixed, based on forward error bounds.
    W should be the weight matrix (a numpy.array).
    V should be a numpy.array of points to check, where V[:,p] is the p^{th} point.
    returns numpy.arrays fixed and margin, where
      fixed[p]==False only if V[:,p] is certainly not fixed
      margin is as in estimate_forward_error
    """
    margin = estimate_forward_error(W, V)
    fixed = (np.fabs(np.tanh(W.dot(V))-V) < margin).all(axis=0)
    return fixed, margin

def identical_fixed_points(W, V, v, Winv=None):
    """
    Looks for identical fixed points based on Taylor expansion and forward error.
    W should be the weight matrix (a numpy.array).
    V should be a numpy.array where each V[:,p] is a fixed point.
    v should be an (N by 1) numpy.array representing a single fixed point.
    Winv should be the inverse of W, unless None, in which case it is computed.
    Returns identical, RR, R, where
      identical[p]==True iff V[:,p] is identical to v
      RD[p]: the relative distance from V[:,p] to v (as a multiple of R)
      R: the radius around v past which another fixed point is considered distinct
    """
    if Winv is None: Winv = np.linalg.inv(W)
    # sig'' has a maximum of sqrt(16/27) obtained at input arctanh(sqrt(1/3))
    N = W.shape[0]
    E = estimate_forward_error(W,np.ones((N,1))).max()
    D2 = np.sqrt(16./27.)
    Df = (1-np.tanh(W.dot(v))**2)*W - np.eye(N)
    s_min = np.linalg.norm(Df.dot(Winv), ord=-2)
    # R = (s_min - np.sqrt(s_min**2 - 4*D2*np.sqrt(N)*E))/D2
    det = s_min**2 - 8*D2*np.sqrt(N)*E
    if det < 0:
        R = 0
        RD = np.inf*np.ones(V.shape[1])
        # identical = np.zeros(V.shape[1],dtype=bool)
        identical = (V == v).all(axis=0) # keep truly identical points
    else:
        R = (s_min - np.sqrt(det))/D2
        RD = np.sqrt((W.dot(V-v)**2).sum(axis=0))/R
        identical = (RD < 1)
    return identical, RD, R

def brute_fiber(W, c, max_iters=1000):
    """
    Finds all components of a fiber in 2 or 3d through brute force find grid sampling.
    W should be the weight matrix (a numpy.array)
    c should be the direction vector (a single column numpy.array)
    max_iters is the maximum steps allowed for traversal on each grid sample
    returns fxpts, fiber, where
      fxpts[i][:,p] is the p^{th} fixed point found, and
      fiber[i][:,n] is the n^{th} point along traversal,
    for the traversal from the i^{th} grid sample.
    The component and its negative are indexed by s == 0 or 1, respectively.
    """
    # normalize c
    c = c/np.sqrt((c*c).sum())

    # Sample grid
    samp = 100
    lim = 1.0
    N = W.shape[0]
    if N == 2:
        V = np.mgrid[-lim:lim:(samp*1j), -lim:lim:(samp*1j)]
        V = np.array([V[i].flatten() for i in [0,1]])
    else:
        V = np.mgrid[-lim:lim:(samp*1j), -lim:lim:(samp*1j), -lim:lim:(samp*1j)]
        V = np.array([V[i].flatten() for i in [0,1,2]])
    C = np.tanh(W.dot(V))-V
    C = C/np.sqrt((C*C).sum(axis=0))

    # local neighborhood minima of c.T.dot(C)
    ndelta = (np.arange(2**N) / (2**np.arange(N)[:,np.newaxis])) % 2 # neighbors: little cube vertices around each point in A
    ndelta = 2*ndelta-1
    ndelta = np.concatenate((ndelta, np.eye(N), -np.eye(N)), axis=1) # add face neighbors
    ndelta = ndelta*lim/(samp-1.0) # neighbors spaced according to grid sampling
    nV = V[:,:,np.newaxis] + ndelta[:,np.newaxis,:]
    nC = np.zeros(nV.shape)
    for n in range(nV.shape[2]):
        nC[:,:,n] = np.tanh(W.dot(nV[:,:,n]))-nV[:,:,n]
    nC = nC/np.sqrt((nC*nC).sum(axis=0))
    neighbor_dots = np.fabs((c[:,:,np.newaxis]*nC).sum(axis=0)).max(axis=1) # maximum dot of c with neighbors
    self_dots = np.fabs(c.T.dot(C)) # dot of c with self

    idx = np.flatnonzero(self_dots.flatten() > neighbor_dots) # closer dot to c than all neighbors
    # V = V[:,idx]
    # return [], V

    # Traverse components from each seed
    fxV, VA = [], []
    for i in idx:
        a = (np.tanh(W.dot(V[:,[i]]))-V[:,[i]])/c
        a = a[~np.isnan(a)].mean()
        va = np.concatenate((V[:,[i]], [[a]]), axis=0)
        _, fxV_i, VA_i, _, _, _, _ = traverse(W, va=va, c=c, max_traverse_steps = max_iters)
        # post-process fixed
        fxV_i, _ = post_process_fxpts(W, fxV_i)
        for s in [-1, 1]:
            fxV.append(s*fxV_i)
            VA.append(s*VA_i)
    return fxV, VA

def drive_initial_va(W, va, c, max_nr_iters, drive_tol):
    """
    Drives an initial seed point, not quite on the fiber, to the fiber (up to machine precision).
    W is the weight matrix (a numpy.array)
    va is the initial seed for traversal (an Nx1 numpy.array)
    c is the direction vector (an Nx1 numpy.array)
    max_nr_iters is the maximum number of iterations for Newton-Raphson refinement
    drive_tol is a tolerance at which Newton-Raphson refinement may terminate
    returns va_refined, the result of driving va to the fiber.
    """
    N = W.shape[0]
    I = np.eye(N)
    for i in it.count(0):
        if i == max_nr_iters: break
        F = np.tanh(W.dot(va[:N,:])) - va[:N,:] - va[N]*c
        if (np.fabs(F) < drive_tol).all(): break
        D = 1 - np.tanh(W.dot(va[:N,:]))**2
        J = np.concatenate((D*W - I, -c), axis=1)
        va = va - mldivide(J, F)
    return va

def calc_delta(e, x):
    """
    Calculates the delta function used by Thm. 1 (Katz and Reggia 2017).
    e should be a (1 by K) numpy.array of epsilons
    x should be an (N by 1) numpy.array of inputs to tanh'
    returns delta, where
      delta[i,j] is the largest d for which |tanh'(x[i]+d) - tanh'(x[i])| <= e[j].
    """
    # e should be 1xK, x should be Kx1
    x = np.fabs(x)
    dsig = 1. - np.tanh(x)**2
    dsig_p = dsig + e
    dsig_m = dsig - e
    delta_p, delta_m = np.inf*np.ones(dsig_p.shape), np.inf*np.ones(dsig_m.shape)
    idx_p, idx_m = (dsig_p > 0) & (dsig_p <= 1), (dsig_m > 0) & (dsig_m <= 1)
    delta_p[idx_p] = np.arctanh(np.sqrt(1 - dsig_p[idx_p]))
    delta_m[idx_m] = np.arctanh(np.sqrt(1 - dsig_m[idx_m]))
    delta_p, delta_m = np.fabs(delta_p - x), np.fabs(delta_m - x)
    delta = np.minimum(delta_p, delta_m)
    return delta

def s_min_calc(_J_):
    """
    Returns the minimum singular value of numpy.array _J_
    """
    return np.linalg.norm(_J_, ord=-2)

def mu_calc(Wv, delta):
    """
    Calculates the mu function used by Thm. 1 (Katz and Reggia 2017).
    Wv should be the (N by 1) numpy.array W.dot(v) for some point v.
    delta should be an (N by K) array of deltas to Wv.
    returns (N by K) numpy.array mu, where
      mu[i,j] is the maximum of tanh'' over (Wv[i] - delta[i,j], Wv[i] + delta[i,j]).
    """
    # sig'' has a maximum of sqrt(16/27) obtained at input arctanh(sqrt(1/3))
    tWv_p, tWv_m = np.tanh(Wv + delta), np.tanh(Wv - delta)
    mu_p = np.fabs(tWv_p - tWv_p**3)
    mu_m = np.fabs(tWv_m - tWv_m**3)
    mu = np.maximum(mu_p, mu_m)
    mu[np.fabs(Wv - np.arctanh(np.sqrt(1./3.))) < delta] = np.sqrt(16./27.)/2.
    return mu

def dsig(x):
    """
    Returns tanh'(x), the derivative of tanh(x)
    """
    return 1. - np.tanh(x)**2

def traverse_step_size(_W_, _Winv_, D, J, va, c, z, num_samples=16):
    """
    Determines a step size according to Thm 1 (Katz and Reggia 2017).
    _W_ should be the augmented weight matrix (an N+1 by N+1 numpy.array)
    _Winv_ should be its inverse
    D should be the diagonal matrix with tanh'(W.dot(va)) along the diagonal
    J should be DF, the Jacobian of F (an N by N+1 numpy.array)
    va should be the current fiber point (an N+1 by 1 numpy.array), where
      va[:N] == v and va[N] == alpha
    c should be the direction vector (an N by 1 numpy.array)
    z should be the tangent vector (an N+1 by 1 numpy.array)
    num_samples should be the number of epsilon to try in (0, lambda)
    """
    N = va.shape[0] - 1
    W = _W_[:N,:N]
    _J_ = np.concatenate((J, z.T), axis=0).dot(_Winv_)
    s_min = s_min_calc(_J_)
    e = np.linspace(0, s_min, num_samples+2)[np.newaxis,1:-1]
    Wv = np.fabs(W.dot(va[:N,:]))
    delta = calc_delta(e, Wv)
    mu = mu_calc(Wv, delta)
    rho = mu.max(axis=0)[np.newaxis,:]/(s_min - e)
    delta = delta.min(axis=0)[np.newaxis,:]
    # theta = delta/(1 + rho*delta) # doesn't account for infinite delta possibility...
    theta = np.empty(delta.shape)
    idx = ~np.isinf(delta)
    theta[idx] = delta[idx]/(1+rho[idx]*delta[idx])
    theta[~idx] = 1/rho[~idx]
    max_idx = theta.flatten().argmax()
    return theta.flat[max_idx]/np.sqrt((_W_.dot(z)**2).sum()), rho.flat[max_idx], s_min

def take_traverse_step(W, I, c, va, z, step_size, max_nr_iters, nr_tol, verbose=1):
    """
    Takes step according to Thm 1 (Katz and Reggia 2017).
    W should be the weight matrix (N by N numpy.array)
    I should be the identity matrix (N by N numpy.array)
    c should be the direction vector (an N by 1 numpy.array)
    va should be the current fiber point (an N+1 by 1 numpy.array), where
      va[:N] == v and va[N] == alpha
    z should be the tangent vector (an N+1 by 1 numpy.array)
    step_size should be as returned by traverse_step_size
    max_nr_iters is the maximum number of iterations for Newton-Raphson refinement
    nr_tol is a tolerance at which Newton-Raphson refinement may terminate
    returns va, F, where
      va is the new point after the step
      F is the residual value of F at the new point.
    """
    N = W.shape[0]
    va_start = va
    g_root = np.zeros((N+1,1))
    va = va_start + z*step_size # fast first step
    for drive_step in it.count(0):
        if drive_step == max_nr_iters: break
        tWv = np.tanh(W.dot(va[:N,:]))
        F = tWv-va[:N,:]-va[N]*c
        if (np.fabs(F) < nr_tol).all(): break
        # gg = np.concatenate((-F, [[0.]]),axis=0)
        g_root[:N] = -F
        D = 1 - tWv**2
        J = np.concatenate((D*W - I, -c), axis=1)
        Dg = np.concatenate((J, z.T), axis=0)
        va = va + solve(Dg, g_root)
    return va, F

def get_term(W, c):
    """
    Get the termination criteria for traversal (Katz and Reggia 2017)
    W is the weight matrix (N by N numpy.array)
    c is the direction vector (N by 1 numpy.array)
    returns term, the bound on alpha past which no more fixed points will be found
    """
    D_bound = min(1, 1/np.linalg.norm(W,ord=2))
    term = ((np.arctanh(np.sqrt(1 - D_bound)) + np.fabs(W).sum(axis=1))/np.fabs(W.dot(c))).max()
    return term

def calc_z_new(J, z):
    """
    Calculate the new tangent vector after the numerical step
    J should be the Jacobian of F at the new point after the step (N by N+1 numpy.array)
    z should be the previous tangent vector before the step (N+1 by 1 numpy.array)
    returns z_new, the tangent vector after the step (N+1 by 1 numpy.array)
    """
    N = J.shape[0]
    z_new = solve(np.concatenate((J,z.T), axis=0), np.concatenate((np.zeros((N,1)), [[1]]), axis=0)) # Fast J null-space
    z_new = z_new / np.sqrt((z_new**2).sum()) # faster than linalg.norm
    return z_new

def traverse(W, va=None, c=None, max_nr_iters=2**8, nr_tol=2**-32, max_traverse_steps=None, max_fxpts=None, logfile=None, max_step_size=None):
    """
    Find fixed points via fiber traversal.
    run_solver invokes this method before post-processing the resulting fixed points.
    W is the weight matrix (N by N numpy.array)
    va is the initial point (N+1 by 1 numpy.array)
      if None, traversal starts at the origin
    c is the direction vector (N by 1 numpy.array)
      if None, a random direction vector is chosen
    max_nr_iters is the maximum iterations allowed for Newton-Raphson refinement
    nr_tol is the tolerance at which Newton-Raphson refinement can terminate
    max_traverse_steps is the maximum number of steps allowed for traversal
      if None, traversal continues until another termination criteria is met
    max_fxpts is the number of fixed points at which traversal can terminate
      if None, traversal continues until another termination criteria is met
    logfile is a file object open for writing that records progress
      if None, no progress is recorded

    returns status, fxV, VA, c, step_sizes, s_mins, residuals, where
      status is one of
        "Success", "Max steps reached", "Max fxpts found", "Closed loop detected"
      fxV[:,p] is the p^{th} (un-post-processed) fixed point found  
      VA[:,n] is the n^{th} point along the fiber  
      c is the direction vector that was used (N by 1 numpy.array)  
      step_sizes[n] is the step size used for the n^{th} step  
      s_mins[n] is the minimum singular value of DF at the n^{th} step  
      residuals[n] is the infinity-norm of F at the n^{th} step
    """

    # Set defaults
    N = W.shape[0]
    if va is None: va = np.zeros((N+1,1))
    if c is None:
        c = np.random.randn(N,1)
        c = c/np.sqrt((c**2).sum())

    # Constants
    I = np.eye(N)
    _W_, _Winv_ = np.eye(N+1), np.eye(N+1)
    _W_[:N,:N], _Winv_[:N,:N] = W, np.linalg.inv(W)

    # Termination criterion
    term = get_term(W, c)

    # Drive initial va to curve
    va = drive_initial_va(W, va, c, max_nr_iters, nr_tol)
    D = 1 - np.tanh(W.dot(va[:N,:]))**2
    J = np.concatenate((D*W - I, -c), axis=1)
    _,_,z = np.linalg.svd(J)
    z = z[[N],:].T

    # Traverse
    VA = []
    step_sizes = []
    s_mins = []
    residuals = []
    fxV = []
    status = "Success"
    for step in it.count(0):
        if step == max_traverse_steps:
            status = "Max steps reached"
            break
 
        # Track path
        VA.append(va)
        if step == 1:
            cloop = np.sqrt(((VA[1]-VA[0])**2).sum())

        # Update quantities
        D = 1 - np.tanh(W.dot(va[:N,:]))**2
        J = np.concatenate((D*W - I, -c), axis=1)

        z_new = calc_z_new(J, z)

        # Get step size
        step_size, rho, s_min = traverse_step_size(_W_, _Winv_, D, J, va, c, z_new)
        if max_step_size is not None: step_size = min(step_size, max_step_size)
        step_sizes.append(step_size)
        s_mins.append(s_min)

        va_new, F_new = take_traverse_step(W, I, c, va, z_new, step_size, max_nr_iters, nr_tol)
        residuals.append(np.fabs(F_new).max())

        # Check fixed point
        if not np.sign(va[N]) == np.sign(va_new[N]):
            # extra redundancy: seed local optimization with both endpoints and linear interpolant
            fxV.append(va[:N,:])
            fxV.append(va_new[:N,:])
            m = -va[N]/(va_new[N]-va[N]) # linear interpolant for alpha == 0
            fxV.append(va[:N,:] + m*(va_new[:N,:]-va[:N,:]))
        va = va_new
        z = z_new

        # Check termination
        if np.fabs(va[N]) > term:
            if logfile is not None:
                hardwrite(logfile,'Asymp: iteration %d of %s, step_size=%f, %d fx found, term: %e > %e\n'%(step,max_traverse_steps,step_size,len(fxV),np.fabs(va[N]),term.max()))
            status = "Success"
            break

        if max_fxpts is not None and len(fxV) >= max_fxpts:
            status = "Max fxpts found"
            break

        # Check for closed loop
        cloop_va = np.sqrt(((va-VA[0])**2).sum())
        if step > 5 and cloop_va < 1.5*cloop:
            if logfile is not None:
                hardwrite(logfile,'Cloop: iteration %d of %s, %d fx found, cloop: %e\n'%(step,max_traverse_steps,2*len(fxV)+1, cloop_va))
            status = "Closed loop detected"
            break

        if (step % 100) == 0 and logfile is not None:
            hardwrite(logfile,'iteration %d of %s,step_size=%f,s_min=%e,%d fx,term:%e>? %e,cloop:%e\n'%(step,max_traverse_steps,step_size,s_min,len(fxV),va[N],term.max(), np.sqrt(((va-VA[0])**2).sum())))

    # clean output
    if len(fxV) == 0:
        fxV = [np.zeros((N,1))]
    fxV = np.concatenate(fxV,axis=1)
    VA = np.concatenate(VA, axis=1)
    step_sizes = np.array(step_sizes)
    s_mins = np.array(s_mins)
    residuals = np.array(residuals)
    return status, fxV, VA, c, step_sizes, s_mins, residuals

def directional_fiber(W, va=None, c=None, max_nr_iters=2**8, nr_tol=2**-32, max_step_size=None, max_traverse_steps=None, max_fxpts=None, stop_time=None, logfile=None):
    """
    Generator version of traverse.
    Yields (unprocessed) fixed point candidates one by one, for use in a for loop.
    Finds candidates around all local |alpha| minima, not only sign changes.
    W is the weight matrix (N by N numpy.array)
    va is the initial point (N+1 by 1 numpy.array)
      if None, traversal starts at the origin
    c is the direction vector (N by 1 numpy.array)
      if None, a random direction vector is chosen
    max_nr_iters is the maximum iterations allowed for Newton-Raphson refinement
    nr_tol is the tolerance at which Newton-Raphson refinement can terminate
    max_step_size is a maximum step size to use for each step
      if None, no limit is imposed on the return value of traverse_step_size
    max_traverse_steps is the maximum number of steps allowed for traversal
      if None, traversal continues until another termination criteria is met
    max_fxpts is the number of fixed points at which traversal can terminate
      if None, traversal continues until another termination criteria is met
    stop_time is a clock time (compared with time.clock()) at which traversal is terminated
      if None, traversal continues until another termination criteria is met
    logfile is a file object open for writing that records progress
      if None, no progress is recorded

    yields status, fxv, VA, c, step_sizes, s_mins, residuals, where
      status is one of
        "Not done", "Success", "Max steps reached", "Max fxpts found", "Closed loop detected", "Timed out"
      fxv is the next fixed point candidate
      VA[:,n] is the n^{th} point along the fiber so far
      c is the direction vector that was used (N by 1 numpy.array)
      step_sizes[n] is the step size used for the n^{th} step so far
      s_mins[n] is the minimum singular value of DF at the n^{th} step so far
      residuals[n] is the infinity-norm of F at the n^{th} step so far
    """

    # Set defaults
    N = W.shape[0]
    if va is None: va = np.zeros((N+1,1))
    if c is None:
        c = np.random.randn(N,1)
        c = c/np.sqrt((c**2).sum())

    # Constants
    I = np.eye(N)
    _W_, _Winv_ = np.eye(N+1), np.eye(N+1)
    _W_[:N,:N], _Winv_[:N,:N] = W, np.linalg.inv(W)

    # Termination criterion
    term = get_term(W, c)

    # Drive initial va to curve
    va = drive_initial_va(W, va, c, max_nr_iters, nr_tol)
    D = 1 - np.tanh(W.dot(va[:N,:]))**2
    J = np.concatenate((D*W - I, -c), axis=1)
    _,_,z = np.linalg.svd(J)
    z = z[[N],:].T

    # Traverse
    VA = []
    step_sizes = []
    s_mins = []
    residuals = []
    num_fxpts = 0
    checking_cloop = False
    status = "Traversing"
    for step in it.count(0):

        # Save fiber
        VA.append(va)

        # Update quantities
        D = 1 - np.tanh(W.dot(va[:N,:]))**2
        J = np.concatenate((D*W - I, -c), axis=1)

        z_new = calc_z_new(J, z)

        # Get step size
        step_size, rho, s_min = traverse_step_size(_W_, _Winv_, D, J, va, c, z_new)
        if max_step_size is not None: step_size = min(step_size, max_step_size)
        step_sizes.append(step_size)
        s_mins.append(s_min)

        # Take step
        va_new, F_new = take_traverse_step(W, I, c, va, z_new, step_size, max_nr_iters, nr_tol)
        residuals.append(np.fabs(F_new).max())
        va = va_new
        z = z_new

        # Check local |alpha| minimum
        if len(VA) == 2 and np.fabs(VA[-2][N]) < np.fabs(VA[-1][N]):
            num_fxpts += 1
            yield status, VA[0][:N,:], VA, c, step_sizes, s_mins, residuals
        if len(VA) >= 3 and np.fabs(VA[-2][N]) < np.fabs(VA[-1][N]) and np.fabs(VA[-2][N]) < np.fabs(VA[-3][N]):
            num_fxpts += 3
            yield status, VA[-3][:N,:], VA, c, step_sizes, s_mins, residuals
            yield status, VA[-2][:N,:], VA, c, step_sizes, s_mins, residuals
            yield status, VA[-1][:N,:], VA, c, step_sizes, s_mins, residuals

        # Check for asymptote
        if np.fabs(va[N]) > term:
            if logfile is not None:
                hardwrite(logfile,'Asymp: iteration %d of %s, step_size=%f, %d fx found, term: %e > %e\n'%(step,max_traverse_steps,step_size,num_fxpts,np.fabs(va[N]),term))
            status = "Success"
            break
            
        # Check for closed loop
        cloop_distance = np.fabs(VA[-1]-VA[0]).max()
        if len(VA) > 3 and cloop_distance < np.fabs(VA[2]-VA[0]).max():
            if logfile is not None:
                hardwrite(logfile,'Cloop: iteration %d of %s, %d fx found, cloop: %e\n'%(step,max_traverse_steps,2*num_fxpts+1, cloop_distance))
            status = "Closed loop detected"
            break

        # Early termination criteria
        if step == max_traverse_steps:
            status = "Max steps reached"
            break
        if max_fxpts is not None and num_fxpts >= max_fxpts:
            status = "Max fxpts found"
            break
        if time.clock() > stop_time:
            status = "Timed out"
            break

        if (step % 100) == 0 and logfile is not None:
            hardwrite(logfile,'iteration %d of %s,step_size=%f,s_min=%e,%d fx,term:%e>? %e,cloop:%e\n'%(step,max_traverse_steps,step_size,s_min,num_fxpts,va[N],term, cloop_distance))

    # final output
    yield status, np.empty((N,0)), VA, c, step_sizes, s_mins, residuals

def get_test_points():
    """
    Construct a set of 300 test points with at most 3 "unique" members.
    returns a numpy.array V, where V[:,p] is the p^{th} point.
    """
    # make 100 copies of 3 distinct, random points
    V = np.tile(np.random.rand(10,3),(1,100))
    # shuffle randomly
    V = V[:,np.random.permutation(300)]
    # perterb by a small multiple of machine precision
    V = V + np.floor(5*np.random.rand(10,300))*eps(V)
    return V

def test_get_connected_components():
    """
    Sanity check for get_connected_components
    """
    V = get_test_points()
    components = get_connected_components(V)
    assert len(np.unique(components)) <= 3
    print('test get connected components passed!')

def test_get_unique_points():
    """
    Sanity check for get_unique_points
    """
    V = get_test_points()
    V_unique, _ = get_unique_points(V)
    assert V_unique.shape[1] <= 3
    for p in range(V.shape[1]):
        l0_diffs = (V_unique - V[:,[p]]).max(axis=0)
        assert l0_diffs.min() < (5*eps(V[:,p])).max()
    print('test get unique points passed!')

def test_fixed_within_eps():
    """
    Sanity check for fixed_within_eps
    """
    for rep in range(3):
        for N in range(1,100):
            # Form W,V fixed in infinite precision
            V = 2*np.random.rand(N,N) - 1
            W = mrdivide(np.arctanh(V), V)
            # Refine in finite precision with Newton-Raphson
            I = np.identity(N)
            for j in range(N):
                for k in range(100):
                    # Check termination
                    fixed, _ = fixed_within_eps(W, V[:,[j]])
                    if fixed[0]:
                        break
                    # Calculate residual
                    f = np.tanh(np.dot(W,V[:,[j]])) - V[:,[j]]
                    # Calculate Jacobian
                    J = (1-np.tanh(np.dot(W,V[:,[j]]))**2)*W - I
                    # Iterate point
                    V[:,[j]] = V[:,[j]] - mldivide(J,f)
            fixed, margin = fixed_within_eps(W, V)
            assert fixed.all()
    print('test fixed within eps passed!')
    # visualize forward error bound on last network
    f = np.tanh(W.dot(V)) - V
    fixed, margin = fixed_within_eps(W,V)
    f, margin = f.flatten(), margin.flatten()
    idx = np.argsort(margin)
    f, margin = f[idx], margin[idx]
    plt.plot(np.log2(f),'-')
    plt.plot(np.log2(margin),'--')
    plt.show()

def test_identical_fixed_points():
    """
    Sanity check for identical_fixed_points
    """
    for rep in range(3):
        for N in range(1,50):
            # Form V, W in infinite precision
            V = 2*np.random.rand(N,N)-1
            W = mrdivide(np.arctanh(V),V)
            # Test each V
            I = np.eye(N)
            for j in range(N):
                # Make P perturbations
                P = 10
                Vp = V[:,[j]] + (eps(V[:,[j]]) * np.random.randn(N,P))
                for p in range(P):
                    # Refine p^th perturbation with Newton-Raphson
                    K = 100
                    for k in range(K):
                        # Check early stop
                        fixed, _ = fixed_within_eps(W, Vp[:,[p]])
                        if fixed[0]: break
                        # Calculate current residual
                        f = np.tanh(np.dot(W,Vp[:,[p]])) - Vp[:,[p]]
                        # Calculate Jacobian
                        J = (1-np.tanh(np.dot(W,Vp[:,[p]]))**2)*W - I
                        # Iterate point
                        Vp[:,[p]] = Vp[:,[p]] - mldivide(J,f)
                # Assign best refinement to V
                p = np.fabs(np.tanh(np.dot(W,Vp))-Vp).max(axis=0).argmin()
                V[:,[j]] = Vp[:,[p]]
                # Check for identity with original
                identical, _, _ = identical_fixed_points(W, Vp, V[:,[j]])
                assert identical.all()
            # Check for non-identity with distincts
            for j in range(N):
                identical, _, _ = identical_fixed_points(W, V, V[:,[j]])
                assert identical[j]
                assert np.count_nonzero(identical)==1
    print('test identical fixed points passed!')

def run_tests():
    """
    Run sanity checks
    """
    #estimate_tanh_eps_error()
    test_get_connected_components()
    test_get_unique_points()
    test_fixed_within_eps()
    # test_identical_fixed_points()

def refine_fxpts(W, V, max_iters=2**5):
    """
    This is a helper function, consider refine_pts instead.
    Refines approximate fixed point locations with the Newton-Raphson method
    W should be the weight matrix (N by N numpy.array)
    V should be the approximate fixed points, where
      V[:,p] is the p^{th} point
    max_iters is the maximum number of iterations allowed for Newton-Raphson refinement
    returns V, converged, where
      V[:,p] is the p^{th} point after refinement
      converged[p] == True iff the p^{th} point is fixed_within_eps after refinement.
    """
    N = W.shape[0]
    I = np.eye(N)
    converged = np.zeros(V.shape[1], dtype=bool)
    for i in range(max_iters):
        V_i = V[:,~converged]
        tWV_i = np.tanh(W.dot(V_i))
        J = (1 - tWV_i**2).T[:,:,np.newaxis] * W[np.newaxis,:,:] - I[np.newaxis,:,:]
        V_i = V_i - solve(J, (tWV_i - V_i).T).T
        fixed, _ = fixed_within_eps(W, V_i)
        V[:,~converged] = V_i
        converged[~converged] = fixed
        if fixed.all(): break
    return V, converged

def refine_fxpts_capped(W, V, max_iters=2**5, cap=10000):
    """
    This is a helper function, consider refine_pts instead.
    Processes points in separate chunks to avoid prohibitive memory usage.
    W, V, and max_iters should be as in refine_fxpts
    cap is the maximum number of points to process at a time.
    returns V, converged as in refine_fxpts.
    """
    num_splits = int(np.ceil(1.0*V.shape[1]/cap))
    Vs = np.array_split(V, num_splits, axis=1)
    # refines = [refine_fxpts(W,V_,max_iters=max_iters) for V_ in Vs]
    refines = []
    for i in range(len(Vs)):
        # print('%d of %d'%(i,len(Vs)))
        refines.append(refine_fxpts(W,Vs[i],max_iters=max_iters))
    V = np.concatenate([r[0] for r in refines],axis=1)
    converged = np.concatenate([r[1] for r in refines])
    return V, converged

def refine_pts(W, V):
    """
    Refines approximate fixed point locations with the Newton-Raphson method.
    W should be the weight matrix (N by N numpy.array)
    V should be the approximate fixed points, where
      V[:,p] is the p^{th} point
    returns V, converged, where
      V[:,p] is the p^{th} point after refinement
      converged[p] == True iff the p^{th} point is fixed_within_eps after refinement.
    """
    return refine_fxpts_capped(W, V)

def baseline_solver_qg(v, W):
    """
    Computes the objective q and gradient g for the baseline solver (Sussillo and Barak 2013).
    v should be the current optimization point as a flat length N numpy.array
    W should be the weight matrix (N by N numpy.array)
    returns q, g, where
      q is the scalar value of the objective
      g is the gradient (a flat length-N numpy.array)
    """
    v = v.reshape((W.shape[0],1))
    tWv = np.tanh(W.dot(v))
    f = tWv - v
    J = (1-tWv**2)*W - np.eye(v.shape[0])
    return (f**2).sum(), J.T.dot(f).flatten() # q, grad
def baseline_solver_G(v, W):
    """
    Computes the approximate Hessian G for the baseline solver (Sussillo and Barak 2013).
    v should be the current optimization point as a flat length N numpy.array
    W should be the weight matrix (N by N numpy.array)
    returns G, the approximate Hessian (N by N numpy.array)
    """
    v = v.reshape((W.shape[0],1))
    tWv = np.tanh(W.dot(v))
    J = (1-tWv**2)*W - np.eye(W.shape[0])
    return J.T.dot(J)
def baseline_solver(W, timeout=60, max_fxpts=None, max_traj_steps=10, logfile=None):
    """
    A baseline fixed point solver (Sussillo and Barak 2013)
    Repeatedly samples and optimizes seeds along random trajectories until timing out.
    W should be the weight matrix (N by N numpy.array)
    timeout should be the number of seconds after which the solver is terminated.
    max_fxpts is the number of fixed points after which the solver is allowed to terminate.
      if None, solver continues until timeout.
    max_traj_steps is the maximum number of steps along a trajectory before optimization starts.
    logfile is a file object open for writing that records progress
      if None, no progress is recorded
    returns fxV, num_reps, where
      fxV[:,p] is the p^{th} (potentially non-fixed or duplicate) point found (a numpy.array)
      num_reps is the number of repetitions performed before timeout (i.e., fxV.shape[1])
    """
    N = W.shape[0]
    fxV = []
    neighbors = lambda X, y: identical_fixed_points(W, X, y)[0]
    start = time.clock()
    for num_reps in it.count(1):

        # get random initial seed anywhere in range
        v = 2*np.random.rand(W.shape[0],1) - 1

        # iterate trajectory a random number of steps
        num_traj_steps = np.random.randint(max_traj_steps)
        for step in range(num_traj_steps):
            v = np.tanh(W.dot(v))

        # run minimization
        res = spo.minimize(baseline_solver_qg, v.flatten(), args=(W,), method='trust-ncg', jac=True, hess=baseline_solver_G)
        fxv = res.x.reshape((W.shape[0],1))
        fxV.append(fxv)

        # check termination
        runtime = time.clock()-start
        if runtime > timeout:
            if logfile is not None:
                hardwrite(logfile,'term: %d reps %fs\n'%(num_reps,runtime))
            break

        if (num_reps % 10) == 0 and logfile is not None:
            hardwrite(logfile,'%d reps (%f of %fs)\n'%(num_reps, runtime, timeout))

    fxV = np.concatenate(fxV,axis=1)
    return fxV, num_reps

def post_process_fxpts(W, fxV, logfile=None, refine_cap=10000, Winv=None, neighbors=None):
    """
    Post-process a set of candidate fixed points:
      1. Refines the approximate point locations via Newton-Raphson
      2. Removes non-fixed points
      3. Adds the origin and the negatives of the remaining points
      4. Removes duplicates.
    W should be the weight matrix (N by N numpy.array)
    fxV[:,p] should be the p^{th} candidate fixed point
    logfile is a file object open for writing that records progress
      if None, no progress is recorded
    refine_cap is the maximum number of candidates refined at a time
    Winv should be the inverse of W, unless None, in which case it is computed
    neighbors should be a neighbor function as in identical_fixed_points
    returns fxV_unique, fxV, where
      fxV_unique[:,p] is the p^{th} refined, unique fixed point found
      fxV[:,q] is the q^{th} refined (potentially duplicate) fixed point found
    """
    if logfile is not None: hardwrite(logfile,'Refining fxpts...')
    if Winv is None: Winv = np.linalg.inv(W)
    fxV, converged = refine_fxpts_capped(W, fxV, cap=refine_cap)
    fxV = fxV[:,converged]
    N = W.shape[0]
    fxV = np.concatenate((-fxV, np.zeros((N,1)), fxV),axis=1)
    if logfile is not None: hardwrite(logfile,'Uniqueing fxpts...\n')
    if neighbors is None:
        neighbors = lambda X, y: identical_fixed_points(W, X, y, Winv)[0]
    fxV_unique = get_unique_points_recursively(fxV, neighbors=neighbors)
    return fxV_unique, fxV

def run_solver(W, c=None):
    """
    Convenience wrapper for the traverse algorithm with post-processing.
    W should be the weight matrix (N by N numpy.array)
    c should be the direction vector (N by 1 numpy.array)
      if None, c is chosen randomly
    returns fxpts, fiber, where
      fxpts[:,p] is the p^{th} fixed point found
      fiber[:,n] is the n^{th} point along the fiber encountered during traversal
    """
    # Run traverse
    _, fxpts, fiber, _, _, _, _ = traverse(W, c=c, max_traverse_steps = 2**20)
    # Post-process
    fxpts, _ = post_process_fxpts(W, fxpts)
    # Return output
    return fxpts, fiber

def show_fiber(W, fxpts, fiber, savefile=None):
    """
    Visualize fibers for 2-neuron networks
    W should be the weight matrix (2 by 2 numpy.array)
    fxpts, fiber should be as returned by run_solver(W)
    savefile should be a file object or name with which to save the plot .
    """
    if W.shape[0] != 2:
        print('show_fiber called with N == %d (!= 2)'%W.shape[0])
        return
    mpl.rcParams['mathtext.default'] = 'regular'
    fiber = np.concatenate((-fiber[:2,::-1],fiber[:2,:]), axis=1)
    fxpts, _ = post_process_fxpts(W, fxpts)
    plt.figure(figsize=(6,6))
    ptr.plot(plt.gca(), fiber[:2,:],'-k')
    ptr.scatter(plt.gca(), fxpts, 75, c=((0,0,0),))
    V = ptr.lattice([-1.25,-1.25],[1.25,1.25], 20)
    C = np.tanh(W.dot(V))-V
    ptr.quiver(plt.gca(),V,C,scale=.01,units='dots',width=1,color=((.7,.7,.7),), headwidth=7)
    V = fiber[:2,:]
    V = V[:,((-1.25 < V) & (V < 1.25)).all(axis=0)]
    C = np.tanh(W.dot(V))-V
    ptr.quiver(plt.gca(),V,C,scale=.005,units='dots',width=2,headwidth=5)
    plt.xlim([-1.25,1.25])
    plt.xlabel('$v_1$')
    plt.ylim([-1.25,1.25])
    plt.ylabel('$v_2$',rotation=0)
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
