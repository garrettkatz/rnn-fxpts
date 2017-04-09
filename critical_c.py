"""
Preliminary investigations of choosing c
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import rnn_fxpts as rfx
import plotter as pltr

def c_path_traversal(W, a0, signs,
        initial_dir_sign=1,
        max_iters=2**16,
        term_tol=2**-16,
        initial_delta=2**-4,
        max_adapt_steps=2**10,
        adapt_tol=2**-10,
        max_drive_steps=2**10,
        drive_tol=2**-32,
        max_fx_steps=2**8,
        arctanh_tol=0.00001,
        zerotanh_tol=0.00001):
    """
    Numerically traverse a set of critical c in 3d (for which the set is a curve)
    Inputs:
        W: The system weight matrix
        a0: Initial low rank null space
        signs: signs to use for a -> v
        max_iters: maximum number of iterations before termination
        term_tol: tolerance on asymptote for termination
        initial_delta: initial step size for Euler step
        max_adapt_steps: maximum number of adaptations to delta during Euler step
        adapt_tol: tolerance on curve residual after adaptive step
        max_drive_steps: maximum number of Newton-Raphson drive steps back to the curve
        drive_tol: tolerance on curve residual after driving back
        max_fx_steps: maximum number of Newton-Raphson steps for fixed point refinement
        arctanh_tol: keeps arctanh from going to inf
        zerotanh_tol: keeps tanh from going to zero (and dca going to inf)
    Outputs:
        C[n]: The n^{th} bad c found
        A[n]: The n^{th} point a along the curve
        DA[n][:,k-1]: The k^{th}-order tangent at the n^{th} point
        NR[n][j]: The j^{th} Newton-Raphson refinement after the n^{th} step
    """

    # Initialize output
    C_norm = []
    C = []
    V = []
    A = []
    DA = []
    NR = []

    # Initialize quantities
    N = W.shape[0]
    I = np.eye(N)
    a = a0 # column vector
    aW = W.T.dot(a) # flipped from row to column for convenience
    t = signs*np.sqrt(1 - a/aW) # tanh
    v = rfx.mldivide(W, np.arctanh(t))
    c = t - v
    c = c/np.sqrt((c*c).sum())
    F = np.concatenate((a.T.dot(a) - 1.0, -a.T.dot(W).dot(c)), axis=0)
    dta = -(aW*I - a*W.T)/(aW**2 * 2 * t)
    dva = rfx.mldivide(W, (1/(1-t**2))*dta)
    dca = dta - dva
    dFa = np.concatenate((2*a.T, -(a.T.dot(W).dot(dca) + W.dot(c).T)), axis=0)
    _, _, Z = np.linalg.svd(dFa) # for null-space
    da1 = Z[[-1],:].T/np.linalg.norm(Z[[-1],:]) # Current tangent vector (unit speed)
    da1 = da1 * initial_dir_sign
    da2 = np.zeros((N,1)) # Current acceleration vector (derivative of tangent)
    delta = initial_delta # Adaptive step size

    # Traverse
    for n in range(max_iters):

        # Take adaptive step
        for adapt_step in range(max_adapt_steps):
            if n == 0: break

            a_del = a + da1*delta + da2*delta**2/2.0
            aW_del = W.T.dot(a_del) # flipped from row to column for convenience
            if (((1-a_del/aW_del) < 0) | ((1-a_del/aW_del) > 1-arctanh_tol)).any():
                delta = delta / 2.0
                continue
            
            t_del = signs*np.sqrt(1 - a_del/aW_del) # tanh
            v_del = rfx.mldivide(W, np.arctanh(t_del))
            c_del = t_del - v_del
            c_del_norm = c_del/np.sqrt((c_del*c_del).sum())
            #F = np.concatenate((a_del.T, c_del.T), axis=0).dot(a_del) - np.array([[1.0],[0.0]])
            #F = np.concatenate((a_del.T, c_del_norm.T), axis=0).dot(a_del) - np.array([[1.0],[0.0]])
            F = np.concatenate((a_del.T.dot(a_del) - 1.0, -a_del.T.dot(W).dot(c_del)), axis=0)
            #print(' adapt steps %d, |F|=%f<?%f, delta=%f'%(adapt_step, np.fabs(F).max(), adapt_tol, delta))
            if (np.fabs(F) < adapt_tol).all():
                # Residual below tolerance, step is successful
                a = a_del
                #c = c_del
                c = c_del_norm
                if adapt_step==0:
                    # Immediate success, speed up
                    delta = delta * 2.0
                break
            else:
                pass
                #print(np.fabs(F).max())
            # Step unsuccessful, reduce size and repeat
            delta = delta / 2.0
        #print('%d of %d adapt steps, residual %f, delta %e'%(adapt_step, max_adapt_steps, np.fabs(F).max(), delta))
       
        # Drive va back to the curve with Newton-Raphson
        NR.append([])
        for drive_step in range(max_drive_steps):
            if (np.fabs(F) < drive_tol).all():
                # Residual is below tolerance, converged
                break
            # Method hasn't converged, take step
            aW = W.T.dot(a) # flipped from row to column for convenience
            if (((1-a/aW) < 0) | ((1-a/aW) > 1-arctanh_tol)).any():
                break
            t = signs*np.sqrt(1 - a/aW) # tanh
            if (np.fabs(t) < zerotanh_tol).any():
                break
            v = rfx.mldivide(W, np.arctanh(t))
            c = t - v
            c_norm = c/np.sqrt((c*c).sum())
            dta = -(aW*I - a*W.T)/(aW**2 * 2 * t)
            dva = rfx.mldivide(W, (1/(1-t**2))*dta)
            dca = dta - dva
            dFa = np.concatenate((2*a.T, -(a.T.dot(W).dot(dca) + W.dot(c).T)), axis=0)
            a = a - rfx.mldivide(dFa, F)
            F = np.concatenate((a.T.dot(a) - 1.0, -a.T.dot(W).dot(c)), axis=0)
            NR[n].append(a)
        if len(NR[n]) > 0:
            NR[n] = np.concatenate(tuple(NR[n]), axis=1)
        else:
            NR[n] = np.empty((N,0))
        if (((1-a/aW) < 0) | ((1-a/aW) > 1-arctanh_tol)).any():
            break

        # Update differentials
        aW = W.T.dot(a) # flipped from row to column for convenience
        if (((1-a/aW) < 0) | ((1-a/aW) > 1-arctanh_tol)).any():
            break
        t = signs*np.sqrt(1 - a/aW) # tanh
        if (np.fabs(t) < zerotanh_tol).any():
            break
        v = rfx.mldivide(W, np.arctanh(t))
        c = t - v
        c_norm = c/np.sqrt((c*c).sum())
        dta = -(aW*I - a*W.T)/(aW**2 * 2 * t)
        dva = rfx.mldivide(W, (1/(1-t**2))*dta)
        dca = dta - dva
        dFa = np.concatenate((2*a.T, -(a.T.dot(W).dot(dca) + W.dot(c).T)), axis=0)
        j = da1 - np.dot(rfx.mrdivide(da1.T, dFa), dFa).T # Fast null-space
        da1 = np.dot(j, np.dot(da1.T, j)) # undo any change in direction
        da1 = da1 / np.linalg.norm(da1) # unit speed
        da2 = np.zeros((N,1)) # Current acceleration vector (derivative of tangent)

        # Save current point and derivatives
        C_norm.append(c_norm)
        C.append(c)
        V.append(v)
        A.append(a)
        DA.append(np.concatenate((da1,da2),axis=1))

        # Check termination criteria
        if True:
            term = 0

        if (n % 100) == 0:
            print('iteration %d of %d, term: %e'%(n,max_iters,term))
    # Post-process output
    #print(len(C), C[0].shape, C[1].shape)
    if len(C) > 0:
        C_norm = np.concatenate(C_norm, axis=1)
        C = np.concatenate(C, axis=1)
        V = np.concatenate(V, axis=1)
        A = np.concatenate(A, axis=1)
    else:
        C_norm = np.empty((N,0))
        C = np.empty((N,0))
        V = np.empty((N,0))
        A = np.empty((N,0))
    return C_norm, C, V, A, DA, NR

def find_low_rank_c_3d(W):
    """
    Compute full set of low-rank c in 3d (in which it is a union of curves)
    Uses numerical traversals, seeded with brute force grid sampling
    """
    N = 3
    if W.shape[0] != N: return
    samp = 100

    # Sample A lattice: a = A[n,:]
    A = np.mgrid[-1:1:(samp*1j), -1:1:(samp*1j)]
    A = np.array([A[0].flatten(), A[1].flatten()])
    A = A[:, np.fabs(A[0,:])+np.fabs(A[1,:]) <= 1]
    A = np.concatenate((A, 1-np.fabs(A).sum(axis=0)[np.newaxis,:]), axis=0)
    A = np.concatenate((A, np.array([[1],[1],[-1]])*A), axis=1)
    A = A.T

    # feasible A:
    # deriv tWv: dtWv = A[m,j]/(A[m,:]W[:,j])
    # feas[n] iff ds in (0,1] for all j
    dtWv = A/A.dot(W)
    feas = ((0 < dtWv) & (dtWv <= 1)).all(axis=1)
    if not feas.any():
        print('no feasible low rank solutions!')
        return
    A = A[feas,:]    
    dtWv = dtWv[feas,:]

    # Get sign possibilities for d sigma
    signs = (np.arange(2**N) / (2**np.arange(N)[:,np.newaxis])) % 2
    signs = 2*signs-1
    
    # get a, v, c for each sign possibility
    # c[s]: N x M c's for sign possibility s
    # tWv[j] = +/- (1 - a[j]/(a.T*W[:,j]))**0.5
    # v = W\arctanh(tWv)
    # c = tWv-v
    neighbors = np.fabs(A[:,np.newaxis,:]-A[np.newaxis,:,:]).max(axis=2) < 1.5*(2.0/(samp-1.0))
    a, v, c, c_path, c_path_inf, v_path, a_path = [], [], [], [], [], [] ,[]
    for s in range(signs.shape[1]):
        tWv = (signs[:,s]*(1-dtWv)**0.5).T
        v_s = rfx.mldivide(W, np.arctanh(tWv))
        c_s = tWv - v_s
        #c_s /= np.fabs(c_s).sum(axis=0)
        c_s /= np.sqrt((c_s*c_s).sum(axis=0))
        # find c where -A[n,:]/D*c ~ 0 (sign change)
        ADC = (A/dtWv*c_s.T).sum(axis=1) # one row of A, dtWv, c_s.T per feasible a
        signchange = np.sign(ADC[:,np.newaxis]*ADC[np.newaxis,:]) <= 0
        neighborchange = (neighbors & signchange).any(axis=1)
        a.append(A.T[:,neighborchange])
        v.append(v_s[:,neighborchange])
        c.append(c_s[:,neighborchange])
        # do numerical traversal
        c_path_s = []
        v_path_s = []
        a_path_s = []
        c_path_s_inf = []
        a_s_dec = a[-1]
        c_s_dec = c[-1]
        max_unfound_steps = 3
        unfound_steps = 0
        while a_s_dec.shape[1] > 0 and unfound_steps < max_unfound_steps:
            idx = np.random.randint(a_s_dec.shape[1])
            c_path_s_p, c_path_s_p_inf, v_path_s_p, a_path_s_p, _, _ = c_path_traversal(W, a_s_dec[:,[idx]], signs[:,[s]], max_iters=1000, max_drive_steps=10)
            c_path_s_n, c_path_s_n_inf, v_path_s_n, a_path_s_n, _, _ = c_path_traversal(W, a_s_dec[:,[idx]], signs[:,[s]], initial_dir_sign=-1, max_iters=1000, max_drive_steps=10)
            c_path_s.append(np.concatenate((c_path_s_p[:,::-1], c_path_s_n), axis=1))
            c_path_s_inf.append(np.concatenate((c_path_s_p_inf[:,::-1], c_path_s_n_inf), axis=1))
            v_path_s.append(np.concatenate((v_path_s_p[:,::-1], v_path_s_n), axis=1))
            a_path_s.append(np.concatenate((a_path_s_p[:,::-1], a_path_s_n), axis=1))
            if c_path_s[-1].shape[1] == 0:
                print('no c_path_s from idx')
            else:
                unfound = (np.fabs((c_s_dec[:,:,np.newaxis]*c_path_s[-1][:,np.newaxis,:]).sum(axis=0)).max(axis=1) < 0.95)
                print('s=%d,%d unfound'%(s,np.count_nonzero(unfound)))
                c_s_dec = c_s_dec[:,unfound]
                a_s_dec = a_s_dec[:,unfound]
            unfound_steps += 1
        print('s=%d,unfound done.'%s)
        c_path.append(c_path_s)
        c_path_inf.append(c_path_s_inf)
        v_path.append(v_path_s)
        a_path.append(a_path_s)

    # plot
    ax = plt.gca(projection='3d')
    for colorful in range(0):
        colors = ['b','g','r','c','m','y','k','w']
        # numerical
        for s in range(3):#len(c)):
            # if colors[s] not in ['b','m','y']: continue
            # if s not in [0,4,5]: continue
            for p in range(len(c_path[s])):
                pltr.plot(ax, c_path[s][p], colors[s])
                #pltr.quiver(ax, a_path[s][p], c_path[s][p])
                # pltr.plot(ax, c_path[s][p] + 0.05*a_path[s][p], 'k--')
                pltr.plot(ax, -c_path[s][p], colors[s])
                # pltr.plot(ax, -c_path[s][p] - 0.05*a_path[s][p], 'k--')
                # pltr.plot(ax, c_path_inf[s][p], colors[s])
                # pltr.plot(ax, -c_path_inf[s][p], colors[s])
            # pltr.plot(ax, c[s], colors[s]+'+')
        # Wc = 0
        angles = np.linspace(0,2*np.pi,100)
        for i in range(N):
            _, _, Z = np.linalg.svd(W[[i],:]) # for null-space
            c_i = Z[-2:,:].T.dot(np.array([np.sin(angles), np.cos(angles)]))
            pltr.plot(ax, c_i, 'k--')
    for summary in range(1):
        # numerical
        for s in range(len(c)):
            if s not in [0,4,5]: continue
            for p in range(len(c_path[s])):
                pltr.plot(ax, c_path[s][p], 'k')
                pltr.plot(ax, -c_path[s][p], 'k')
                # if c_path[s][p].shape[1] > 0:
                #     print(c_path[s][p].shape)
                #     print(int(c_path[s][p].shape[1]/2))
                #     print(c_path[s][p][:,[int(c_path[s][p].shape[1]/2)]])
                #     print(['%d,%d'%(s,p)])
                #     pltr.text(ax, c_path[s][p][:,[int(c_path[s][p].shape[1]/2)]],['+%d,%d'%(s,p)])
                #     # pltr.text(ax, -c_path[s][p][:,[int(c_path[s][p].shape[1]/2)]],['-%d,%d'%(s,p)])
        # Wc = 0
        angles = np.linspace(0,2*np.pi,100)
        for i in range(N):
            _, _, Z = np.linalg.svd(W[[i],:]) # for null-space
            c_i = Z[-2:,:].T.dot(np.array([np.sin(angles), np.cos(angles)]))
            pltr.plot(ax, c_i, 'k--')

    pltr.set_lims(ax, 2*np.array([[1],[1],[1]])*np.array([[-1,1]]))
    # ax.set_xlabel('$c_1$')
    # ax.set_ylabel('$c_2$')
    # ax.set_zlabel('$c_3$')
    #plt.show()
    return a, W, v, c, c_path, v_path, a_path

def find_low_rank_c_3d_data(W):
    """
    Compute full set of low-rank c in 3d (in which it is a union of curves)
    Uses numerical traversals, seeded with brute force grid sampling
    """
    N = 3
    if W.shape[0] != N: return
    samp = 100

    # Sample A lattice: a = A[n,:]
    A = np.mgrid[-1:1:(samp*1j), -1:1:(samp*1j)]
    A = np.array([A[0].flatten(), A[1].flatten()])
    A = A[:, np.fabs(A[0,:])+np.fabs(A[1,:]) <= 1]
    A = np.concatenate((A, 1-np.fabs(A).sum(axis=0)[np.newaxis,:]), axis=0)
    A = np.concatenate((A, np.array([[1],[1],[-1]])*A), axis=1)
    A = A.T

    # feasible A:
    # deriv tWv: dtWv = A[m,j]/(A[m,:]W[:,j])
    # feas[n] iff ds in (0,1] for all j
    dtWv = A/A.dot(W)
    feas = ((0 < dtWv) & (dtWv <= 1)).all(axis=1)
    if not feas.any():
        print('no feasible low rank solutions!')
        return
    A = A[feas,:]    
    dtWv = dtWv[feas,:]

    # Get sign possibilities for d sigma
    signs = (np.arange(2**N) / (2**np.arange(N)[:,np.newaxis])) % 2
    signs = 2*signs-1
    
    # get a, v, c for each sign possibility
    # c[s]: N x M c's for sign possibility s
    # tWv[j] = +/- (1 - a[j]/(a.T*W[:,j]))**0.5
    # v = W\arctanh(tWv)
    # c = tWv-v
    neighbors = np.fabs(A[:,np.newaxis,:]-A[np.newaxis,:,:]).max(axis=2) < 1.5*(2.0/(samp-1.0))
    a, v, c, c_path, c_path_inf, v_path, a_path = [], [], [], [], [], [] ,[]
    npz = {}
    for s in range(signs.shape[1]):
        tWv = (signs[:,s]*(1-dtWv)**0.5).T
        v_s = rfx.mldivide(W, np.arctanh(tWv))
        c_s = tWv - v_s
        #c_s /= np.fabs(c_s).sum(axis=0)
        c_s /= np.sqrt((c_s*c_s).sum(axis=0))
        # find c where -A[n,:]/D*c ~ 0 (sign change)
        ADC = (A/dtWv*c_s.T).sum(axis=1) # one row of A, dtWv, c_s.T per feasible a
        signchange = np.sign(ADC[:,np.newaxis]*ADC[np.newaxis,:]) <= 0
        neighborchange = (neighbors & signchange).any(axis=1)
        a.append(A.T[:,neighborchange])
        v.append(v_s[:,neighborchange])
        c.append(c_s[:,neighborchange])
        npz['a_%d'%s] = a[-1]
        npz['v_%d'%s] = v[-1]
        npz['c_%d'%s] = c[-1]
        # do numerical traversal
        c_path_s = []
        v_path_s = []
        a_path_s = []
        c_path_s_inf = []
        a_s_dec = a[-1]
        c_s_dec = c[-1]
        max_unfound_steps = 3
        unfound_steps = 0        
        while a_s_dec.shape[1] > 0 and unfound_steps < max_unfound_steps:
            idx = np.random.randint(a_s_dec.shape[1])
            c_path_s_p, c_path_s_p_inf, v_path_s_p, a_path_s_p, _, _ = c_path_traversal(W, a_s_dec[:,[idx]], signs[:,[s]], max_iters=1000, max_drive_steps=10)
            c_path_s_n, c_path_s_n_inf, v_path_s_n, a_path_s_n, _, _ = c_path_traversal(W, a_s_dec[:,[idx]], signs[:,[s]], initial_dir_sign=-1, max_iters=1000, max_drive_steps=10)
            c_path_s.append(np.concatenate((c_path_s_p[:,::-1], c_path_s_n), axis=1))
            c_path_s_inf.append(np.concatenate((c_path_s_p_inf[:,::-1], c_path_s_n_inf), axis=1))
            v_path_s.append(np.concatenate((v_path_s_p[:,::-1], v_path_s_n), axis=1))
            a_path_s.append(np.concatenate((a_path_s_p[:,::-1], a_path_s_n), axis=1))
            npz['c_path_%d_%d'%(s,len(c_path_s)-1)] = c_path_s[-1]
            npz['v_path_%d_%d'%(s,len(v_path_s)-1)] = v_path_s[-1]
            npz['a_path_%d_%d'%(s,len(a_path_s)-1)] = a_path_s[-1]
            if c_path_s[-1].shape[1] == 0:
                print('no c_path_s from idx')
            else:
                unfound = (np.fabs((c_s_dec[:,:,np.newaxis]*c_path_s[-1][:,np.newaxis,:]).sum(axis=0)).max(axis=1) < 0.95)
                print('s=%d,%d unfound'%(s,np.count_nonzero(unfound)))
                c_s_dec = c_s_dec[:,unfound]
                a_s_dec = a_s_dec[:,unfound]
            unfound_steps += 1
        print('s=%d,unfound done.'%s)
        c_path.append(c_path_s)
        c_path_inf.append(c_path_s_inf)
        v_path.append(v_path_s)
        a_path.append(a_path_s)
    npz['lens'] = np.array([len(c_path_s) for c_path_s in c_path])
    np.savez('lork_c_3d_data.npz', W=W, signs=signs, **npz)
    return a, W, v, c, c_path, v_path, a_path

def c_space_fig_old():
    """
    Plot both the set of bad c and some nearby fibers
    """
    mpl.rcParams['mathtext.default'] = 'regular'
    # mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'font.size': 14})

    N = 3
    num_C = 8 # should be even or else land exactly on W[i,:]c=0

    # known loop:
    W = np.array([[ 1.20005258, -0.09232104, -0.12745533],
              [-0.03161879,  1.26916341, -0.11257691],
              [-0.03246188,  0.08548894,  1.24386724]])
    # c1 = np.array([[.7,.5,2]]).T
    #c2 = np.array([[.1,1.5,2]]).T
    c1 = np.array([[.3,.7,.7]]).T
    c2 = np.array([[.7,.5,.7]]).T

    # # Random W, C
    # W = 1.1*np.eye(N) + 0.1*np.random.randn(N,N)
    # c1 = mldivide(W, np.array([[+0.5,0.3,1]]).T)
    # c2 = mldivide(W, np.array([[-0.5,0.3,1]]).T)
    interp = np.arange(num_C)/(num_C-1.0)
    C = c1*interp[np.newaxis,:] + c2*(1-interp[np.newaxis,:])
    C = C/np.sqrt((C*C).sum(axis=0))
    bright_cap = .7

    print('low rank c...')
    # show c space partition
    fig = plt.figure(figsize=(11,6))
    ax = fig.add_subplot(1,2,1,projection='3d')
    _ = find_low_rank_c_3d(W)
    # ax.set_title('(a)')
    mpl.rcParams['mathtext.default'] = 'it'
    # mpl.rcParams['font.family'] = 'serif'
    ax.set_title('$c/||c||\in\mathrm{\mathbb{S}}^2$')

    pltr.scatter(ax, C, c=np.array([[bright_cap*g for _ in range(3)] for g in interp]),edgecolor='face')
    # pltr.scatter(ax, C, s=20, c=[str(bright_cap*g) for g in interp],edgecolor='face')
    # pltr.quiver(ax, c1, c1-c2)

    # ax.azim, ax.elev = 16, 51
    ax.azim, ax.elev = -74, 2

    # plt.show()
    # return None

    # show paths
    ax = fig.add_subplot(1,2,2,projection='3d')
    for p in range(2):
        print('fxpath brute %d...'%p)
        fxV, VA = rfx.brute_fiber(W, C[:,[p*-1]])

        for i in range(len(VA)):
            for s in [1]:#[-1,1]:
                pltr.plot(ax, s*VA[i][:N, (np.fabs(VA[i][:N,:]) < 3).all(axis=0)], color=str(bright_cap*interp[p*-1]))
                pltr.plot(ax, s*fxV[i], 'ko')
        # ax.set_title('c_%d'%p)

    pltr.set_lims(ax, 3*np.array([[1,1,1]]).T*np.array([[-1,1]]))
    ax.azim, ax.elev = -94, 8
    # ax.set_xlabel('$v_1$') #,rotation=0)
    # ax.set_ylabel('$v_2$') #,rotation=0)
    # ax.set_zlabel('$v_3$') #,rotation=0)
    ax.set_title('$v\in\mathrm{\mathbb{R}}^3$')

    plt.tight_layout()
    plt.show()
    # ax.xaxis.label.set_rotation(0)
    # ax.yaxis.label.set_rotation(0)
    # ax.zaxis.label.set_rotation(0)

    return fxV, VA, W, C

def c_space_fig_fiber_data():
    """
    Plot both the set of bad c and some nearby fibers
    """

    N = 3
    num_C = 8 # should be even or else land exactly on W[i,:]c=0

    # known loop:
    W = np.array([[ 1.20005258, -0.09232104, -0.12745533],
              [-0.03161879,  1.26916341, -0.11257691],
              [-0.03246188,  0.08548894,  1.24386724]])
    # c1 = np.array([[.7,.5,2]]).T
    #c2 = np.array([[.1,1.5,2]]).T
    c1 = np.array([[.3,.7,.7]]).T
    c2 = np.array([[.7,.5,.7]]).T

    # # Random W, C
    # W = 1.1*np.eye(N) + 0.1*np.random.randn(N,N)
    # c1 = mldivide(W, np.array([[+0.5,0.3,1]]).T)
    # c2 = mldivide(W, np.array([[-0.5,0.3,1]]).T)
    interp = np.arange(num_C)/(num_C-1.0)
    C = c1*interp[np.newaxis,:] + c2*(1-interp[np.newaxis,:])
    C = C/np.sqrt((C*C).sum(axis=0))
    bright_cap = .7

    # show paths
    # for p in range(2):
    npz = {}
    for p in range(len(interp)):
        print('fxpath brute %d...'%p)
        fxV, VA = rfx.brute_fiber(W, C[:,[p]])
        # fxpts[i][:,j] is the j^{th} fixed point found, and
        # fiber[i][:,n] is the n^{th} point along traversal,
        for i in range(len(fxV)):
            npz['fxV_%d_%d'%(p,i)] = fxV[i]
            npz['VA_%d_%d'%(p,i)] = VA[i]
        npz['lens_%d'%p] = len(fxV)
    np.savez('brute_fiber_data.npz',**npz)
    return fxV, VA, W, C
    
def c_space_fig():
    """
    Plot both the set of bad c and some nearby fibers
    """
    mpl.rcParams['mathtext.default'] = 'regular'
    # mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'font.size': 14})

    N = 3
    num_C = 8 # should be even or else land exactly on W[i,:]c=0

    # known loop:
    W = np.array([[ 1.20005258, -0.09232104, -0.12745533],
              [-0.03161879,  1.26916341, -0.11257691],
              [-0.03246188,  0.08548894,  1.24386724]])
    # c1 = np.array([[.7,.5,2]]).T
    #c2 = np.array([[.1,1.5,2]]).T
    c1 = np.array([[.3,.7,.7]]).T
    c2 = np.array([[.7,.5,.7]]).T

    # # Random W, C
    # W = 1.1*np.eye(N) + 0.1*np.random.randn(N,N)
    # c1 = mldivide(W, np.array([[+0.5,0.3,1]]).T)
    # c2 = mldivide(W, np.array([[-0.5,0.3,1]]).T)
    interp = np.arange(num_C)/(num_C-1.0)
    C = c1*interp[np.newaxis,:] + c2*(1-interp[np.newaxis,:])
    C = C/np.sqrt((C*C).sum(axis=0))
    bright_cap = .7

    # show c space partition
    if not os.path.exists('lork_c_3d_data.npz'):
        _ = find_low_rank_c_3d_data(W)
    npz = np.load('lork_c_3d_data.npz')
    
    fig = plt.figure(figsize=(4.5,9))
    ax = fig.add_subplot(2,1,1,projection='3d')
    for colorful in range(0):
        colors = ['b','g','r','c','m','y','k','w']
        # numerical
        for s in npz['signs'].shape[1]:
            # if colors[s] not in ['b','m','y']: continue
            # if s not in [0,4,5]: continue
            # for p in range(len(c_path[s])):
            for p in range(npz['lens'][s]):
                c_path_s_p = npz['c_path_%d_%d'%(s,p)]
                pltr.plot(ax, c_path_s_p, colors[s])
                #pltr.quiver(ax, a_path[s][p], c_path_s_p)
                # pltr.plot(ax, c_path_s_p + 0.05*a_path[s][p], 'k--')
                pltr.plot(ax, -c_path_s_p, colors[s])
                # pltr.plot(ax, -c_path_s_p - 0.05*a_path[s][p], 'k--')
                # pltr.plot(ax, c_path_inf[s][p], colors[s])
                # pltr.plot(ax, -c_path_inf[s][p], colors[s])
            # pltr.plot(ax, c[s], colors[s]+'+')
        # Wc = 0
        angles = np.linspace(0,2*np.pi,100)
        for i in range(N):
            _, _, Z = np.linalg.svd(W[[i],:]) # for null-space
            c_i = Z[-2:,:].T.dot(np.array([np.sin(angles), np.cos(angles)]))
            pltr.plot(ax, c_i, 'k--')
    for summary in range(1):
        # numerical
        for s in range(npz['signs'].shape[1]):
            if s not in [0,4,5]: continue
            for p in range(npz['lens'][s]):
                c_path_s_p = npz['c_path_%d_%d'%(s,p)]
                pltr.plot(ax, c_path_s_p, 'k')
                pltr.plot(ax, -c_path_s_p, 'k')
                # if c_path[s][p].shape[1] > 0:
                #     print(c_path[s][p].shape)
                #     print(int(c_path[s][p].shape[1]/2))
                #     print(c_path[s][p][:,[int(c_path[s][p].shape[1]/2)]])
                #     print(['%d,%d'%(s,p)])
                #     pltr.text(ax, c_path[s][p][:,[int(c_path[s][p].shape[1]/2)]],['+%d,%d'%(s,p)])
                #     # pltr.text(ax, -c_path[s][p][:,[int(c_path[s][p].shape[1]/2)]],['-%d,%d'%(s,p)])
        # Wc = 0
        angles = np.linspace(0,2*np.pi,100)
        for i in range(N):
            _, _, Z = np.linalg.svd(W[[i],:]) # for null-space
            c_i = Z[-2:,:].T.dot(np.array([np.sin(angles), np.cos(angles)]))
            pltr.plot(ax, c_i, 'k--')

    # plt.xticks([-1,0,1])
    plt.yticks([-1,0,1])
    # ax.set_zticks([-1,0,1])

    pltr.set_lims(ax, 1*np.array([[1],[1],[1]])*np.array([[-1,1]]))
    # ax.set_xlabel('$c_1$')
    # ax.set_ylabel('$c_2$')
    # ax.set_zlabel('$c_3$')

    # ax.set_title('(a)')
    mpl.rcParams['mathtext.default'] = 'it'
    # mpl.rcParams['font.family'] = 'serif'
    # ax.set_title('$\frac{c}{||c||}\in\mathrm{\mathbb{S}}^2$')
    ax.set_title('$c/||c||\in\mathrm{\mathbb{S}}^2$')

    pltr.scatter(ax, C, c=np.array([[bright_cap*g for _ in range(3)] for g in interp]),edgecolor='face')
    # pltr.scatter(ax, C, s=20, c=[str(bright_cap*g) for g in interp],edgecolor='face')
    # pltr.quiver(ax, c1, c1-c2)

    # ax.azim, ax.elev = 16, 51
    ax.azim, ax.elev = -74, 2

    # plt.show()
    # return None

    # show paths
    if not os.path.exists('brute_fiber_data.npz'):
        _ = c_space_fig_fiber_data()
    npz = np.load('brute_fiber_data.npz')
    #         npz['fxV_%d_%d'%(p,i)] = fxV[i]
    #         npz['VA_%d_%d'%(p,i)] = VA[i]
    #     npz['lens_%d'%p] = len(fxV)
    # np.savez('brute_fiber_data.npz',**npz)

    ax = fig.add_subplot(2,1,2,projection='3d')
    for p in range(0,len(interp),2):
        for i in range(npz['lens_%d'%p]):
            fxV_i = npz['fxV_%d_%d'%(p,i)]
            VA_i = npz['VA_%d_%d'%(p,i)]
            for s in [1]:#[-1,1]:
                col = np.array([bright_cap*interp[p] for _ in range(3)])
                pltr.plot(ax, s*VA_i[:N, (np.fabs(VA_i[:N,:]) < 3).all(axis=0)], color=col)
                pltr.plot(ax, s*fxV_i, 'ko')
        # ax.set_title('c_%d'%p)

    pltr.set_lims(ax, 1.5*np.array([[1,1,1]]).T*np.array([[-1,1]]))
    ax.azim, ax.elev = -94, 8
    # ax.set_xlabel('$v_1$') #,rotation=0)
    # ax.set_ylabel('$v_2$') #,rotation=0)
    # ax.set_zlabel('$v_3$') #,rotation=0)
    
    plt.xticks([-1,0,1])
    plt.yticks([-1,0,1])
    ax.set_zticks([-1,0,1])
    ax.set_title('$v\in\mathrm{\mathbb{R}}^3$')

    plt.tight_layout()
    plt.show()
    # ax.xaxis.label.set_rotation(0)
    # ax.yaxis.label.set_rotation(0)
    # ax.zaxis.label.set_rotation(0)

    # return fxV, VA, W, C
    
def loop_humps():
    W = 1.1*np.eye(3)+0.1*np.random.randn(3,3)
    a, W, v, c, c_path, v_path, a_path = find_low_rank_c_3d(W)
    while True:
        s = np.random.randint(len(c_path))
        if len(c_path[s]) == 0: continue
        p = np.random.randint(len(c_path[s]))
        if c_path[s][p].shape[1] == 0: continue
        if (np.fabs(v_path[s][p]) < 1.0).all(axis=0).any():
            small_v = np.flatnonzero((np.fabs(v_path[s][p]) < 1.0).all(axis=0))
            idx = small_v[np.random.randint(len(small_v))]
        else:
            idx = np.argmin(np.fabs(v_path[s][p]).max(axis=0))
        print('index %d'%idx)
        break
    c_sample = c_path[s][p][:,[idx]]
    v_sample = v_path[s][p][:,[idx]]
    a_sample = a_path[s][p][:,[idx]]
    t = np.tanh(W.dot(v_sample))
    D = np.diag(1.0/(1.0-t**2).flatten())
    print(a_sample.T.dot(np.concatenate((W - D, D.dot(c_sample)), axis=1))) 
    c_ax = plt.gca(projection='3d')
    pltr.plot(c_ax, c_sample, 'ro')
    # humps
    N = W.shape[0]
    signs = (np.arange(2**N) / (2**np.arange(N)[:,np.newaxis])) % 2
    signs = 2*signs-1
    V_humps = mldivide(W, np.arctanh(np.sqrt(1.0/3.0))*signs)
    # show
    nsp = 2
    v_fig = plt.figure()
    v_ax = v_fig.add_subplot(nsp, nsp, 1, projection='3d')
    rfx.brute_fiber(W, c_sample, ax=v_ax)
    # pltr.plot(plt.gca(projection='3d'), v_path[s][p][:,[idx]], 'r+')
    pltr.plot(v_ax, V_humps, 'bo')
    for s in range(len(v_path)):
        for p in range(len(v_path[s])):
            pltr.plot(v_ax, v_path[s][p], 'r')
    for sp in range(2,nsp**2+1):
        print('brute...%d'%sp)
        c_sample = np.random.randn(3,1)
        c_sample = c_sample/np.sqrt((c_sample**2).sum())
        v_ax = v_fig.add_subplot(nsp, nsp, sp, projection='3d')
        rfx.brute_fiber(W, c_sample, ax=v_ax)
        pltr.plot(v_ax, V_humps, 'bo')
        v_ax.set_title(str(sp))
        pltr.plot(c_ax, c_sample, 'r+')
        c_ax.text(c_sample[0,0], c_sample[1,0], c_sample[2,0], str(sp))
    plt.show()
