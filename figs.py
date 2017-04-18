"""
Render the figures shown in (Katz and Reggia 2017)
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import rnn_fxpts as rfx
import fxpt_experiments as fe
import critical_c as cc
import roundoff as ro
import plotter as ptr

def nullclines(W):
    """
    Helper for plotting the nullclines of 2d weight matrix
    """
    v = np.linspace(-1,1,100)[1:-1]
    v0 = (np.arctanh(v) - W[1,1]*v)/W[1,0]
    v1 = (np.arctanh(v) - W[0,0]*v)/W[0,1]
    plt.plot(v,v1,'-k')
    plt.plot(v0,v,'--k')
    plt.xlim([-1,1])
    plt.xlabel('$v_1$')
    plt.ylim([-1,1])
    plt.ylabel('$v_2$',rotation=0)

def orbit(W, v0, num_steps):
    """
    Helper for plotting a trajectory of a 2-neuron network
    """
    v = [v0]
    for step in range(num_steps):
        v.append(np.tanh(W.dot(v[-1])))
    v = np.concatenate(v[::4],axis=1)
    c = [(x,x,x) for x in (np.linspace(.8,0,v.shape[1]))**1]
    plt.scatter(v[0,:],v[1,:],75,c=c,marker='o',edgecolors='none')

def pert_fig():
    """
    Show that slight perturbations to network weight can change fixed points into attractor waypoints
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'font.size': 20})
    plt.figure(figsize=(11,6))
    plt.subplot(1,2,1)
    W = np.array([[1.1, 0.035],[-0.035, 1.1]])
    orbit(W, np.array([[0],[1]]), 50)
    nullclines(W)
    plt.subplot(1,2,2)
    W = np.array([[1.1, 0.04],[-0.04, 1.1]])
    orbit(W, np.array([[0],[1]]), 400)
    nullclines(W)
    plt.show()

def fiber_fig():
    """
    Show an example of a directional fiber for a two-neuron network
    """
    # mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams['mathtext.default'] = 'it'
    mpl.rcParams.update({'font.size': 18})
    W = np.array([[1.5, 0.05],[-0.01, 1.5]])
    c = np.array([[1.],[2]])
    _, fxV, VA, _, _, _, _ = rfx.traverse(W, c=c)
    fxV, _ = rfx.post_process_fxpts(W, fxV)
    VA = np.concatenate((-VA[:2,::-1],VA[:2,:]), axis=1)
    plt.figure(figsize=(7,7))
    ptr.plot(plt.gca(), VA[:2,:],'-k')
    ptr.scatter(plt.gca(), fxV, 75, c=((0,0,0),))
    V = ptr.lattice([-1.25,-1.25],[1.25,1.25], 20)
    C = np.tanh(W.dot(V))-V
    ptr.quiver(plt.gca(),V,C,scale=.01,units='dots',width=1,color=((.7,.7,.7),), headwidth=7)
    V = VA[:2,:]
    V = V[:,((-1.25 < V) & (V < 1.25)).all(axis=0)]
    C = np.tanh(W.dot(V))-V
    ptr.quiver(plt.gca(),V,C,scale=.005,units='dots',width=2,headwidth=5)
    plt.xlim([-1.25,1.25])
    plt.xlabel('$v_1$',fontsize=24)
    plt.ylim([-1.25,1.25])
    plt.ylabel('$v_2$',fontsize=24,rotation=0)
    # plt.tight_layout()
    plt.show()

def ex1_fig():
    plt.figure(figsize=(8,4.5))
    mpl.rcParams.update({'font.size': 12})
    v = np.linspace(-1.5,1.5,100)
    wc = [(5,'-'),(1,'--'),(.5,'.'),(-1,'.-')]
    for w,c in wc:
        plt.plot(v,np.tanh(w*v)-v,'k'+c)
    mpl.rcParams['mathtext.default'] = 'it'
    plt.legend(['$w=%s$'%w for w,_ in wc],fontsize=18)
    plt.plot(v,np.zeros(v.shape),'k-')
    plt.xlabel('$v$',fontsize=18)
    plt.ylabel('$\sigma(wv)-v$',fontsize=18)
    plt.ylim([-.75,.75])
    plt.tight_layout()
    plt.show()

def bad_c_fig():
    """
    Illustrate critical c, regular regions, fiber topology on a 3-neuron network
    """
    cc.c_space_fig()

def delta_fig():
    """
    Illustrate computation of the delta function (Katz and Reggia 2017)
    """
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams.update({'font.size': 20})
    x = np.linspace(-3,3,100)
    plt.plot(x,1-np.tanh(x)**2,'k-')
    plt.plot(x,np.zeros(x.shape),'k-')
    for (Wv,e,ls) in [(1.2,.2,'k--'),(-.2,.1,'k:')]:
        dtWv = 1-np.tanh(Wv)**2
        ex = [np.arctanh(np.sqrt(1 - (dtWv - e))),np.arctanh(np.sqrt(1 - (dtWv + e)))]
        plt.plot(Wv,dtWv,'ko')
        plt.plot([Wv,Wv],[dtWv-e,dtWv+e],'k-')
        plt.plot([Wv,Wv],[0,dtWv+e],ls)
        plt.plot([x[0],x[-1]],[dtWv-e,dtWv-e],ls)
        plt.plot([x[0],x[-1]],[dtWv+e,dtWv+e],ls)
        plt.plot([-ex[0],-ex[0]],[0,dtWv-e],ls)
        plt.plot([+ex[0],+ex[0]],[0,dtWv-e],ls)
        plt.plot([-ex[1],-ex[1]],[0,dtWv+e],ls)
        plt.plot([+ex[1],+ex[1]],[0,dtWv+e],ls)
    plt.text(1.25,.375,'$\\varepsilon$')
    plt.text(.9,-.075,'$\\delta_i$')
    plt.text(-.4,.985,'$\\varepsilon$')
    plt.text(-.375,-.075,'$\\delta_i$')
    mpl.rcParams.update({'font.size': 12})
    plt.ylabel("$\\sigma'(\\tilde{W}_ix)$",rotation=90,fontsize=16)
    plt.ylim([-.15,1.15])
    plt.xlabel('$\\tilde{W}_ix$',fontsize=16)
    # plt.ylabel("$d\sigma/dx$",rotation=0)
    plt.show()

def mu_fig():
    """
    Illustrate computation of the mu function (Katz and Reggia 2017)
    """
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams.update({'font.size': 20})
    def d2sig(x):
        return 2*np.tanh(x)*(1-np.tanh(x)**2)
    x = np.linspace(-3,3,100)
    plt.plot(x,d2sig(x),'k-')
    plt.plot(x,np.zeros(x.shape),'k-')
    for (Wv,d,ls) in [(2,.4,'k--'),(.5,.3,'k:')]:
        pts = np.array([Wv-d,Wv,Wv+d])
        d2 = d2sig(pts)
        for p in range(3):
            plt.plot([pts[p],pts[p]],[0,d2[p]],ls)
    plt.plot(2-.4,d2sig(2-.4),'ko')
    plt.plot(np.arctanh(np.sqrt(1./3)),np.sqrt(16./27),'ko')
    plt.text(2-.4+.05,d2sig(2-.4),'$2\\mu$')
    plt.text(np.arctanh(np.sqrt(1./3))+.05,np.sqrt(16./27),'$2\\mu$')
    plt.text(2-.1,-.125,'$\\Delta_i$')
    plt.text(.5-.1,-.125,'$\\Delta_i$')
    mpl.rcParams.update({'font.size': 12})
    plt.ylabel("$\\sigma''(\\tilde{W}_ix)$",fontsize=16)
    plt.ylim([-1,1])
    plt.xlabel('$\\tilde{W}_ix$',fontsize=16)
    # plt.ylabel("$d\sigma/dx$",rotation=0)
    plt.show()

def estimate_tanh_eps_error():
    """
    Empirically estimate forward errors in np.tanh implementation
    Seeks e such that error of np.tanh(x) <= e*eps(np.tanh(x))
    Shows plot to aid manual selection of e
    Based on second-order finite differences
    """
    num_samples = 2**12
    num_steps = 16
    high_limit = 1.0
    x = np.zeros((num_samples, num_steps))
    x[:,0] = np.sort(high_limit * np.random.rand(num_samples))
    for i in range(1,num_steps):
        x[:,i] = x[:,i-1] + eps(x[:,i-1])
    ex = eps(x)
    t = np.tanh(x+ex)-np.tanh(x)
    plt.subplot(1,2,1)
    plt.plot(x.flat, t.flat/eps(np.tanh(x.flat)))
    plt.subplot(1,2,2)
    tt = (np.tanh(x[:,1:-1])-np.tanh(x[:,:-2])) - (np.tanh(x[:,2:])-np.tanh(x[:,1:-1]))
    tt = np.maximum(-tt,0)
    plt.plot(x[:,0:-2].flat, (tt/eps(np.tanh(x[:,0:1]))).flat)
    plt.show()

def show_all(comp_test_data_ids, Ns, samp_range, Wc_test_data_id):
    """
    Render all figures from (Katz and Reggia 2017) except critical c visualization
    """
    pert_fig()
    fiber_fig()
    fe.show_tvb_results(test_data_ids=comp_test_data_ids)
    fe.show_tvb_dist_results(test_data_ids=comp_test_data_ids)
    fe.show_tvb_runtimes(test_data_ids=comp_test_data_ids)
    fe.show_tvb_rawcounts(test_data_ids=comp_test_data_ids)
    fe.show_Wc_results(test_data_id=Wc_test_data_id)
    ro.show_traverse_re_fig(comp_test_data_ids, Ns, samp_range)
    ro.show_baseline_re_fig(comp_test_data_ids, Ns, samp_range)
    ro.show_traverse_rd_fig(comp_test_data_ids, Ns, samp_range)
    ro.show_baseline_rd_fig(comp_test_data_ids, Ns, samp_range)
