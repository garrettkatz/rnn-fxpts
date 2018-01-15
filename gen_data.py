import numpy as np

def get_random_discrete(m, n):
    """Get random m-by-n data in {-1,1}."""

    return np.random.choice([-1.0,1.0], size=(m,n))

def get_letters():
    """Get letters dataset (TODO: implement)"""
    
    pass

def get_random_uniform(m,n):
    """Get random m-by-n data in (-1,1)."""

    return 2*np.random.random(size=(m,n)) - 1

def get_random_gap(m,n, bound=0.5):
    """Get random m-by-n data in (-1,-bound]U[bound,1)."""

    return (np.random.random(size=(m,n))*(1-bound)+bound)*np.random.choice([-1,1], size=(m,n))
    
def get_random_approx_discrete(m,n):
    """Get random m-by-n data in {-0.99,0.99}."""

    return np.random.choice([-0.99,0.99], size=(m,n))


def proj(u, v, epsilon=1e-8):
    """Return the projection of v onto u (proj_u(v) = ((v.u)/(u.u))*u"""

    if np.all(np.abs(u)<epsilon): # u ~= 0
        return np.zeros(u.shape)
    return (float(np.dot(v,u))/np.dot(u,u))*u

def gram_schmidt(vects):
    """Perform the Gram-Schmidt process on a set of vectors"""

    res = [vects[:,0] / np.linalg.norm(vects[:,0])]
    for i in xrange(1, vects.shape[1]):
        curr = vects[:,i] - reduce(lambda x,y: x+y, map(lambda x: proj(x,vects[:,i]), res))
        res.append(curr / np.linalg.norm(curr))
    return np.stack(res, axis=1)