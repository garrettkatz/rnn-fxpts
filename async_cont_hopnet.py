import numpy as np
import sys


class Hopnet:
    def __init__(self, n, gain=1.0, stochastic=False):
        """Initialize an asynchronous Hopnet"""

        self.n = n
        self.a = np.zeros(n, dtype=np.float32)
        self.W = np.zeros((n,n), dtype=np.float32)
        self.e = 0
        self.gain = gain
        self.activation=np.tanh
        self.stochastic=stochastic

    def learn(self, data):
        """Learn the data using Hebbian learning"""

        self.W = np.matmul(data, data.T)
        for i in range(self.n):
            self.W[i,i] = 0
        self.W *= (1.0/self.n)

    def update(self):
        """Update the current activation"""

        if self.stochastic:
            order = np.random.permutation(self.n)
        else:
            order = range(self.n)
        for i in order:
            self.a[i] = self.activation(self.gain*np.dot(self.W[i,:], self.a))


    def simhop(self, a_init, tolerance=1e-05, max_steps=500, silent=False, fileid=sys.stdout):
        """Simulate the Hopnet until termination conditions are reached"""

        if len(a_init.shape) == 1:
            a_init = a_init.flatten()
        if len(a) != self.n:
            raise ValueError('The given a_init does not have {} entries.'.format(self.n))

        self.a = np.copy(a_init)

        t = 0
        cont = True

        if silent: # More optimized mode without any output
            while cont and t < max_steps:
                prev_a = np.copy(self.a)
                self.update()
                cont = not np.allclose(prev_a, self.a, rtol=0, atol=tolerance)
                t += 1
            # Update the energy only at the very end
            self.e = self.energy(self.a)
        else: # Slower mode that provides output at each step
            self.e = self.energy(self.a)
            while cont and t < max_steps:
                self.show_state(t, fileid=fileid)

                prev_a = np.copy(self.a)
                self.update()
                self.e = self.energy(self.a)
                cont = not np.allclose(prev_a, self.a, rtol=0, atol=tolerance)
                t += 1
            # Show final state
            self.show_state(t, fileid=fileid)
        return t, self.a, self.e

        
    def energy(self, cur_act):
        """
        Uses eq. 4.3 in Soulie et al. 1989 with b_i=0
        But modifies eq. 4.2 to include gain, and adds gain term to first sum in eq. 4.3 (equivalent to just multiplying the weights by the gain before use).
        """

        # summation = -0.5 * cur_act^T * W * cur_act
        summation = -0.5 * self.gain*np.matmul(cur_act[None,:], np.matmul(self.W, cur_act[:,None]))

        # integral(arctanh(x)) = 0.5*log(1-x^2) + x*arctanh(x); note that this equals 0 at 0

        sum_int = np.sum(0.5*np.log(1 - cur_act**2) + cur_act * np.arctanh(cur_act))

        return (summation + sum_int)[0,0]

    def show_state(self, t, fileid=sys.stdout):
        """Print the current state"""

        fileid.write("t:{:4d} [".format(t))
        for i in self.a:
            fileid.write(' {0: 5.3f}'.format(i))
        fileid.write(" ]  E: {}\n".format(self.e))

    def show_wts(self, fileid=sys.stdout):
        """Print the weight matrix"""

        fileid.write("\nWeights =\n")
        for i in range(self.n):
            for j in range(self.n):
                fileid.write(" {0: 7.3f}".format(self.W[i,j]))
            fileid.write("\n")

    # def jacobian(self, v):
    #     """Old Jacobian derivation; INCORRECT"""
    #     intermediate_v = [v]
    #     # Assume stochastic=False
    #     for i in xrange(self.n-1):
    #         prev = np.copy(intermediate_v[i])
    #         prev[i] = self.activation(self.gain*np.dot(self.W[i,:], prev))
    #         intermediate_v.append(prev)
    #     # Note: intermediate_v is shifted: intermediate_v[i] = v^(i-1) in notes

    #     intermediate_inv_cosh = [self.gain/np.cosh(self.gain*np.dot(self.W[i,:],intermediate_v[i]))**2 for i in xrange(self.n)]

    #     # Array of dv^(i)_i/dv^(k)_k
    #     partial_intermediate_of_intermediate = np.zeros((self.n,self.n))
    #     for i in xrange(1,self.n):
    #         for k in xrange(i-1,-1,-1): # iterates through 0...i-1 backwards
    #             partial_intermediate_of_intermediate[i,k] = self.W[i,k]*intermediate_inv_cosh[i]
    #             for l in xrange(k+1, i): # i-1 in notes; will be empty range for some i,k
    #                 partial_intermediate_of_intermediate[i,k] += partial_intermediate_of_intermediate[i,l]*partial_intermediate_of_intermediate[l,k]
    #     # for k in xrange(self.n-1):
    #     #     for i in xrange(k+1,self.n):
    #     #         partial_intermediate_of_intermediate[i,k] = self.W[i,k]*intermediate_inv_cosh[i]
    #     #         for l in xrange(k+1, i): # Will be empty range for some i,k
    #     #             partial_intermediate_of_intermediate[i,k] += partial_intermediate_of_intermediate[l,k]

    #     # Array of dv^(i)_i/dv_j
    #     partial_intermediate_of_v = np.zeros((self.n,self.n))
    #     for i in xrange(self.n):
    #         for j in xrange(self.n):
    #             for k in xrange(i): # i-1 in notes; will be empty range for i=0
    #                 partial_intermediate_of_v[i,j] += partial_intermediate_of_intermediate[i,k]*partial_intermediate_of_v[k,j]
    #             if j >= i:
    #                 partial_intermediate_of_v[i,j] += self.W[i,j]*intermediate_inv_cosh[i]

    #     return partial_intermediate_of_v #- np.eye(self.n)


    def jacobian(self, v):
        """ 
        Computes the Jacobian of f at v, where f(v)=tanh(gain*Wv).
        See Garrett's notes for derivation.
        TODO: optimize
        """

        def partial_intermediate_partial_intermediate(v_prev,i,j,k):
            if i == j:
                return self.gain*self.W[i,k]/np.cosh(self.gain*np.dot(self.W[i,:],v_prev))**2
            elif j == k:
                return 1.0
            else:
                return 0.0

        intermediate_v = [v]
        # Assume stochastic=False
        for i in xrange(self.n):
            prev = np.copy(intermediate_v[i])
            prev[i] = self.activation(self.gain*np.dot(self.W[i,:], prev))
            intermediate_v.append(prev)
        # Note: intermediate_v is shifted: intermediate_v[i] = v^(i-1) in notes

        prev_M_prod = np.zeros((self.n,self.n))

        # M^(0)
        for j in xrange(self.n):
            for l in xrange(self.n):
                prev_M_prod[j,l] = partial_intermediate_partial_intermediate(v, 0, j, l)


        for i in xrange(1,self.n+1):
            cur_M_prod = np.zeros((self.n,self.n))

            for j in xrange(self.n):
                for l in xrange(self.n):
                    for k in xrange(self.n):
                        cur_M_prod[j,l] += partial_intermediate_partial_intermediate(intermediate_v[i], i, j, k)*prev_M_prod[k,l]
            prev_M_prod = cur_M_prod

        return prev_M_prod



        
