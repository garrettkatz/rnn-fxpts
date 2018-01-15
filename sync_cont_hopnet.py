import numpy as np
import sys


class Hopnet:
    def __init__(self, n, gain=1.0):
        """Initialize a synchronous Hopnet"""

        self.n = n
        self.a = np.zeros(n, dtype=np.float32)
        self.W = np.zeros((n,n), dtype=np.float32)
        self.e = 0
        self.gain = gain
        self.activation = np.tanh

    def learn(self, data):
        """Learn the data using Hebbian learning"""

        self.W = np.matmul(data, data.T)
        for i in range(self.n):
            self.W[i,i] = 0
        self.W *= (1.0/self.n)

    def update(self):
        """Update the current activation"""

        self.a = self.activation(self.gain*np.dot(self.W,self.a))


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
            self.e = self.energy(self.a, prev_a)
        else: # Slower mode that provides output at each step
            self.e = self.energy(self.a, None)
            while cont and t < max_steps:
                self.show_state(t, fileid=fileid)

                prev_a = np.copy(self.a)
                self.update()
                self.e = self.energy(self.a, prev_a)
                cont = not np.allclose(prev_a, self.a, rtol=0, atol=tolerance)
                t += 1
            # Show final state
            self.show_state(t, fileid=fileid)
        return t, self.a, self.e

        
    def energy(self, cur_act, prev_act):
        """
        Uses eq. 3.4 in Soulie et al. 1989
        But modifies eq. 3.2 to include gain, and adds gain term to first sum in eq. 3.4 (equivalent to just multiplying the weights by the gain before use).
        If prev_act is None, tries to invert W to find the energy.
        """

        if prev_act is None:
            W_inv = np.linalg.inv(self.W)
            # Test that W_inv can actually successfully invert W
            if not np.allclose(np.eye(self.n), np.matmul(self.W, W_inv), rtol=0, atol=1e-7):
                return float('nan') # TODO: pseudoinverse?
            prev_act = np.dot(W_inv, np.arctanh(cur_act)/self.gain)


        cur_field = self.gain*np.dot(self.W,cur_act)
        prev_field = self.gain*np.dot(self.W,prev_act)

        # summation = prev_act^T * W * cur_act
        summation = self.gain*np.matmul(prev_act[None,:], np.matmul(self.W, cur_act[:,None]))

        # Let c=0; note that log(cosh(0))=0

        # sum_int1 = sum(log(cosh(prev_field)))
        sum_int1 = np.sum(np.log(np.cosh(prev_field)))

        # sum_int2 = sum(log(cosh(cur_field)))
        sum_int2 = np.sum(np.log(np.cosh(cur_field)))

        return (summation - sum_int1 - sum_int2)[0,0]

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

    # def hopfield_energy(self, act):
    #     """
    #     Calculate the energy using Hopfield's derivation.
    #     Note: not a valid energy function for this model.
    #     """
    #     e = 0
    #     for i in xrange(self.n):
    #         for j in xrange(self.n):
    #             e -= self.W[i,j]*act[i]*act[j]
    #     return 0.5*e

    def jacobian(self, v):
        """ 
        Computes the Jacobian of f at v, where f(v)=tanh(gain*Wv)
        f(v)[i] = tanh(gain*W[i,:].v)
        So df[i]/dv[j] = tanh'(gain*W[i,:].v)*gain*W[i,j]
        where tanh' = sech^2 = 1/cosh^2
        """

        res = self.gain/np.cosh(self.gain*np.matmul(self.W, v))**2

        return np.matmul(np.diag(res), self.W)
