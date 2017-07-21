import numpy as np
import sys


class Hopnet:
    # TODO: implement batch simulations
    def __init__(self, N, gain=1.0, delta=1.0, synchronous=False, stochastic=True, activation=np.tanh):
        self.N = N
        self.a = np.zeros(N, dtype=np.float32)
        self.W = np.zeros((N,N), dtype=np.float32)
        self.e = 0
        self.gain = gain
        self.delta = delta
        self.synchronous=synchronous
        self.stochastic=stochastic
        self.activation=activation

    def learn(self, data):
        self.W = np.zeros((self.N,self.N), dtype=np.float32)
        for i in range(data.shape[1]):
            self.W += np.outer(data[:,i],data[:,i])
        for i in range(self.N):
            self.W[i,i] = 0
        self.W *= (1.0/self.N)

    def update(self):
        if self.synchronous:
            self.a += self.delta*(self.activation(self.gain*np.dot(self.W,self.a))-self.a)
        else:
            # TODO: does stochastic mean that we update all of them before repeats?
            if self.stochastic:
                order = np.random.permutation(self.N)
            else:
                order = range(self.N)
            for i in order:
                self.a[i] += self.delta*(self.activation(self.gain*np.dot(self.W[i,:],self.a))-self.a[i])

    def simhop(self, a_init, fileid=sys.stdout, trace=True, tolerance=1e-05):
        self.a = np.copy(a_init)
        a_old = None
        self.e = self.energy()

        t = 0
        cont = True

        fileid.write('\n')
        
        while cont:
            if trace:
                self.show_state(t, fileid=fileid)
            t += 1
            a_old = np.copy(self.a)
            self.update()
            self.e = self.energy()
            cont = not np.allclose(a_old, self.a, rtol=0, atol=tolerance)

        self.show_state(t, fileid=fileid)
        return t, np.linalg.norm(a_init-self.a)

        
    def energy(self):
        return -0.5 * np.dot(self.a, np.dot(self.W,self.a))

    def show_state(self, t, fileid=sys.stdout):
        fileid.write("t: {} [".format(t))
        for i in self.a:
            fileid.write(' {:.2f}'.format(i))
        fileid.write(" ]  E: {}\n".format(self.e))

    def show_wts(self, fileid=sys.stdout):
        fileid.write("\nWeights =\n")
        for i in range(self.N):
            for j in range(self.N):
                fileid.write(" {0:7.3f}".format(self.W[i,j]))
            fileid.write("\n")