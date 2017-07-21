from numpy import *

# Hamming dist between vectors a, b
def hamdist(a,b): return sum(a != b)  # Hamming dist vectors a, b

# Hopfield Network Class Definition
class hopnet(object):

  def __init__ (self,N):              # create N-node Hopfield Net 
    self.N = N
    self.A = zeros(N)                 # A = activity state
    self.W = zeros((N,N))             # W = N x N weight matrix
    self.E = 0                        # E = current network energy

  def learn (self,data,N,M):          # learn matrix W from data
    for i in range(M):                # for each input pattern i
       self.W = self.W + outer(data[:,i],data[:,i])
    self.W = (1.0 / N) * (self.W - M * eye(N,N))  # zeros main diag.

  def sgn(self,input,oldval):         # compute a = sgn(input)
    if input > 0: return 1
    elif input < 0: return -1
    else: return oldval

  def update(self):                       # asynchronously update A
    # indices = random.permutation(self.N)  # determine order node updates
    indices = range(self.N)
    for i in indices:                     # for each node i
      scalprod = dot(self.W[i,:],self.A)         # compute i's input
      self.A[i] = self.sgn(scalprod,self.A[i])   # assign i's act. value
    return

  def simhop(self,fileid,Ainit,trace):
    # Simulate Hopfield net starting in state Ainit.
    # Returns iteration number tlast and Hamming distance dist
    # of A from stored pattern Ainit when final state reached.
    # trace = 1 prints in fileid state A and energy E at each t
    t = 0                        # initialize time step t
    self.A = copy(Ainit)         # assign initial state A
    self.E = self.energy()       # compute energy E
    fileid.write("\n")
    self.showstate(fileid,t,self.E)
    Aold = copy(Ainit)           # A at previous t
    moretodo = True              # not known to be at equilibrium yet
    while moretodo:              # while fixed point not reached
      t += 1                     #   increment iteration counter
      self.update()              #   update all A values once per t
      self.E = self.energy()     #   compute energy E of state A
      if all(self.A == Aold):    #   if at fixed point
        tlast = t                #      record ending iteration
        dist = hamdist(Ainit,self.A)    # distance from Ainit
        moretodo = False         #      and quit
      elif trace:
        self.showstate(fileid,t,self.E)
      Aold = copy(self.A)
    self.showstate(fileid,t,self.E)
    return tlast,dist

  def energy(self):             # Returns network's energy E 
    return -0.5 * dot(self.A,dot(self.W,self.A))

  def showstate(self,fileid,t,E):  # display A at time t
    fileid.write("t: {} ".format(t))
    for i in range(self.N):
      if self.A[i] == 1: fileid.write("+")
      else: fileid.write("-")
    fileid.write("  E: {} ".format(E))
    fileid.write("\n")

  def showwts(self,fileid):    # display W 
    fileid.write("\nWeights =\n")
    for i in range(self.N):
      for j in range(self.N):
        fileid.write(" {0:7.3f}".format(self.W[i,j]))
      fileid.write("\n")

