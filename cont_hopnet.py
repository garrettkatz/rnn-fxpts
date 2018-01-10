import sync_cont_hopnet as sch
import async_cont_hopnet as ach

def Hopnet(n, synchronous, gain=1.0, stochastic=False):
    """
    Factory function for creating Hopnets.
    Note that the stochastic parameter is ignored when
    synchronous==True.
    """

    print('Initializing a{} discrete-time continuous-valued Hopnet with n={}, gain={}, and tanh activation.'.format(
            ' synchronous' if synchronous else 'n asynchronous'+(' stochastic' if stochastic else ' deterministic'), n, gain))

    if synchronous:
        return sch.Hopnet(n=n, gain=gain)
    else:
        return ach.Hopnet(n=n, gain=gain, stochastic=stochastic)