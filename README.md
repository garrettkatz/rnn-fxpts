# rnn-fxpts

`rnn-fxpts` is a fixed point solver for discrete-time, continuous-valued, Hopfield-type recurrent neural networks with arbitrary connectivity, written in Python.

## Requirements

`rnn-fxpts` has been tested using Python 2.7 on Fedora 24 and Ubuntu 14.04, but it may work with other OS's and more recent versions.  ``rnn-fxpts`` depends on the [SciPy](http://www.scipy.org) scientific computing stack.  In particular, it has been tested using the following SciPy modules:
* [scipy](http://www.scipy.org/scipylib/index.html) (version 0.13.3)
* [numpy](http://www.numpy.org/) (version 1.11.1)
* [matplotlib](http://matplotlib.org/) (version 1.3.1)

## Installation

`rnn-fxpts` isn't currently set up for automatic installation.  To use `rnn-fxpts`, simply [clone or download](https://help.github.com/articles/cloning-a-repository/) the repository into a directory of your choice.  You may wish to configure your environment so that `rnn-fxpts` is readily available in other directories and projects (e.g., add the directory you chose to your [PYTHONPATH](https://docs.python.org/2/using/cmdline.html#envvar-PYTHONPATH) environment variable).

## Documentation

[Basic usage](https://github.com/garrettkatz/rnn-fxpts#basic-usage) of ``rnn-fxpts`` is described below.  For full API documentation, you can use Python's built-in ``help`` function:

```python
>>> import rnn_fxpts
>>> help(rnn_fxpts)
>>> help(rnn_fxpts.run_solver)
```
and so on.

For more information on algorithmic details and proofs, please consult the following:

[Katz, G. E., Reggia, J. A. (2017). Using Directional Fibers to Locate Fixed Points of Recurrent Neural Networks. IEEE Transactions on Neural Networks and Learning Systems (accepted). IEEE.](https://doi.org/10.1109/TNNLS.2017.2733544) (Here's a [preprint](https://www.cs.umd.edu/~gkatz/TNNLS-2016-P-7293.R2.pdf))

[Katz, G. E and Reggia, J. A. (2016).  Identifying Fixed Points in Recurrent Neural Networks Using Directional Fibers: Supplemental Material on Theoretical Results and Practical Aspects of Numerical Traversal.  University of Maryland, College Park, Technical Report CS-TR-5051.](http://hdl.handle.net/1903/18918)

Release [v1.0](https://github.com/garrettkatz/rnn-fxpts/releases/tag/v1.0) contains all of the code used to produce the figures and results reported in the foregoing references.

### Errata

After publication of the foregoing references, a methodological flaw was discovered in the stability analysis.  In particular, stability was incorrectly measured using the eigenvalues of *Df*, where *f(v) = tanh(Wv) - v*.  This can sometimes falsely classify stable fixed points as unstable and vice versa.  Instead, stability should have been measured using *Dm*, where *m(v) = tanh(Wv)*.  This correction has been made in the latest version of the code (as of [this commit](https://github.com/garrettkatz/rnn-fxpts/commit/df761d993b3ba83026b01f1550babb6075cdb1fa)).  As it happens, there were no substantial qualitative changes to the results or conclusions as a whole, although there were some minor quantitative differences.

## Reproducing the Experimental Results

Release [v1.0](https://github.com/garrettkatz/rnn-fxpts/releases/tag/v1.0) contains all of the code used to produce the figures and results reported in the foregoing references.  To run all of the experiments, invoke the ``reproduce_results.py`` script from the command line:

```shell
$ python reproduce_results.py
```

The script will prompt you to choose the number of processors to use, and one of three experimental scales:
- **Full**: This option runs the experiments at full scale, using the same number of networks with the same sizes (up to ``N=128``) as reported in the papers.  This option is computationally expensive - on our workstation, using ten 3.5GHz Intel Xeon CPU cores, it ran for 50 hours, at times using upwards of 32GB of RAM, and ultimately saving almost 86GB of results to the hard-drive.  If you have more limited computational resources consider choosing the second option.
- **Mini**: This option runs the experiments at reduced scale, using fewer networks limited to smaller sizes (up to ``N=64``).  This scale is more appropriate for personal computing resources - on one of our laptops, using four 2.4GHz Intel Core i7 CPU cores, it ran for about 8 hours, using no more than 8GB of RAM, and ultimately saving about 1.5GB of results to the hard-drive.
- **Micro**: This option runs the experiments at very small scale, and should finish in a matter of minutes.  Good for quick testing, but the figures will have very few data points.

Over the course of the experiments, results for each network tested are written to data files in the ``results`` sub-directory.  Progress updates as each network is being tested are written to text files in the ``logs`` sub-directory.  In a Linux shell, you can use

```shell
$ ls -lst logs/* | head
```

to see the log files most recently updated, and

```shell
$ tail logs/<logfile> && echo
```

to monitor the latest progress in one of the logs.  When the experiments are all complete, the figures will be generated and displayed one at a time (close the current figure and the next will automatically open).

## Basic Usage

### Neural Network Model

`rnn-fxpts` is designed for discrete-time, continuous-valued, Hopfield-type recurrent neural networks with arbitrary connectivity.  In code, the network model is:

```python
>>> import numpy as np
>>> N = np.random.randint(128) # arbitrary number of neurons (within reason)
>>> W = np.random.randn(N,N) # arbitrary connection weight matrix
>>> v = np.random.randn(N,1) # arbitrary neural state
>>> v_new = np.tanh(W.dot(v)) # activation rule
``` 

### Running the Solver

For the sake of example, let's use ``N = 2``.  Sampling distributions for ``W`` with larger diagonals tend to have more fixed points, so we'll do that to keep things interesting:
```python
>>> import rnn_fxpts as rfx
>>> N = 2
>>> W = 1.25*np.eye(N) + 0.1*np.random.randn(N,N)
>>> W
array([[ 1.24980837, -0.01702046],
       [ 0.16869383,  1.23345761]])
>>> fxpts, _ = rfx.run_solver(W)
>>> fxpts
array([[-0.        , -0.04823415, -0.69112426,  0.04823415,  0.69112426],
       [-0.        , -0.70572931, -0.80303357,  0.70572931,  0.80303357]])
```

Every column of ``fxpts`` is a distinct fixed point (up to machine precision).  Let's check that:

```python
>>> residual_error = np.tanh(W.dot(fxpts))-fxpts
>>> residual_error
array([[  0.00000000e+00,  -6.93889390e-18,   1.11022302e-16,   6.93889390e-18,  -1.11022302e-16],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00]])
```

``fixed_within_eps`` is a helper function for dealing with those small round-off errors, based on a numerical forward error analysis.  It returns a boolean mask indicating whether each point should be considered fixed, as well as an acceptable margin of error, given the inherent limitations of finite-precision arithmetic.  If any residual error is larger than the error margin, the corresponding point is definitely not fixed.

```python
>>> is_fixed, error_margin = rfx.fixed_within_eps(W, fxpts)
>>> error_margin
array([[  4.94065646e-323,   8.68897712e-017,   1.13984698e-015,   8.68897712e-017,   1.13984698e-015],
       [  4.94065646e-323,   1.13731257e-015,   1.37691541e-015,   1.13731257e-015,   1.37691541e-015]])
>>> np.fabs(residual_error) < error_margin
array([[ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True]], dtype=bool)
>>> is_fixed
array([ True,  True,  True,  True,  True], dtype=bool)
```

The solver works by numerically traversing mathematical objects called *directional fibers*.  Directional fibers are curves in (``N+1``)-dimensional space that pass through the fixed points of the network.  ``run_solver`` returns the fiber that it traversed as its second output:

```python
>>> fxpts, fiber = rfx.run_solver(W)
>>> fiber[:,:10]
array([[ 0.        , -0.05901627, -0.08883874, -0.09876817, -0.09286486, -0.07291058, -0.03912487,  0.00832132,  0.06924304,  0.14426246],
       [ 0.        ,  0.15760582,  0.26935026,  0.36039185,  0.44021681,  0.51471122,  0.58729197,  0.65849854,  0.72753705,  0.79252402],
       [ 0.        ,  0.03020415,  0.04591744,  0.05259942,  0.05256338,  0.04657952,  0.0344436 ,  0.01596019, -0.00835462, -0.03674596]])
```

Each column of ``fiber`` is another point along the fiber that was computed over the course of the traversal.  The first ``N`` coordinates coincide with the neural state space; the last coordinate is an additional parameter that vanishes when fixed points are encountered.  To help you get a better grasp on the intuition behind directional fibers, ``show_fiber`` will visualize them for you when ``N == 2``:

```python
>>> rfx.show_fiber(W, fxpts, fiber)
```
![Directional Fiber 1](https://cloud.githubusercontent.com/assets/6537102/21059296/bc76324e-be0f-11e6-9b5f-24a3cc928711.png)

This command plots the first ``N`` coordinates of the fiber as a black curve, superimposed on the neural state space.  The fixed points are plotted as black circles.  At regularly spaced grid points ``v``, the update vector ``np.tanh(W.dot(v)) - v`` is shown with light gray arrows.  It is also shown with black arrows at each point along the fiber.  As you can see from the plot, every point in a directional fiber updates in the same constant direction, just with different sign and magnitude.  In fact, this signed magnitude is precisely what the extra coordinate (i.e., ``fiber[N,:]``) represents.  It changes sign (passes through zero) every time another fixed point is encountered.

By default, ``traverse`` chooses a constant direction vector ``c`` for you, which determines the directional fiber that gets traversed.  The choice of ``c`` also determines which fixed points get found - not every choice will return every fixed point, especially in higher dimensions.  If you want to try a specific choice, you can supply the desired direction vector as a keyword argument, like so:

```python
>>> fxpts, fiber = rfx.run_solver(W, c = np.ones((N,1)))
>>> rfx.show_fiber(W, fxpts, fiber)
```

![Directional Fiber 2](https://cloud.githubusercontent.com/assets/6537102/21059295/bc75e58c-be0f-11e6-83e2-635b06f7b1e6.png)

### Baseline Solver

As a baseline for comparison, ``rnn-fxpts`` also includes a Python port of a [more conventional approach](http://dx.doi.org/10.1162/NECO_a_00409), based on randomly seeded local optimization.  The optimization is designed to minimize the magnitude of the network update vector (i.e., ``np.tanh(W.dot(v))-v``), which will be zero at fixed points.  This method repeatedly samples and optimizes random seeds until reaching a user-specified timeout, which you can supply as a keyword argument (in seconds).

```python
>>> W = 1.25*np.eye(N) + 0.1*np.random.randn(N,N)
>>> W
array([[ 1.3216786 , -0.06273015],
       [ 0.15134484,  1.19697091]])
>>> pts, _ = rfx.baseline_solver(W, timeout=1):
```

Each column of ``pts`` is a local optimum corresponding to one repetition.  This method can perform many repetitions very quickly:
```python
>>> pts.shape[1]
1075
```

However, many of these points may be duplicates where different random seeds converged to the same result.  In addition, many of these points might not be fixed, for two reasons.  First, the optimization might find non-fixed "slow" points, which are points where the magnitude of the update is a non-zero local minimum.  Second, since the method uses a generic optimization routine, it might terminate with looser tolerances than prescribed by ``fixed_within_eps``:

```python
>>> is_fixed, error_margin = rfx.fixed_within_eps(W, pts[:,:5])
>>> np.tanh(W.dot(pts[:,:5]))-pts[:,:5]
array([[ -1.36280648e-02,  -1.23707628e-07,   6.68663798e-07,  -1.05537150e-06,  -1.36632679e-02],
       [ -5.03948586e-02,  -9.41497404e-08,   1.62992910e-07,  -1.24318014e-07,  -5.03851346e-02]])
>>> error_margin
array([[  1.37146335e-15,   1.15290097e-15,   1.15290097e-15,   1.15290097e-15,   1.37146335e-15],
       [  6.93870549e-16,   1.37093845e-15,   1.37093845e-15,   1.37093845e-15,   6.93870549e-16]])
>>> is_fixed
array([False, False, False, False, False], dtype=bool)
```

The helper function ``refine_pts`` refines the results to greater precision using the Newton-Raphson method.  It returns the refined points as well as a boolean mask indicating whether each one converged to a fixed point (according to ``fixed_within_eps``).

```python
>>> pts, converged = rfx.refine_pts(W, pts)
>>> _, error_margin = rfx.fixed_within_eps(W, pts[:,:5])
>>> np.tanh(W.dot(pts[:,:5]))-pts[:,:5]
array([[ -1.68152987e-03,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,  -4.94232180e-04],
       [  6.79826584e-02,  -1.11022302e-16,   0.00000000e+00,   0.00000000e+00,   5.79939812e-02]])
>>> error_margin
array([[  1.37494557e-15,   1.15290097e-15,   1.15290097e-15,   1.15290097e-15,   1.37494557e-15],
       [  1.14889384e-15,   1.37093845e-15,   1.37093845e-15,   1.37093845e-15,   8.71338085e-16]])
>>> converged[:5]
array([False,  True,  True,  True, False], dtype=bool)
```

As for duplicates, the helper function ``get_unique_fxpts`` uses pair-wise comparisons and more error analyses to detect and remove them for you:

```python
>>> fxpts = pts[:,converged]
>>> unique_fxpts = rfx.get_unique_fxpts(W, fxpts)
>>> fxpts.shape[1]
703
>>> unique_fxpts.shape[1]
5
```

This workflow is automated by the helper function ``post_process_fxpts``:

```python
unique_fxpts, _ = rfx.post_process_fxpts(W, pts)
```

This function call will do both refinement and duplicate removal.  It also includes ``-fxpts`` in its output, since the fixed points of our network model always come in +/- pairs.  This same post-processing is also used by ``run_solver`` under the hood.  In principle, fiber traversal should encounter every fixed point at most once, obviating the need for duplicate removal.  However, for added redundancy, *three* fixed point seeds are actually refined at every step where ``fiber[N,:]`` changes sign:  the point on the fiber before the step, the point after the step, and a linear interpolant of the two.
