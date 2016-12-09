import numpy as np
import rnn_fxpts as rfx

np.set_printoptions(linewidth=1000)

N = np.random.randint(128) # arbitrary number of neurons (within reason)
W = np.random.randn(N,N) # arbitrary connection weight matrix
v = np.random.randn(N,1) # arbitrary neural state
v_new = np.tanh(W.dot(v)) # activation rule

N = 2
# W = 1.25*np.eye(N) + 0.1*np.random.randn(N,N)
# print('>>> W')
# print(repr(W))
# fxpts, _ = rfx.run_solver(W)
# print('>>> fxpts')
# print(repr(fxpts))

# residual_error = np.tanh(W.dot(fxpts)) - fxpts
# print('>>> residual_error')
# print(repr(residual_error))

# is_fixed, error_margin = rfx.fixed_within_eps(W, fxpts)
# print('>>> error_margin')
# print(repr(error_margin))
# print('>>> np.fabs(residual_error) < error_margin')
# print(repr(np.fabs(residual_error) < error_margin))
# print('>>> is_fixed')
# print(repr(is_fixed))

# fxpts, fiber = rfx.run_solver(W)
# print('>>> fiber[:,:10]')
# print(repr(fiber[:,:10]))

# rfx.show_fiber(W, fxpts, fiber)
# # rfx.show_fiber(W, fxpts, fiber, savefile='dfiber1.png')

# fxpts, fiber = rfx.run_solver(W, c = np.ones((N,1)))
# rfx.show_fiber(W, fxpts, fiber)
# # rfx.show_fiber(W, fxpts, fiber, savefile='dfiber2.png')

W = 1.25*np.eye(N) + 0.1*np.random.randn(N,N)
print('>>> W')
print(repr(W))
pts, _ = rfx.baseline_solver(W, timeout=1)

print('>>> pts.shape[1]')
print(repr(pts.shape[1]))

is_fixed, error_margin = rfx.fixed_within_eps(W, pts[:,:5])
print('>>> np.tanh(W.dot(pts[:,:5]))-pts[:,:5]')
print(repr(np.tanh(W.dot(pts[:,:5]))-pts[:,:5]))
print('>>> error_margin')
print(repr(error_margin))
print('>>> is_fixed')
print(repr(is_fixed))

pts, converged = rfx.refine_pts(W, pts)
_, error_margin = rfx.fixed_within_eps(W, pts[:,:5])
print('>>> np.tanh(W.dot(pts[:,:5]))-pts[:,:5]')
print(repr(np.tanh(W.dot(pts[:,:5]))-pts[:,:5]))
print('>>> error_margin')
print(repr(error_margin))
print('>>> converged[:5]')
print(repr(converged[:5]))

fxpts = pts[:,converged]
unique_fxpts = rfx.get_unique_fxpts(W, fxpts)
print('>>> fxpts.shape[1]')
print(repr(fxpts.shape[1]))
print('>>> unique_fxpts.shape[1]')
print(repr(unique_fxpts.shape[1]))

unique_fxpts, _ = rfx.post_process_fxpts(W, pts)
print('>>> unique_fxpts')
print(repr(unique_fxpts))
