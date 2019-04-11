import numpy as np
import torch


# jitter_input_img = np.roll(np.roll(input_img, ox, -1), oy, -2) # apply jitter shift
# # then do the gradient step on jitter_input_img to get next_img
# next_img = np.roll(np.roll(next_img, -ox, -1), -oy, -2) # unshift image
# ox, oy = np.random.randint(-jitter, jitter+1, 2)

# convert im_torch back to unnormalized numpy im
# 1 x 3 x 224 x 224 -> 224 x 224 x 3
def im_to_np(im_torch):
    means = np.array([0.485/0.229, 0.456/0.224, 0.406/0.255]).T
    stds = np.array([0.229, 0.224, 0.255]).T
    im_np = deepcopy(im_torch.cpu().detach().numpy()[0]).transpose((1, 2, 0))
    im_np +=  means
    im_np *=  stds
    return im_np

# convert im_torch back to unnormalized numpy im
# 1 x 3 x 224 x 224 -> 224 x 224 x 3
def clip_to_np(im_torch):
    means = np.array([0.485/0.229, 0.456/0.224, 0.406/0.255]).T
    stds = np.array([0.229, 0.224, 0.255]).T
    im_np = deepcopy(im_torch.cpu().detach().numpy()[0]).transpose((1, 2, 0))
    im_np +=  means
    im_np *=  stds
    return im_np


"""
Compute the total variation norm and its gradient.

The total variation norm is the sum of the image gradient
raised to the power of beta, summed over the image.
We approximate the image gradient using finite differences.
We use the total variation norm as a regularizer to encourage
smoother images.

Note - only naive works in 3d right now!
Inputs:
- x: numpy array of shape (1, delays, C, H, W)
Returns a tuple of:
- loss: Scalar giving the value of the norm
- dx: numpy array of shape (1, delays, C, H, W) giving gradient of the loss
      with respect to the input x.
"""
def tv_norm_torch_3d(x, beta=2.0, verbose=False, operator='naive'):
    
    assert x.shape[0] == 1
    if operator == 'naive':
        x_diff = x[:, :, :, :-1, :-1] - x[:, :, :, :-1, 1:]
        y_diff = x[:, :, :, :-1, :-1] - x[:, :, :, 1:, :-1]
        z_diff = x[:, :-1, :, :-1, :-1] - x[:, 1:, :, :-1, :-1]
    elif operator == 'sobel':
        x_diff = x[:, :, :, :-2, 2:] + 2 * x[:, :, :, 1:-1, 2:] + x[:, :, :, 2:, 2:]
        x_diff -= x[:, :, :, :-2, :-2] + 2 * x[:, :, :, 1:-1, :-2] + x[:, :, :, 2:, :-2]
        y_diff = x[:, :, :, 2:, :-2] + 2 * x[:, :, :, 2:, 1:-1] + x[:, :, :, 2:, 2:]
        y_diff -= x[:, :, :, :-2, :-2] + 2 * x[:, :, :, :-2, 1:-1] + x[:, :, :, :-2, 2:]
    elif operator == 'sobel_squish':
        x_diff = x[:, :, :, :-2, 1:-1] + 2 * \
            x[:, :, :, 1:-1, 1:-1] + x[:, :, :, 2:, 1:-1]
        x_diff -= x[:, :, :, :-2, :-2] + 2 * x[:, :, :, 1:-1, :-2] + x[:, :, :, 2:, :-2]
        y_diff = x[:, :, :, 1:-1, :-2] + 2 * \
            x[:, :, :, 1:-1, 1:-1] + x[:, :, :, 1:-1, 2:]
        y_diff -= x[:, :, :, :-2, :-2] + 2 * x[:, :, :, :-2, 1:-1] + x[:, :, :, :-2, 2:]
    else:
        assert False, 'Unrecognized operator %s' % operator
    
    grad_norm2 = x_diff ** 2.0 + y_diff ** 2.0
#     grad_norm2[grad_norm2 < 1e-3] = 1e-3
    grad_norm_beta = grad_norm2 ** (beta / 2.0)
    
    # add z dim
    grad_norm2z = z_diff ** 2.0
#     grad_norm2z[grad_norm2z < 1e-3] = 1e-3
    grad_norm_betaz = grad_norm2z ** (beta / 2.0)
    
    loss = torch.sum(grad_norm_beta) + torch.sum(grad_norm_betaz)
    return loss
    


"""
Compute the total variation norm and its gradient.

The total variation norm is the sum of the image gradient
raised to the power of beta, summed over the image.
We approximate the image gradient using finite differences.
We use the total variation norm as a regularizer to encourage
smoother images.
Inputs:
- x: numpy array of shape (1, C, H, W)
Returns a tuple of:
- loss: Scalar giving the value of the norm
- dx: numpy array of shape (1, C, H, W) giving gradient of the loss
      with respect to the input x.
"""
def tv_norm_torch(x, beta=2.0, verbose=False, operator='naive'):
    assert x.shape[0] == 1
    if operator == 'naive':
        x_diff = x[:, :, :-1, :-1] - x[:, :, :-1, 1:]
        y_diff = x[:, :, :-1, :-1] - x[:, :, 1:, :-1]
    elif operator == 'sobel':
        x_diff = x[:, :, :-2, 2:] + 2 * x[:, :, 1:-1, 2:] + x[:, :, 2:, 2:]
        x_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, 1:-1, :-2] + x[:, :, 2:, :-2]
        y_diff = x[:, :, 2:, :-2] + 2 * x[:, :, 2:, 1:-1] + x[:, :, 2:, 2:]
        y_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, :-2, 1:-1] + x[:, :, :-2, 2:]
    elif operator == 'sobel_squish':
        x_diff = x[:, :, :-2, 1:-1] + 2 * \
            x[:, :, 1:-1, 1:-1] + x[:, :, 2:, 1:-1]
        x_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, 1:-1, :-2] + x[:, :, 2:, :-2]
        y_diff = x[:, :, 1:-1, :-2] + 2 * \
            x[:, :, 1:-1, 1:-1] + x[:, :, 1:-1, 2:]
        y_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, :-2, 1:-1] + x[:, :, :-2, 2:]
    else:
        assert False, 'Unrecognized operator %s' % operator
        
#     print('norm!')
#     return  x_diff.norm()

    grad_norm2 = x_diff ** 2.0 + y_diff ** 2.0
#     grad_norm2[grad_norm2 < 1e-3] = 1e-3 
    grad_norm_beta = grad_norm2 ** (beta / 2.0)
    loss = torch.sum(grad_norm_beta)
    return loss

    
    
    '''
    dgrad_norm2 = (beta / 2.0) * grad_norm2 ** (beta / 2.0 - 1.0)
    dx_diff = 2.0 * x_diff * dgrad_norm2
    dy_diff = 2.0 * y_diff * dgrad_norm2
    dx = torch.zeros_like(x)
    if operator == 'naive':
        dx[:, :, :-1, :-1] += dx_diff + dy_diff
        dx[:, :, :-1, 1:] -= dx_diff
        dx[:, :, 1:, :-1] -= dy_diff
    elif operator == 'sobel':
        dx[:, :, :-2, :-2] += -dx_diff - dy_diff
        dx[:, :, :-2, 1:-1] += -2 * dy_diff
        dx[:, :, :-2, 2:] += dx_diff - dy_diff
        dx[:, :, 1:-1, :-2] += -2 * dx_diff
        dx[:, :, 1:-1, 2:] += 2 * dx_diff
        dx[:, :, 2:, :-2] += dy_diff - dx_diff
        dx[:, :, 2:, 1:-1] += 2 * dy_diff
        dx[:, :, 2:, 2:] += dx_diff + dy_diff
    elif operator == 'sobel_squish':
        dx[:, :, :-2, :-2] += -dx_diff - dy_diff
        dx[:, :, :-2, 1:-1] += dx_diff - 2 * dy_diff
        dx[:, :, :-2, 2:] += -dy_diff
        dx[:, :, 1:-1, :-2] += -2 * dx_diff + dy_diff
        dx[:, :, 1:-1, 1:-1] += 2 * dx_diff + 2 * dy_diff
        dx[:, :, 1:-1, 2:] += dy_diff
        dx[:, :, 2:, :-2] += -dx_diff
        dx[:, :, 2:, 1:-1] += dx_diff

    def helper(name, x):
        num_nan = torch.isnan(x).sum()
        num_inf = torch.isinf(x).sum()
        num_zero = (x == 0).sum()
        print('%s: NaNs: %d infs: %d zeros: %d' % (name, num_nan, num_inf, num_zero))

    if verbose:
        print('-' * 40)
        print('tv_norm debug output')
        helper('x', x)
        helper('x_diff', x_diff)
        helper('y_diff', y_diff)
        helper('grad_norm2', grad_norm2)
        helper('grad_norm_beta', grad_norm_beta)
        helper('dgrad_norm2', dgrad_norm2)
        helper('dx_diff', dx_diff)
        helper('dy_diff', dy_diff)
        helper('dx', dx)
        print('\n')

    return loss, dx
    '''

# same as above function but uses np instead of torch
def tv_norm_np(x, beta=2.0, verbose=False, operator='naive'):
    assert x.shape[0] == 1
    if operator == 'naive':
        x_diff = x[:, :, :-1, :-1] - x[:, :, :-1, 1:]
        y_diff = x[:, :, :-1, :-1] - x[:, :, 1:, :-1]
    elif operator == 'sobel':
        x_diff = x[:, :, :-2, 2:] + 2 * x[:, :, 1:-1, 2:] + x[:, :, 2:, 2:]
        x_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, 1:-1, :-2] + x[:, :, 2:, :-2]
        y_diff = x[:, :, 2:, :-2] + 2 * x[:, :, 2:, 1:-1] + x[:, :, 2:, 2:]
        y_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, :-2, 1:-1] + x[:, :, :-2, 2:]
    elif operator == 'sobel_squish':
        x_diff = x[:, :, :-2, 1:-1] + 2 * \
            x[:, :, 1:-1, 1:-1] + x[:, :, 2:, 1:-1]
        x_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, 1:-1, :-2] + x[:, :, 2:, :-2]
        y_diff = x[:, :, 1:-1, :-2] + 2 * \
            x[:, :, 1:-1, 1:-1] + x[:, :, 1:-1, 2:]
        y_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, :-2, 1:-1] + x[:, :, :-2, 2:]
    else:
        assert False, 'Unrecognized operator %s' % operator
    grad_norm2 = x_diff ** 2.0 + y_diff ** 2.0
    grad_norm2[grad_norm2 < 1e-3] = 1e-3
    grad_norm_beta = grad_norm2 ** (beta / 2.0)
    loss = np.sum(grad_norm_beta)
    dgrad_norm2 = (beta / 2.0) * grad_norm2 ** (beta / 2.0 - 1.0)
    dx_diff = 2.0 * x_diff * dgrad_norm2
    dy_diff = 2.0 * y_diff * dgrad_norm2
    dx = np.zeros_like(x)
    if operator == 'naive':
        dx[:, :, :-1, :-1] += dx_diff + dy_diff
        dx[:, :, :-1, 1:] -= dx_diff
        dx[:, :, 1:, :-1] -= dy_diff
    elif operator == 'sobel':
        dx[:, :, :-2, :-2] += -dx_diff - dy_diff
        dx[:, :, :-2, 1:-1] += -2 * dy_diff
        dx[:, :, :-2, 2:] += dx_diff - dy_diff
        dx[:, :, 1:-1, :-2] += -2 * dx_diff
        dx[:, :, 1:-1, 2:] += 2 * dx_diff
        dx[:, :, 2:, :-2] += dy_diff - dx_diff
        dx[:, :, 2:, 1:-1] += 2 * dy_diff
        dx[:, :, 2:, 2:] += dx_diff + dy_diff
    elif operator == 'sobel_squish':
        dx[:, :, :-2, :-2] += -dx_diff - dy_diff
        dx[:, :, :-2, 1:-1] += dx_diff - 2 * dy_diff
        dx[:, :, :-2, 2:] += -dy_diff
        dx[:, :, 1:-1, :-2] += -2 * dx_diff + dy_diff
        dx[:, :, 1:-1, 1:-1] += 2 * dx_diff + 2 * dy_diff
        dx[:, :, 1:-1, 2:] += dy_diff
        dx[:, :, 2:, :-2] += -dx_diff
        dx[:, :, 2:, 1:-1] += dx_diff

    def helper(name, x):
        num_nan = np.isnan(x).sum()
        num_inf = np.isinf(x).sum()
        num_zero = (x == 0).sum()
        print('%s: NaNs: %d infs: %d zeros: %d' % (name, num_nan, num_inf, num_zero))

    if verbose:
        print('-' * 40)
        print('tv_norm debug output')
        helper('x', x)
        helper('x_diff', x_diff)
        helper('y_diff', y_diff)
        helper('grad_norm2', grad_norm2)
        helper('grad_norm_beta', grad_norm_beta)
        helper('dgrad_norm2', dgrad_norm2)
        helper('dx_diff', dx_diff)
        helper('dy_diff', dy_diff)
        helper('dx', dx)
        print('\n')

    return loss, dx