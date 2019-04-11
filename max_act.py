import torch
import numpy as np
from tqdm import tqdm
import regularization
from copy import deepcopy


'''
maximize image wrt to some objective

args:
    model - model to be maximized
    im_shape - shape of input (should be something like 1 x 3 x H x W)
    objective - 'pred' (maximizes pred) or 'diff' (maximizes diff between feats of two images)
        output_weights - vector to dot with output (decides which classes to maximize)
        im1 - im1 for diff
        im2 - im2 for diff (visualize the vector which maximizes im2 - im1)

returns:
    ims_opt (list) - images
    losses (list) - loss for each of the images
'''
def maximize_im(model, im_shape, 
                objective='pred', output_weights=None, im1=None, im2=None, # objective
                init='zero', num_iters=int(1e3), lr=1e-2, center_crop=False, # params
                lambda_pnorm=0, lambda_tv=0, jitter=0, # regularization
                device='cuda', save_freq=50, viz_online=False): # saving

    # initialize
    if init == 'zero':
        im = torch.zeros(im_shape, requires_grad=True, device=device)
    elif init == 'randn':
        im = torch.randn(im_shape, requires_grad=True, device=device)
        im.data = im.data / 1e4 + 133
    
    if objective == 'diff':
        output_weights = model(im2.to(device)).squeeze().flatten() - model(im1.to(device)).squeeze().flatten()
        
    # setup
    opt = torch.optim.SGD({im}, lr=lr)
    losses = np.zeros(num_iters + 1)
    ims_opt = []
    ox, oy = 0, 0
    for i in tqdm(np.arange(num_iters) + 1):
        model.zero_grad()
        pred = model(im).squeeze() # forward pass
        
        # what we want to minimize
        if objective == 'pred':
            loss = -1 * torch.dot(pred, output_weights).norm()
        elif objective == 'diff':
            loss = -1 * torch.dot(pred.flatten(), output_weights).norm()
        # regularization    
        loss = loss + lambda_pnorm * im.norm(p=6) # p-norm (keeps any pixel from having too large a value)
        loss = loss + lambda_tv * regularization.tv_norm_torch(im) # regularization.tv_norm_torch_3d(im)
        
        # update
        losses[i] = loss.detach().item()
        loss.backward(retain_graph=True)
        opt.step()   
        
        # what to save
        if i % save_freq == 0:
            ims_opt.append(deepcopy(im.detach().cpu()))
            if viz_online:
                viz.viz_clip(ims_opt[-1])
                
        if losses[i] > 1e3: # make sure lr isn't too high
            print('not decreasing!')
            return ims_opt, losses
        if (im != im).sum() > 0: # this works to detect nan
            print('nan!')
            return ims_opt, losses
    
        if jitter:
            im.data = torch.roll(torch.roll(im, shifts=-ox, dims=-1), shifts=-oy, dims=-2).data # apply jitter shift  
            ox, oy = np.random.randint(-jitter, jitter + 1, 2)
            ox, oy = int(ox), int(oy)
            im.data = torch.roll(torch.roll(im, shifts=ox, dims=-1), shifts=oy, dims=-2).data # apply jitter shift              
            
        # zero everything that's not center_crop
        if center_crop:
            mid = im.data.shape[-1] // 2
            low = mid - center_crop // 2
            high = mid + center_crop // 2
            orig_val = im.data[:, :, low: high, low: high].detach()
            im.data = 0 * im.data
            im.data[:, :, low: high, low: high] = orig_val

    
    return ims_opt, losses