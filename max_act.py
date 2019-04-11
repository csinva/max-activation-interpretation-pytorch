import torch
import numpy as np
from tqdm import tqdm
import regularization
from copy import deepcopy

# maximize im specified by pattern
def maximize_im(model, im_shape, output_weights, 
                init='zero', num_iters=int(1e3), lr=1e-2, 
                lambda_pnorm=0, lambda_tv=0, jitter=0,
                device='cuda', save_freq=50, viz_online=False, center_crop=False):
    if init == 'zero':
        im = torch.zeros(im_shape, requires_grad=True, device=device)
    elif init == 'randn':
        im = torch.randn(im_shape, requires_grad=True, device=device)
        im.data = im.data / 1e4 + 133
    opt = torch.optim.SGD({im}, lr=lr)
    losses = np.zeros(num_iters + 1)
    ims_opt = []
    for i in tqdm((np.arange(num_iters) + 1)):
        model.zero_grad()
        
        if jitter:
            ox, oy = np.random.randint(-jitter, jitter + 1, 2)
            ox, oy = int(ox), int(oy)
            im.data = torch.roll(torch.roll(im, shifts=ox, dims=-1), shifts=oy, dims=-2).data # apply jitter shift    
        pred = model(im).squeeze() # forward pass
        
        # what we want to minimize
        loss = -1 * torch.dot(pred, output_weights).norm()
        loss = loss + lambda_pnorm * im.norm(p=6) # p-norm (keeps any pixel from having too large a value)
#         loss = loss + lambda_tv * regularization.tv_norm_torch_3d(im)
        loss = loss + lambda_tv * regularization.tv_norm_torch(im)
        losses[i] = loss.detach().item()
        loss.backward(retain_graph=True)
        opt.step()   
        
        # what to save
        if i % save_freq == 0:
            ims_opt.append(deepcopy(im.detach().cpu()))
            if viz_online:
                viz.viz_clip(ims_opt[-1])
                
        # this works to detect nan
        def torch_is_nan(x):
            return x != x
        if torch_is_nan(im).sum() > 0:
            print('nan!')
            return ims_opt, losses
    
        if jitter:
            im.data = torch.roll(torch.roll(im, shifts=-ox, dims=-1), shifts=-oy, dims=-2).data # apply jitter shift    
            
        # zero everything that's not center_crop
        if center_crop:
#             torch.Size([1, 3, 224, 224])
            mid = im.data.shape[-1] // 2
            low = mid - center_crop // 2
            high = mid + center_crop // 2
            orig_val = im.data[:, :, low: high, low: high].detach()
            im.data = 0 * im.data
            im.data[:, :, low: high, low: high] = orig_val

    
    return ims_opt, losses