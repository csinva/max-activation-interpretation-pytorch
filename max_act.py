import torch
import regularization

# maximize im specified by pattern
def maximize_im(model, output_weights, 
                init='zero', num_iters=int(1e3), lr=1e-2, 
                lambda_pnorm=0, lambda_tv=0,
                device='cuda', save_freq=50, viz_online=False):
    if init == 'zero':
        im = torch.zeros(1, 11, 3, 64, 64, requires_grad=True, device=device)
    elif init == 'randn':
        im = torch.randn(1, 11, 3, 64, 64, requires_grad=True, device=device) 
    opt = torch.optim.SGD({im}, lr=lr) #, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#     im = im * 0.0001 #* torch.Tensor([0.01]).cuda()
    losses = np.zeros(num_iters + 1)
    ims_opt = []
    for i in (np.arange(num_iters) + 1):
        model.zero_grad()
        pred = model(im).squeeze() # forward pass
        # what we want to minimize
        loss = -1 * torch.dot(pred, output_weights).norm()
        loss = loss + lambda_pnorm * im.norm(p=6) # p-norm
        loss = loss + lambda_tv * regularization.tv_norm(im)[0]
        losses[i] = loss.detach().item()
        loss.backward(retain_graph=True)
        opt.step()   
        
        if i % save_freq == 0:
            ims_opt.append(deepcopy(im.detach().cpu()))
            if viz_online:
                viz.viz_clip(ims_opt[-1])
    
    print('finished')
    return ims_opt, losses


device = 'cuda'
output_weights = torch.zeros(10).to(device)
output_weights[8] = 1
model = model.to(device)
ims_opt_sweep = {}

ims_opt, losses = maximize_im(model, output_weights, 
                              init='zero', num_iters=int(1e3),
                              lambda_pnorm=1e2, lambda_tv=1e-3,
                              device=device, viz_online=False)