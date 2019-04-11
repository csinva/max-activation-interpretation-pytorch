import torch
import regularization

# maximize im specified by pattern
# return list of optimized images along with loss for each image
def maximize_im(model, im, num_iters=int(1e3), lr=1e-2, 
                lambda_pnorm=0, lambda_tv=0,
                device='cuda', save_freq=50):

    opt = torch.optim.SGD({im}, lr=lr) #, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    losses = np.zeros(num_iters + 1)
    ims_opt = []
    for i in (np.arange(num_iters) + 1):
        model.zero_grad()
        pred = model(im).squeeze() # forward pass
        
        # what we want to minimize
        loss = -1 * pred.norm() # how large output is
        loss = loss + lambda_pnorm * im.norm(p=6) # p-norm (keeps any pixel from having too large a value)
        loss = loss + lambda_tv * regularization.tv_norm(im)[0] # tv regularization
        
        # optimize
        loss.backward(retain_graph=True)
        opt.step()   
        
        # saving
        losses[i] = loss.detach().item()
        if i % save_freq == 0:
            ims_opt.append(deepcopy(im.detach().cpu()))

    return ims_opt, losses


device = 'cuda'
model = model.to(device)
im = torch.zeros(1, 3, 64, 64, requires_grad=True, device=device)
ims_opt, losses = maximize_im(model, im)