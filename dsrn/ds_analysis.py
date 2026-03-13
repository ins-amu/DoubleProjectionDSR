
import numpy as np
import torch



def lyapunov_exponent(model, z0, n_init, n_iter):
    """Calculate maximal Lyapunov coefficient for deterministic part of the model"""

    torch.set_grad_enabled(False)
   
    d0 = np.sqrt(2**-23) # For float32    

    # Initial iterations to get on the attractor
    _, z = model.sample(z0[None,:], n_init+n_iter, noise=False, return_latent=True, nsamples=1)
   
    # Set initial point and perturbed initial point
    z0  = z[:,-1,:]
    v = torch.empty(z0.shape).normal_(0., 1.)
    v /= torch.norm(v, dim=1)    
    z0p = z0 + d0*v

    lambd = []
    
    for i in range(n_iter):    
        # Iterate one step
        dummy_eps = torch.zeros((1, 1, model.n_stoch))
        dummy_x   = torch.zeros((1, 1, model.n_guiding))
        
        z1  = model.f(dummy_eps, dummy_x, z0,  forcing=False)[:,1,:]
        z1p = model.f(dummy_eps, dummy_x, z0p, forcing=False)[:,1,:]        

        # Calculate distances
        d1 = torch.norm(z1p - z1, dim=1)[0]    
        ld = torch.log(torch.abs(d1/d0))

        lambd.append(ld)
    
        # Readjust orbits
        z0 = z1
        v = z1p - z1
        v /= torch.norm(v)
        z0p = z0 + d0*v

    lambd = torch.tensor(lambd).cpu().numpy()
    torch.set_grad_enabled(True)

    return np.mean(lambd)


def calculate_forced_lyap(model, x):
    torch.set_grad_enabled(False)

    bs, nt, nobs = x.shape
        
    # Project 
    xguiding = model.guiding_encoder(x)
    x_and_xguiding = torch.cat((x, xguiding), dim=2)
    eps, _, _ = model.noise_encoder(x_and_xguiding, nsamples=1)
    nteacher = xguiding.shape[2]
    
    # Mask start and end
    mask = torch.ones(nt, dtype=bool)
    mask[0:model.skip[0]]     = False
    mask[nt-model.skip[1]:nt] = False
    
    x = x[:,mask]
    xguiding = xguiding[:,mask]
    x_and_xguiding = x_and_xguiding[:,mask]
    eps = eps[:,mask]

    ntm = x.shape[1]

    # Initial point
    z0 = torch.cat([xguiding[:,0,:], model.init(x_and_xguiding[:,0,:])], dim=1)

    # Initial perturbation
    d0 = np.sqrt(2**-23)   # For float32
    v = torch.empty(z0.shape).normal_(0., 1.)
    v /= torch.norm(v, dim=1)[:,None]
    z0p = z0 + d0*v

    lambd = []
    zs  = []
    zps = []
    
    for i in range(ntm):
        dummy_x   = torch.zeros((1, 1, model.n_guiding))
       
        z1  = model.f(eps[:,[i]], dummy_x, z0,  forcing=False)[:,1,:]
        z1p = model.f(eps[:,[i]], dummy_x, z0p, forcing=False)[:,1,:]

        # Calculate distances
        d1 = torch.norm(z1p - z1, dim=1)
        ld = torch.log(torch.abs(d1/d0))
        lambd.append(ld.cpu().numpy())
        
        zs.append(z1.cpu().numpy())
        zps.append(z1p.cpu().numpy())
    
        # TF and readjust orbits
        if i < ntm-1:
            v = z1p - z1
            v /= torch.norm(v, dim=1)[:,None]
            z0  = torch.cat([xguiding[:,i+1,:], z1[:,nteacher:]], dim=1)
            z0p = z0 + d0*v

    lambd = np.array(lambd).T
    zs  = np.array(zs)
    zps = np.array(zps)

    torch.set_grad_enabled(True)

    return np.mean(lambd, axis=1)


def find_attractors(model, npoints, nwarmup, nt, nlyap, x=None, tolfp=1e-5, tollc=1e-2, tolca=1e-2, to_numpy=True):

    # Initial conditions
    if x == None:
        # Uniform from the whole state space
        z0 = 2*torch.rand(npoints, model.n_states) - 1.
    else:
        z = model.get_latent_state_init(x)[0,:,:]
        inds = np.random.choice(np.r_[:x.shape[1]], size=npoints, replace=False)
        z0 = z[inds,:]

    # Simulate
    _, z = model.sample(z0, nwarmup+nt, noise=False, return_latent=True, nsamples=1)
    z = z[:, nwarmup+1:,:]

    attractors = []
    for i in range(0,npoints):
        maxerr = 10000.
        is_new = True

        for j, (attrtype, lmax, n, za) in enumerate(attractors):
            if attrtype == 'ca':
                dist = torch.norm(za[:,None,:] - z[i,None,:,:], dim=2)
                err1 = torch.quantile(dist.min(dim=0).values, 0.8)
                err2 = torch.quantile(dist.min(dim=1).values, 0.8)
                maxerr = min(max(err1, err2), maxerr)
            else:
                # Same as above, actually. For now? Quantiles might be set differently for LCs
                err1 = torch.quantile(torch.norm(za[:,None,:] - z[i,None,:,:], dim=2).min(dim=0).values, 0.8)
                err2 = torch.quantile(torch.norm(za[:,None,:] - z[i,None,:,:], dim=2).min(dim=1).values, 0.8)
                maxerr = min(max(err1, err2), maxerr)

            tol = {'fp': tolfp, 'lc': tollc, 'ca': tolca}[attrtype]
            if maxerr < tol:
                is_new = False
                attractors[j] = (attrtype, lmax, n+1, za)
                break

        if is_new:
            # Calculate Lyapunov exponent
            lmax = lyapunov_exponent(model, z[i,-1,:], n_init=0, n_iter=nlyap)

            if lmax > 0:
                attrtype = 'ca'
            elif torch.max(torch.norm(z[i,-100:,:] - z[i,-1,:], dim=1)) < tolfp:
                attrtype = 'fp'
            else:
                attrtype = 'lc'
            
            attractors.append((attrtype, lmax, 1, z[i].detach()))

    attractors = sorted(attractors, key=lambda x: x[2], reverse=True)
            
    if to_numpy:
        return [(t, l, n, z.cpu().numpy()) for (t,l,n,z) in attractors]
    else:
        return attractors      