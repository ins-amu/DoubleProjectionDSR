import numpy as np

import torch
import torch.nn as nn

class ClippedShallowPLRNN(nn.Module):
    """Reimplementation of Clipped shallow PLRNN model from Hess et al., ICML 2023.
    
    No loss function is implemented, as the GTF training is not implemented
    """


    def __init__(self, n_states, n_hidden):
        
        super().__init__()

        self.n_states = n_states

        self.A  = torch.zeros((n_states))
        self.W1 = torch.zeros((n_states, n_hidden))
        self.W2 = torch.zeros((n_states, n_hidden))
        self.h1 = torch.zeros((n_states,))
        self.h2 = torch.zeros((n_hidden,))

        self.OB = torch.zeros((n_states, n_states))
        self.Ob = torch.zeros((n_states,))

    
    @classmethod
    def from_file(cls, filename):

        d = np.load(filename)
        n_states, n_hidden = d['W1'].shape

        for param in ['A', 'W1', 'W2', 'h1', 'h2', 'OB', 'Ob']:
            if np.any(np.isnan(d[param])):
                raise Exception("nan values in the parameters.")

        model = cls(n_states, n_hidden)
        model.A  = torch.tensor(d['A'])
        model.W1 = torch.tensor(d['W1']).T
        model.W2 = torch.tensor(d['W2']).T
        model.h1 = torch.tensor(d['h1'])
        model.h2 = torch.tensor(d['h2'])

        model.OB = torch.tensor(d['OB'].T)
        model.Ob = torch.tensor(d['Ob'])

        model.OBinv = torch.linalg.pinv(model.OB)
       
        return model
    
    def step(self, z):
        W2z = torch.matmul(z, self.W2)
        return self.A * z + torch.matmul(torch.relu(W2z + self.h2) - torch.relu(W2z), self.W1) + self.h1    

    def observation(self, z):
        return torch.matmul(z, self.OB) + self.Ob
    
    def inverse_observation(self, x):
        z = torch.matmul(x - self.Ob, self.OBinv)
        return z

    def sample(self, z0, nt, noise=True, eps=None, return_latent=False, nsamples=1):
        """
        Simulate the model from z0 for nt steps. 

        noise and eps variables are ignored, as the model is deterministic.
        """

        z0 = torch.repeat_interleave(z0, nsamples, dim=0)
        bs = z0.shape[0]

        z = torch.zeros((bs, nt+1, self.n_states))
        z[:,0,:] = z0
        for i in range(nt):
            z[:,i+1,:] = self.step(z[:,i,:])
        x = self.observation(z)

        if return_latent:
            return x, z
        else:
            return x
       
    def get_latent_states(self, x):
        return self.inverse_observation(x[:,:,:])
    
    def get_latent_state_last(self, x, n=1):
        z0 = self.inverse_observation(x[:,-1,:])
        z0 = torch.repeat_interleave(z0, n, dim=0)
        return z0






        
        