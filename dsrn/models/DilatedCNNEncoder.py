import torch
import torch.nn as nn


from .SLSTM import SLSTM
from .MaskedConvolution import MaskedConvolution


class GatedTCBlock(nn.Module):
    def __init__(self, n_in, n_out, k, d, causal=0):
        super().__init__()

        conv1 = torch.nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=k, dilation=d, padding='same')
        conv2 = torch.nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=k, dilation=d, padding='same')

        if causal == 0:
            self.conv1 = conv1
            self.conv2 = conv2

        else:
            if causal == 1:
                mask = torch.ones(k)
                mask[k//2+1:k] = 0.
            elif causal == -1:
                mask = torch.ones(k)
                mask[0:k//2] = 0.
            else:
                raise ValueError("Unexpected value for 'causal', allowed are {-1,0,1}")
            self.conv1 = MaskedConvolution(conv1, mask)
            self.conv2 = MaskedConvolution(conv2, mask)

        if n_in == n_out:
            self.resize = None
        else:
            self.resize = torch.nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, padding=0)

    def __call__(self, x):
        xa = self.conv1(x)
        xb = self.conv2(x)

        xa = torch.sigmoid(xa)
        xb = torch.tanh(xb)

        xn = xa * xb

        res = self.resize(x) if self.resize is not None else x
        out = xn + res

        return out


class DilatedCNNEncoder(nn.Module):
    #  Inspired by WaveNet (van den Oord et al, 2016)

    def __init__(self, n_in, n_out, n_channels, kernel_size, n_levels=None, output='point',
                 rnn_size=32, initializer='standard', c_init=0.01, causal=0):
        super().__init__()

        self.output = output
        self.n_in = n_in
        self.n_out = n_out

        self._ismasked = (causal != 0)
        self.causal = causal

        try:
            n_levels = len(n_channels)
        except TypeError:
            n_channels = n_levels * [n_channels]

        layers  = []

        for i in range(n_levels):
            dilation = 2**i
            in_channels = n_in if i == 0 else n_channels[i-1]
            out_channels = n_channels[i]
            layers  += [GatedTCBlock(in_channels, out_channels, k=kernel_size, d=dilation, causal=causal)]

        self.net  = nn.Sequential(*layers)

        if output == 'point':
            self.proj = nn.Linear(out_channels, n_out, bias=True)
        elif output == 'normal':
            self.proj = nn.Linear(out_channels, 2*n_out)
        elif output == 'ar':
            self.ar_hidden_size = rnn_size
            self.ar = SLSTM(input_size=out_channels, hidden_size=self.ar_hidden_size, output_size=n_out)
        else:
            raise ValueError(f"Unknown output '{output}'")
        
        if initializer == 'timedelay':
            self.init_timedelay(c_init)
        elif initializer == 'standard':
            pass
        else:
            raise ValueError(f"Unknown initializer {initializer}")


    def init_timedelay(self, c=0.):
        n_obs = self.n_in
        
        for i, layer in enumerate(self.net):
            if not self._ismasked:
                conv1 = layer.conv1
                conv2 = layer.conv2
            else:
                conv1 = layer.conv1._conv_layer
                conv2 = layer.conv2._conv_layer

            n_in  = conv1.in_channels
            n_out = conv1.out_channels
            ks    = conv1.kernel_size[0]            

            if layer.resize is not None:                
                kr = c*torch.sqrt(1./(torch.tensor(n_in)*1))
                w = torch.distributions.Uniform(-kr, kr).sample((n_out, n_in, 1))
                b = torch.distributions.Uniform(-kr, kr).sample((n_out,))
                n = min(n_in, n_out, 2*i*n_obs+1)
                w[0:n,0:n,:] += torch.eye(n)[:,:,None]
                layer.resize.weight = torch.nn.Parameter(w)                
                layer.resize.bias = torch.nn.Parameter(b)
                
            kr = c*torch.sqrt(1./(torch.tensor(n_in*ks)))
            w1 = torch.distributions.Uniform(-kr, kr).sample((n_out, n_in, ks))
            w2 = torch.distributions.Uniform(-kr, kr).sample((n_out, n_in, ks))
            b1 = torch.distributions.Uniform(-kr, kr).sample((n_out,))
            b2 = torch.distributions.Uniform(-kr, kr).sample((n_out,))
       
            if self.causal == 0:
                for j in range(n_obs):
                    if 2*i*n_obs+1+j < n_out:
                        b1[2*i*n_obs+1+j] += 1.
                        w2[2*i*n_obs+1+j,j,-1] += 1.
                    if 2*i*n_obs+1+n_obs+j < n_out:
                        b1[2*i*n_obs+1+n_obs+j] += 1.
                        w2[2*i*n_obs+1+n_obs+j,j,0] += 1.
            elif self.causal == 1:
                for j in range(n_obs):
                    if i*n_obs+1+j < n_out:
                        b1[i*n_obs+1+j] += 1.
                        w2[i*n_obs+1+j,j,0] += 1.
            elif self.causal == -1:
                for j in range(n_obs):
                    if i*n_obs+1+j < n_out:
                        b1[i*n_obs+1+j] += 1.
                        w2[i*n_obs+1+j,j,-1] += 1.
            else:
                raise ValueError("causal can be only {{-1,0,1}}, is '{causal}'.")

            with torch.no_grad():
                if not self._ismasked:
                    conv1.weight[:,:,:] = w1
                    conv2.weight[:,:,:] = w2    
                else:
                    conv1.weight[:,:,:] = layer.conv1.mask * w1
                    conv2.weight[:,:,:] = layer.conv2.mask * w2
                conv1.bias[:] = b1
                conv2.bias[:] = b2
       

        if self.output == 'point':            
            n_out, n_in = self.proj.weight.shape
            kr = c*torch.sqrt(1./torch.tensor(n_in))
            n = min(n_in, n_out)
            w = torch.distributions.Uniform(-kr, kr).sample((n_out, n_in))
            b = torch.distributions.Uniform(-kr, kr).sample((n_out,))
            w[0:n,0:n] += torch.eye(n)
            with torch.no_grad():
                self.proj.weight[:,:] = w
                self.proj.bias[:] = b

        elif self.output == 'normal':
            n_out, n_in = self.proj.weight.shape
            kr = c*torch.sqrt(1./torch.tensor(n_in))
            n_out_units = n_out // 2
            n = min(n_in, n_out_units)
            w = torch.distributions.Uniform(-kr, kr).sample((n_out, n_in))
            b = torch.distributions.Uniform(-kr, kr).sample((n_out,))
            w[0:n,0:n] += torch.eye(n)
            with torch.no_grad():
                self.proj.weight[:,:] = w
                self.proj.bias[:] = b

        else:
            raise NotImplementedError(f"Time delay initialization not implemented for output='{self.output}'.")


    def dummy_call(self, x, nsamples):
        bs, nt, nobs = x.shape

        if self.output == 'point':
            return torch.zeros((bs*nsamples, nt, 0))

        elif self.output in ['normal', 'ar']:
            dummy_sample = torch.zeros((bs*nsamples, nt, 0))
            dummy_mu = torch.zeros((bs*nsamples, nt, 0))
            dummy_lv = torch.zeros((bs*nsamples, nt, 0))
            return dummy_sample, dummy_mu, dummy_lv

        else:
            return None


    def __call__(self, x, nsamples: int=1, return_states=False):
        if self.n_out == 0 and not return_states:
            return self.dummy_call(x, nsamples)

        bs, nt, _ = x.shape
        bs = bs*nsamples

        x = torch.transpose(x, 1, 2)
        outs = self.net(x)

        outs = torch.transpose(outs, 1, 2)
        outs = torch.repeat_interleave(outs, nsamples, dim=0)

        if self.output == 'point':
            if not return_states:
                return self.proj(outs)
            else:
                return self.proj(outs), outs
            
        elif self.output == 'normal':
            mu, lv = torch.split(self.proj(outs), [self.n_out,self.n_out], dim=2)
            eps = torch.randn_like(mu).mul(torch.exp(0.5 * lv)).add_(mu)

            if not return_states:
                return eps, mu, lv
            else:
                return eps, mu, lv, outs

        elif self.output == 'ar':
            h0 = torch.zeros((bs, self.ar_hidden_size))
            c0 = torch.zeros((bs, self.ar_hidden_size))
            y0 = torch.zeros((bs, self.n_out))
            eps, mu, lv = self.ar(outs, h0, c0, y0)
            if not return_states:
                return eps, mu, lv
            else:
                return eps, mu, lv, outs

        else:
            return None

