import torch
import torch.nn as nn

from src.models.realnvp.affine_coupling import AffineCoupling

class RealNVP_2D(nn.Module):
    '''
    A vanilla RealNVP class for modeling 2 dimensional distributions
    '''
    def __init__(self, masks, hidden_dim):
        '''
        initialized with a list of masks. each mask define an affine coupling layer
        '''
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask = nn.ParameterList(
            [nn.Parameter(torch.Tensor(m), requires_grad=False) for m in masks]
        )

        self.affine_couplings = nn.Modulelist(
            [AffineCoupling(self.masks[i], self.hidden_dim) for i in range(len(self.masks))]
        )

    def forward(self, x):
        ## convert latent space variable into observed variables
        y = x
        logdet_tot = 0
        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y)
            logdet_tot += logdet

        ## a normalization layer is added such that the observed variables is within
        ## the range [-4, 4]
        logdet = torch.sum(torch.log(torch.abs(4*(1-(torch.tanh(y))**2))), -1)        
        y = 4*torch.tanh(y)
        logdet_tot = logdet_tot + logdet

        return y, logdet_tot
    
    def inverse(self, y):
        ## convert observed variables into latent space variables
        x = y
        logdet_tot = 0

        # inverse the normalization layer
        logdet = torch.sum(torch.log(torch.abs(1.0/4.0* 1/(1-(x/4)**2))), -1)
        x  = 0.5*torch.log((1+x/4)/(1-x/4))
        logdet_tot = logdet_tot + logdet

        ## inverse affine coupling layers
        for i in range(len(self.affine_couplings)-1, -1, -1):
            x, logdet = self.affine_couplings[i].inverse(x)
            logdet_tot = logdet_tot + logdet
            
        return x, logdet_tot