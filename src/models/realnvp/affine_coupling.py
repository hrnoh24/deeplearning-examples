import torch
import torch.nn as nn
import torch.nn.init as init

class AffineCoupling(nn.Module):
    def __init__(self, mask, hidden_dim):
        super(AffineCoupling, self).__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim

        ## mask to seperate positions that do not change and positions that change.
        ## mask[i] = 1 means the i-th position does not change.
        self.mask = nn.Parameter(mask, requires_grad=False)

        ## layers used to compute scale in affine transformation
        self.scale_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.scale_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_fc3 = nn.Linear(self.hidden_dim, self.input_dim)
        self.scale = nn.Parameter(torch.Tensor(self.input_dim))
        init.normal_(self.scale)

        ## layers used to compute translation in affine transformation
        self.translation_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.translation_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.translation_fc3 = nn.Linear(self.hidden_dim, self.input_dim)

    def _compute_scale(self, x):
        ## compute scaling factor using unchanged part of x with a neural network
        s = torch.relu(self.scale_fc1(x * self.mask))
        s = torch.relu(self.scale_fc2(s))
        s = torch.relu(self.scale_fc3(s)) * self.scale
        return s
    
    def _compute_translation(self, x):
        ## compute translation factor using unchanged part of x with a neural network
        t = torch.relu(self.translation_fc1(x * self.mask))
        t = torch.relu(self.translation_fc2(t))
        t = self.translation_fc3(t)
        return t
    
    def forward(self, x):
        ## convert latent space variable to observed variable
        s = self._compute_scale(x)
        t = self._compute_translation(x)

        y = self.mask*x + (1-self.mask)*(x*torch.exp(s) + t)
        logdet = torch.sum((1 - self.mask)*s, -1)

        return y, logdet
    
    def inverse(self, y):
        ## convert observed variable to latent space variable
        s = self._compute_scale(y)
        t = self._compute_translation(y)

        x = self.mask*y + (1-self.mask)*((y-t)*torch.exp(-s))
        logdet = torch.sum((1 - self.mask)*(-s), -1)

        return x, logdet