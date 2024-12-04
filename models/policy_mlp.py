import torch
import torch.nn as nn
import einops


def repeat_start_to_length(x: torch.Tensor, length: int, dim: int = 0):
    """
    Pad tensor x to length along dim, repeating the first value at the start.
    """
    pad_size = length - x.shape[dim]
    if pad_size <= 0:
        return x
    first_frame = x.index_select(dim, torch.tensor(0, device=x.device))
    repeat_shape = [1] * len(x.shape)
    repeat_shape[dim] = pad_size
    pad = first_frame.repeat(*repeat_shape)
    return torch.cat([pad, x], dim=dim)

class policy_mlp(nn.Module):
    def __init__(self, obs_dim, act_dim, goal_dim, views, obs_window_size, act_window_size, hidden_dim, layers):
        super().__init__()
                
        self.obs_window_size = obs_window_size
        self.act_window_size = act_window_size
        
        self.input_dim = (obs_dim * views) + (goal_dim * views)
        self.output_dim = act_dim * act_window_size
        
 
        
        network_layers = [nn.Linear(self.input_dim * obs_window_size, hidden_dim), nn.GELU()]
        
        for i in range(layers):

            network_layers.append(nn.Linear(hidden_dim, hidden_dim))
            network_layers.append(nn.GELU())
        
        network_layers.append(nn.Linear(hidden_dim, self.output_dim))
        
        self.net = nn.Sequential(*network_layers)
    
    
    
    def forward(self, obs_seq):
        
        if obs_seq.shape[1] < self.obs_window_size:
            obs_seq = repeat_start_to_length(obs_seq, self.obs_window_size, dim=1)
        
        x = obs_seq
        x = x[:, :, :self.input_dim]   

        
        x = x.flatten(start_dim=1)
        x = self.net(x)

        x = x.view(x.shape[0], -1, 2)

        
        return x
    