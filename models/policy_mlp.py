import torch
import torch.nn as nn
import einops

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
    
    
    
    def forward(self, x):
        
        x = x[:, :, :self.input_dim]   

        
        x = x.flatten(start_dim=1)
        x = self.net(x)

        x = x.view(x.shape[0], -1, 2)

        
        return x
    