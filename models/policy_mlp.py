import torch
import torch.nn as nn

class policy_mlp(nn.Module):
    def __init__(self, obs_dim, act_dim, goal_dim, views, obs_window_size, act_window_size, hidden_dim, layers):
        super(policy_mlp, self).__init__()
        self.obs_window_size = obs_window_size
        self.act_window_size = act_window_size
        
        input_dim = (obs_dim * views) + (goal_dim * views)
        output_dim = act_dim * act_window_size
        
        layers = [nn.Linear(input_dim * obs_window_size, hidden_dim), nn.GELU()]
        
        for i in layers:
            layers.append(nn.Linear(input_dim * obs_window_size, hidden_dim))
            layers.append(nn.GELU())
            
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(layers)
                    
    
        
    def forward(self, obs_seq, goal_seq=None):
        if goal_seq is not None:
            x = torch.cat([obs_seq, goal_seq], dim=-1)
        else:
            x = obs_seq
        
        x = x.view(x.size(0), -1) 
        
        x = self.net(x)
        
        return x.view(x.size(0), -1, self.obs_window_size, self.act_window_size)