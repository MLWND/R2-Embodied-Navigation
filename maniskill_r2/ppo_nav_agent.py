"""PPO 强化学习导航避障策略网络 (PPO Actor-Critic Network)"""
import torch
import torch.nn as nn
from torch.distributions import Normal

class PPONavAgent(nn.Module):
    def __init__(self, obs_dim=52, act_dim=2):
        super().__init__()
        
        # Actor 网络
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim) - 0.5)

        # Critic 网络
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            action = torch.clamp(action, -1.0, 1.0)
            
        log_prob = probs.log_prob(action).sum(axis=-1)
        entropy = probs.entropy().sum(axis=-1)
        value = self.critic(x)
        return action, log_prob, entropy, value

    def act_deterministic(self, x):
        with torch.no_grad():
            action_mean = self.actor(x)
            return torch.clamp(action_mean, -1.0, 1.0)
