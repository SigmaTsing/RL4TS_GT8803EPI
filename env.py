import gymnasium as gym
from gymnasium import spaces
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.nn import functional as F
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from tqdm import tqdm

class TSFEnv(gym.Env):
    def __init__(self, data_path, input_length = 512, output_length = 128, step_size = 32):
        print('Building Env')
        self.input_length = input_length
        self.output_length = output_length
        
        self.action_space = spaces.Box(low = -10, high = 10, shape = (output_length,), dtype = np.float32)
        self.observation_space = spaces.Box(low = -10, high = 10, shape = (input_length,), dtype = np.float32)
        
        print('Data Readin')
        # raw = pd.read_csv(os.path.join(data_path, 'illness', 'national_illness.csv'))
        raw = pd.read_csv(os.path.join(data_path, 'weather', 'weather.csv'))
        
        print('Preprocessing')
        self.raw = MinMaxScaler((-1, 1)).fit_transform(raw['OT'].to_numpy().reshape((-1, 1))).squeeze()
        self.train_span = int(0.8*len(self.raw))
        self.test_span = len(self.raw) - self.train_span
        self.current_step = 0
        self.current_epoch = 0
        self.step_size = step_size
        # self.input_data = np.lib.stride_tricks.sliding_window_view(raw, window_shape= input_length)[:-self.output_length]
        # self.output_data = np.lib.stride_tricks.sliding_window_view(raw, window_shape= output_length)[self.input_length:]
        # print('Input shape {} Output shape {}'.format(self.input_data.shape, self.output_data.shape))

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        self.current_step = np.random.randint(0, self.train_span - self.input_length - self.output_length)
        self.current_epoch = 0
        obs = self.raw[self.current_step:self.current_step + self.input_length].flatten()
        return obs, {}
    
    def step(self, action):        
        true_future = self.raw[self.current_step + self.input_length : self.current_step + self.input_length + self.output_length].flatten()
        reward = -F.mse_loss(torch.tensor(action), torch.tensor(true_future)).item()
        self.current_epoch += 1
        self.current_step += self.step_size
        terminated = ((self.current_step >= self.train_span - self.input_length - self.output_length) | (self.current_epoch > 10))
        
        next_obs = self.raw[self.current_step:self.current_step + self.input_length].flatten()
        
        return next_obs, reward, terminated, False, {}
    
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_length=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, input_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * (-np.log(10000.0) / input_dim))

        if input_dim % 2 == 0:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

        if input_dim % 2 == 1:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(-2), :]
        return x

class TransformerAgent(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=128, action_dim = 128, value_dim = 32, num_layers=2, num_heads=1, dropout=0.25):
        super().__init__()
        
        self.latent_dim_pi = action_dim
        self.latent_dim_vf = value_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # self.embedding = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,batch_first=True, dropout=dropout)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        #make meta data shape = number of heads so that adding meta data will equal the dimensions
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, \
            nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        
        self.linear = nn.Linear(in_features = hidden_dim, out_features = value_dim)
        self.policy_proj = nn.Linear(in_features = value_dim, out_features = self.latent_dim_pi)
        # self.value_proj = nn.Linear(in_features = mlp_dim, out_features = self.latent_dim_vf)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_actor(self, features):
        x = F.gelu(self.embedding(features))
        # x = self.positional_encoding(x)
        x = self.encoder(x)
        # print(x.shape, self.hidden_dim)
        x = self.linear(F.gelu(torch.mean(x, dim = -2)))
        action = self.policy_proj(F.gelu(x))
        return action
    
    def forward_critic(self, features):
        x = F.gelu(self.embedding(features))
        # x = self.positional_encoding(x)
        x = self.encoder(x)
        # print(x.shape, self.hidden_dim)
        # print(x.shape)
        x = self.linear(F.gelu(torch.mean(x, dim = -2)))
        # print(x.shape, self.latent_dim_vf)
        return x
        
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = TransformerAgent(input_dim= self.features_dim, action_dim=self.action_space.shape[0])

def train_and_evaluate_custom_ppo():
    # Create the environment
    env = TSFEnv('~/Time-LLM/dataset/', input_length= 36, output_length=12, step_size=4)
    vec_env = DummyVecEnv([lambda: env])
    
    # Initialize PPO model with custom policy
    model = PPO(
        CustomActorCriticPolicy, 
        vec_env, 
        verbose=1, 
        learning_rate=0.001, 
        n_steps=2048, 
        batch_size=64, 
        n_epochs=10
    )
    
    # Train the model
    for i in range(10):
        model.learn(total_timesteps=10000, progress_bar= True)
        
        # Evaluate the model
        
        losses = 0
        print('Evaluating')
        for i in tqdm(range(env.train_span, len(env.raw) - env.input_length - env.output_length)):
            obs = env.raw[i:i+env.input_length]
            ground = env.raw[i+env.input_length:i+env.input_length+env.output_length]
            action, _ = model.predict(obs, deterministic=True)
            losses += F.mse_loss(torch.tensor(action), torch.tensor(ground)).item()
        print(losses / (len(env.raw) - env.input_length - env.output_length - env.train_span))
    
    return model, env

if __name__ == '__main__':
    model, env = train_and_evaluate_custom_ppo()
    # print(env.reset())
    
    # for i in range(20):
    #     outp = np.random.uniform(-1, 1, env.action_space.shape)
    #     print(env.step(outp)[1:]) 
    