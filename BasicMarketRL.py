import gymnasium as gym
import torch
from stable_baselines3 import PPO
from envs.static_market_env import StaticMarketEnv
from stable_baselines3.common.env_util import make_vec_env

# Set up
env = make_vec_env(StaticMarketEnv)
print(torch.cuda.is_available())
model = PPO("MlpPolicy", env, verbose=1, device='cuda')
# Train
model.learn(10000)

# Test it
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()