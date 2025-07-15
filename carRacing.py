import os
import gymnasium as gym
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
import cv2


log_path = os.path.join('Training', 'Logs')
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_Model_carRacing')
def make_env():
    return gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

def train_model():
    env = make_vec_env(make_env, n_envs=1)
    env = VecTransposeImage(env)
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=5000)
    model.save(PPO_path)

def evaluate_model():
    env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
    model = PPO.load(PPO_path, env=env)
    
    episodes = 5
    for ep in range(episodes):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated = env.step(action)
            done = terminated or truncated
            score += reward.item()
        print(f"Episode {ep+1} Score: {score}")

def record_video():
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False
    )
    model = PPO.load(PPO_path, env=env)

    obs, _ = env.reset()
    done = False
    frames = []
    score = 0

    while not done:
        frame = env.render()
        frames.append(frame)

        cv2.imshow("CarRacing", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        score += reward

    print(f"Episode Score: {score}")


    cv2.destroyAllWindows()
    
    video_path = "carracing_episode.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Saved video to {video_path}")
    
    
    
def only_train():
    env = make_vec_env(make_env, n_envs=1)
    env = VecTransposeImage(env)
    model = PPO.load(PPO_path, env=env)
    model.learn(total_timesteps=30000)
    model.save(PPO_path)

if __name__=='__main__':
    #train_model()
    #only_train()
    record_video()