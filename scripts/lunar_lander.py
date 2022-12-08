import gym

from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv


# Create the environment
env = make_vec_env('LunarLander-v2', n_envs=16)

# PPO Model (agent)
model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1
)

# Train it
model.learn(total_timesteps=1000000)

# Save the model
model_name = "ppo-LunarLander-v2-baseline"
model.save(model_name)

print("Evaluating...")
eval_env = gym.make("LunarLander-v2")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


# Push to HF Hub
## repo_id is the id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
repo_id = "ayut/ppo-LunarLander-v2"
env_id = "LunarLander-v2"

eval_env = DummyVecEnv([lambda: gym.make(env_id)])

model_architecture = "PPO"
commit_message = "baseline PPO agent for LunarLander-v2"

# method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
package_to_hub(model=model, # Our trained model
               model_name=model_name, # The name of our trained model 
               model_architecture=model_architecture, # The model architecture we used: in our case PPO
               env_id=env_id, # Name of the environment
               eval_env=eval_env, # Evaluation Environment
               repo_id=repo_id, # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
               commit_message=commit_message)
