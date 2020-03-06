import gym

from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.deepq.policies import MlpPolicy

# Create environment
env = gym.make('CartPole-v1')
# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
# Train the agent
model.learn(total_timesteps=500000)
# Save the agent
model.save("dqn_cartpole")
# Remove to demonstrate saving and loading
del model

# Load the trained ag
model = DQN.load("dqn_cartpole")

# Evaluate the agent
## TODO: evaluate_policy function has bugs
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
