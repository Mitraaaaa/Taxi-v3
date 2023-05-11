import numpy as np
import gym

# Initialize the Taxi-v3 environment
env = gym.make("Taxi-v3")

# Initialize the Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

def epsilon_greedy_policy(state, epsilon, stuck_in_loop):
    if np.random.uniform() < epsilon:
        action = env.action_space.sample()
    else:
        if stuck_in_loop :
            action_values = q_table[state, :] + noise_scale * np.random.randn(env.action_space.n)
            action = np.argmax(action_values)
        else :
            action = np.argmax(q_table[state, :])
        
    return action

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.3
noise_scale = 0.1

# Training parameters
num_episodes = 1000
max_steps_per_episode = 200

# Train the Q-learning agent
for episode in range(num_episodes):
    state = env.reset()
    done = False
    stuck_in_loop = 0
    step = 0
    while not done and step < max_steps_per_episode:
        if step > 70:
            stuck_in_loop = 1
            noise_scale = 0.7

        action = epsilon_greedy_policy(state, epsilon,stuck_in_loop)
        stuck_in_loop = 0

        new_state, reward, done, _ = env.step(action)

        #check if we are heading to a wall consequently
        if new_state == state:
            reward = -50

        if reward == -10 :
            reward = -50
        # Update the Q-table using the Q-learning update rule
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
        
        state = new_state
        step += 1


# Test the trained Q-learning agent
num_test_episodes = 10
noise_scale = 0.1
cnt1 = 0 
for episode in range(num_test_episodes):
    state = env.reset()
    done = False
    step = 0
    while not done and step < max_steps_per_episode:
        if step > 70 :
            noise_scale = 0.6

        # action = np.argmax(q_table[state, :])
        action_values = q_table[state, :] + noise_scale * np.random.randn(env.action_space.n)
        action = np.argmax(action_values)

        new_state, _, done, _ = env.step(action)

        # Render the environment
        # env.render()
        state = new_state
        step += 1

    print(f"Episode {episode + 1}: {step} steps")
#     if step == 200:
#         cnt1 += 1
    
# print(cnt1/200)

# Close the environment
env.close()
