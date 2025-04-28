import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
env.reset()


DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
WIN_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

episode_rewards = []
aggregated_episode_rewards = {'min': [], 'max': [], 'ep': [], 'avg': []}

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
SHOW_PER = 500

epsilon = 0.75
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAY - START_EPSILON_DECAY)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / WIN_SIZE
    return tuple(discrete_state.astype(int))


for episode in range(EPISODES):

    if episode % SHOW_PER == 0:
        print(episode)
        env.close()
        env = gym.make("MountainCar-v0", render_mode="human")
    else:
        env.close()
        env = gym.make("MountainCar-v0", render_mode=None)

    state, info = env.reset()
    discrete_state = get_discrete_state(state)
    
    done = False
    episode_reward = 0

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])  
        else:
            action = np.random.randint(0, env.action_space.n)  

        new_state, reward, terminated, truncated, info = env.step(action)

        new_discrete_state = get_discrete_state(new_state)
        
        done = terminated or truncated

        episode_reward += reward

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            q_table[discrete_state + (action, )] = new_q
        
        elif new_state[0] >= env.unwrapped.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0
        
        discrete_state = new_discrete_state

    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
        epsilon -= epsilon_decay_value

    episode_rewards.append(episode_reward)
    if episode % SHOW_PER == 0:
        average_reward = sum(episode_rewards[-SHOW_PER:]) / len(episode_rewards[-SHOW_PER:])
        aggregated_episode_rewards['ep'].append(episode)
        aggregated_episode_rewards["avg"].append(average_reward)
        aggregated_episode_rewards['min'].append(min(episode_rewards[-SHOW_PER:]))
        aggregated_episode_rewards['max'].append(max(episode_rewards[-SHOW_PER:]))

env.close()

plt.plot(aggregated_episode_rewards['ep'], aggregated_episode_rewards['avg'], label="avg")
plt.plot(aggregated_episode_rewards['ep'], aggregated_episode_rewards['min'], label="min")
plt.plot(aggregated_episode_rewards['ep'], aggregated_episode_rewards['max'], label="max")
plt.legend()
plt.show()