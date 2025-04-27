import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()


DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
WIN_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_PER = 2500

epsilon = 0.5
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
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])  
        else:
            action = np.random.randint(0, env.action_space.n)  

        new_state, reward, terminated, truncated, info = env.step(action)

        new_discrete_state = get_discrete_state(new_state)
        
        done = terminated or truncated

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
env.close()