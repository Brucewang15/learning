import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")
env.reset()

done = False

while not done:
    action = 2
    new_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render() 

env.close()