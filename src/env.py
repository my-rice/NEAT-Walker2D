import gymnasium as gym
import numpy as np
import time
import pygame
from agent import PIDController

env = gym.make('Pendulum-v1', g=9.81, render_mode="human")

# Then we reset this environment
observation = env.reset()
# Observation and action space 
# select render mode to human
env.render()

observed_space = env.observation_space
action_space = env.action_space
print("The observation space:{}" .format(observed_space))
print("The action space: {}".format(action_space))

# Perform 10 random actions
while True:
  # Take a random action
  action = env.action_space.sample()
  print("Action taken:", action)

  # Do this action in the environment and get
  # next_state, reward, done and info
  obs, reward, terminated, truncated, info = env.step(action)
  # If the game is done (in our case we land, crashed or timeout)
  if terminated or truncated:
      # Reset the environment
      print("Environment is reset")
      observation = env.reset()

# pygame.init()
# DISPLAYSURF = pygame.display.set_mode((450, 450))
# clock = pygame.time.Clock()
# pygame.display.set_caption('Hello World!')
# pygame.display.flip()

# env = gym.make('Pendulum-v1', g=9.81,render_mode="rgb_array")

# dt = env.unwrapped.dt # Time step of the simulation

# Choose appropriate PID gains (experimentation is key)
# controller = PIDController(kp=0.05, ki=0.03, kd=0.02)

# state = env.reset()
# image = env.render()

# Convert image to pygame surface
# image = pygame.surfarray.make_surface(image)
# DISPLAYSURF.blit(image, (0, 0))
# pygame.display.update()


# print("Initial state:", state)
# print("sin(theta):", state[0][0], "cos(theta):", state[0][1], "theta_dot:", state[0][2])
# terminated = truncated = False

# id = 0
# while not terminated or not truncated:
#   Get control action based on current state
#   action = controller.control(state, dt)

#   Take action and observe next state, reward, etc.
#   next_state, reward,terminated, truncated, info = env.step(action)
#   state = next_state

#   time.sleep(0.01)
#   image = env.render()
#   Convert image to pygame surface
#   image = pygame.surfarray.make_surface(image)
#   DISPLAYSURF.blit(image, (0, 0))
#   pygame.display.update()

#   id += 1

# env.close()