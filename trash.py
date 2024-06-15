
import numpy as np
from dm_control.utils import rewards
import matplotlib.pyplot as plt

# create a 1D array of 100 elements from 0 to 1 with a step of 0.01
x = np.arange(0, 2.05, 0.05)
move_reward_list = []

move_reward = rewards.tolerance(x,
                        bounds=(1-0.1, 1+0.1),
                        margin=1/2,
                        value_at_margin=0.45,
                        sigmoid='linear')

# plot the move_reward_list array using matplotlib
plt.figure()
plt.plot(x, move_reward)
plt.show()

action_cost_x = np.arange(0,1,0.01)

action_cost = rewards.tolerance(action_cost_x, bounds=(0.225, 0.425),value_at_margin=0.2, margin=0.30, sigmoid='hyperbolic')


# new figure
plt.figure()
plt.plot(action_cost_x, action_cost)
plt.show()

    