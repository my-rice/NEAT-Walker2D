
import numpy as np
from dm_control.utils import rewards
import matplotlib.pyplot as plt

# create a 1D array of 100 elements from 0 to 1 with a step of 0.01
x = np.arange(0, 1, 0.01)
move_reward_list = []
alternate_legs = rewards.tolerance(x,bounds=(0.45,0.55),margin=0.2,value_at_margin=0.01, sigmoid="gaussian")

# move_reward = rewards.tolerance(x,
#                         bounds=(1-0.1, 1+0.1),
#                         margin=1/2,
#                         value_at_margin=0.0,
#                         sigmoid='linear')

# # plot the move_reward_list array using matplotlib
# plt.figure()
# plt.plot(x, move_reward)
# plt.show()

# action_cost_x = np.arange(0,1,0.01)

# action_cost = rewards.tolerance(action_cost_x, bounds=(0.225, 0.425),value_at_margin=0.2, margin=0.30, sigmoid='hyperbolic')


# # new figure
# plt.figure()
# plt.plot(action_cost_x, action_cost)
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# file_path = '/home/davide/nc_project/legged_locomotion/file.csv'  # Sostituisci con il percorso del tuo file CSV
# data = pd.read_csv(file_path)
# #print(data)
# colonna1 = data['left']
# colonna2 = data['right']
# #print(colonna1)
# plt.figure()

# plt.plot(colonna1, label='left', color='b')

# plt.plot(colonna2, label='right', color='r')

# plt.title('Confronto tra Colonna1 e Colonna2')
# plt.xlabel('Indice')
# plt.ylabel('Valori')
# plt.legend()
# plt.grid(True)
# plt.show()


# angle_left_thigh_list = np.arange(0.26, -0.27, -0.01)
# # concatenate the two arrays

# angle_left_thigh_list = np.concatenate((angle_left_thigh_list, angle_left_thigh_list[::-1]))
# angle_right_thigh_list = np.arange(-0.26, 0.27, 0.01)
# angle_right_thigh_list = np.concatenate((angle_right_thigh_list, angle_right_thigh_list[::-1]))

# left_dominant = False
# last_left_thigh_angle = 0.261
# last_right_thigh_angle = -0.261
# rewards = []
# for i in range(len(angle_left_thigh_list)):
#     angle_left_thigh = angle_left_thigh_list[i]
#     angle_right_thigh = angle_right_thigh_list[i]
#     if left_dominant:
#         if np.abs(angle_left_thigh-angle_right_thigh) >= (0.52) and angle_left_thigh > angle_right_thigh:
#             print("left dominant: ", left_dominant)
#             left_dominant = False
#             left_reward = 1
#             right_reward = 1
#         else:
#             if angle_left_thigh >= last_left_thigh_angle:
#                 left_reward = 1
#             else:
#                 left_reward = 0
#             if angle_right_thigh <= last_right_thigh_angle:
#                 right_reward = 1
#             else:
#                 right_reward = 0
#     else:
#         if np.abs(angle_left_thigh-angle_right_thigh) >= (0.52) and angle_right_thigh > angle_left_thigh:
#             print("left dominant: ", left_dominant)
#             left_dominant = True   
#             left_reward = 1
#             right_reward = 1
#         else:
#             if angle_left_thigh <= last_left_thigh_angle:
#                 left_reward = 1
#             else:
#                 left_reward = 0
#             if angle_right_thigh >= last_right_thigh_angle:
#                 right_reward = 1
#             else:
#                 right_reward = 0
    

#     last_left_thigh_angle = angle_left_thigh
#     last_right_thigh_angle = angle_right_thigh
#     rewards.append(left_reward + right_reward)


plt.figure()
plt.plot(alternate_legs)
plt.show()