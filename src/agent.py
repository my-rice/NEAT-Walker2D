import numpy as np

class PIDController:
  def __init__(self, kp, ki, kd):
    self.kp = kp  # Proportional gain
    self.ki = ki  # Integral gain
    self.kd = kd  # Derivative gain
    self.integral_error = 0  # Integral term accumulator

  def control(self, state, dt):

    # Extract relevant state variables
    if isinstance(state[0], np.ndarray):
        state_array = state[0]
        sin_theta = state_array[0]  # Angle of the pendulum
        cos_theta = state_array[1]  # Cosine of the pendulum angle
        theta_dot = state_array[2]  # Angular velocity of the pendulum
    elif isinstance(state[0], np.float32):
        sin_theta = state[0]  # Angle of the pendulum
        cos_theta = state[1]  # Cosine of the pendulum angle
        theta_dot = state[2]  # Angular velocity of the pendulum
    
    else:
        print("Unexpected type for state[0]:", type(state[0]))
    
    theta = np.arcsin(sin_theta)  # Convert sin(theta) to theta
    #print("sin(theta):", sin_theta, "cos(theta):", cos_theta, "theta_dot:", theta_dot, "theta:", theta)

    # Calculate error
    error = theta
    last_sign_error = np.sign(error)
    # Update integral term
    if np.sign(error) != last_sign_error:
      self.integral_error = 0
    else:
      self.integral_error += error * dt

    # Calculate control action with weighted terms
    control_action = self.kp * error + self.ki * self.integral_error + self.kd * theta_dot

    print("Error:", error, "Integral error:", self.integral_error, "control action:", control_action)

    # saturate the control action to the valid range
    control_action = np.clip(control_action, -2, 2)
    #print("Control action:", control_action)
    # Convert control_action to an array-like structure
    control_action = np.array([control_action])

    return control_action
