import math
import numpy as np

def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)
    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)
    return u, trajectory

def calc_dynamic_window(x, config):
    """
    Calculate Dynamic Window based on current state x
    """
    Vs = [config.min_speed, config.max_speed, -config.max_yawrate, config.max_yawrate]

    Vd = [
        x[3] - config.max_accel * config.dt,
        x[3] + config.max_accel * config.dt,
        x[4] - config.max_dyawrate * config.dt,
        x[4] + config.max_dyawrate * config.dt
    ]

    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw

def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    Calculation of final input with dynamic window
    """
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array(x)

    for v in np.arange(dw[0], dw[1], config.v_reso):
        for y in np.arange(dw[2], dw[3], config.yawrate_reso):
            trajectory = predict_trajectory(x_init, v, y, config)

            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost

            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory

    return best_u, best_trajectory

def predict_trajectory(x_init, v, y, config):
    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt
    return trajectory

def motion(x, u, dt):
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[2] += u[1] * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

def calc_to_goal_cost(trajectory, goal):
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    cost = math.hypot(dx, dy)
    return cost

def calc_obstacle_cost(trajectory, ob, config):
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0][:, np.newaxis] - ox[np.newaxis, :]
    dy = trajectory[:, 1][:, np.newaxis] - oy[np.newaxis, :]
    r = np.hypot(dx, dy)
    if np.array(r <= config.robot_radius).any():
        return float("Inf")
    return 1.0 / np.min(r)
