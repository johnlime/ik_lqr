from env.ik_pigeon_gym import IKPigeon, BODY_WIDTH, BODY_HEIGHT, LIMB_WIDTH, LIMB_HEIGHT, HEAD_WIDTH
import numpy as np
import pickle

env = IKPigeon()
for iter_num in range(1):
    x = env.reset()
    print(x) # 45 deg is 0
    for _ in range(1):
        env.render()
        u = env.action_space.sample()
        x, _, done, _ = env.step(u)
        print(x)
        if done:
            break
env.close()


def calc_end_effector_pos(start_to_end_angles, index = len(start_to_end_angles) - 1):
    """
    start_to_end_angles: array of joint angles starting from those
    closest to the body to the those furthest from the body
    """
    assert index < len(start_to_end_angles)

    end_effector = np.array([- BODY_WIDTH, BODY_HEIGHT])

    angle_cumul = 0
    for i in range(len(start_to_end_angles) - 1):
        angle_cumul += start_to_end_angles[i]
        end_effector += 2 * LIMB_WIDTH * np.array([- np.cos(angle_cumul), np.sin(angle_cumul)])

    angle_cumul += start_to_end_angles[i]
    end_effector += HEAD_WIDTH * np.array([- np.cos(angle_cumul), np.sin(angle_cumul)])

    return end_effector
