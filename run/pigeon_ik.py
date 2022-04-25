from env.ik_pigeon_gym import IKPigeon, calc_end_effector_pos, get_jacobian
import numpy as np
import pickle

env = IKPigeon()
for iter_num in range(1):
    x = env.reset()
    for _ in range(1000):
        env.render()
        e = np.array(x[-2:] - x[:2]).reshape((2, 1))
        J = get_jacobian(x[[3, 5, 7]])
        J_pinv = np.linalg.pinv(J)
        u = np.matmul(J_pinv, e)
        x, _, done, _ = env.step(u.reshape(3).astype(np.float32))
        if done:
            break
env.close()
