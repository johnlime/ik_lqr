from env.pigeon_gym import *

import numpy as np
from math import pi, sqrt
from Box2D import *
from copy import copy, deepcopy

class IKPigeon(PigeonEnv3Joints):
    def __init__(self,
                 body_speed = 0,
                 reward_code = "head_stable_manual_reposition",
                 max_offset = 0.5,
                 joints = 3,
                 velocity_control = True):

        self.num_joints = joints - 1
        self.velocity_control = velocity_control
        super().__init__(
            body_speed = body_speed,
            reward_code = reward_code,
            max_offset = max_offset)

    """
    Box2D Pigeon Model
    """
    def _pigeon_model(self):
        # params
        body_anchor = np.array([float(-BODY_WIDTH), float(BODY_HEIGHT)])
        limb_width_cos = LIMB_WIDTH / sqrt(2)

        self.bodyRef = []
        # body definition
        self.body = self.world.CreateKinematicBody(
            position = (0, 0),
            shapes = b2PolygonShape(box = (BODY_WIDTH, BODY_HEIGHT)), # x2 in direct shapes def
            linearVelocity = (-self.body_speed, 0),
            angularVelocity = 0,
            )
        self.bodyRef.append(self.body)

        # neck as limbs + joints definition
        self.joints = []
        current_center = deepcopy(body_anchor)
        current_anchor = deepcopy(body_anchor)
        offset = np.array([-limb_width_cos, limb_width_cos])
        prev_limb_ref = self.body
        for i in range(self.num_joints):
            if i == 0:
                current_center += offset

            else:
                current_center += offset * 2
                current_anchor += offset * 2

            tmp_limb = self.world.CreateDynamicBody(
                position = (current_center[0], current_center[1]),
                fixtures = b2FixtureDef(density = LIMB_DENSITY,
                                        friction = LIMB_FRICTION,
                                        restitution = 0.0,
                                        shape = b2PolygonShape(
                                            box = (LIMB_WIDTH, LIMB_HEIGHT)),
                                        ),
                angle = -pi / 4
            )
            self.bodyRef.append(tmp_limb)

            tmp_joint = self.world.CreateRevoluteJoint(
                bodyA = prev_limb_ref,
                bodyB = tmp_limb,
                anchor = current_anchor,
                lowerAngle = -ANGLE_FREEDOM * b2_pi, # -90 degrees
                upperAngle = ANGLE_FREEDOM * b2_pi,  #  90 degrees
                enableLimit = True,
                maxMotorTorque = MAX_JOINT_TORQUE,
                motorSpeed = 0.0,
                enableMotor = True,
            )

            self.joints.append(tmp_joint)
            prev_limb_ref = tmp_limb

        # head def + joints
        current_center += offset
        current_anchor += offset * 2
        self.head = self.world.CreateDynamicBody(
            position = (current_center[0] - HEAD_WIDTH, current_center[1]),
            fixtures = b2FixtureDef(density = LIMB_DENSITY,
                                    friction = LIMB_FRICTION,
                                    restitution = 0.0,
                                    shape = b2PolygonShape(
                                        box = (HEAD_WIDTH, LIMB_HEIGHT)),
                                    ),
        )
        self.bodyRef.append(self.head)

        head_joint = self.world.CreateRevoluteJoint(
            bodyA = prev_limb_ref,
            bodyB = self.head,
            anchor = current_anchor,
            lowerAngle = -ANGLE_FREEDOM * b2_pi, # -90 degrees
            upperAngle = ANGLE_FREEDOM * b2_pi,  #  90 degrees
            enableLimit = True,
            maxMotorTorque = MAX_JOINT_TORQUE,
            motorSpeed = 0.0,
            enableMotor = True,
        )
        self.joints.append(head_joint)

        # head tracking
        self.head_prev_pos = np.array(self.head.position)
        self.head_prev_ang = self.head.angle


    def _get_obs(self):
        # (self.head{relative}, self.joints -> obs) operation
        obs = np.array(self.head.position) - np.array(self.body.position)
        obs = np.concatenate((obs, self.head.angle), axis = None)

        # more appropriate to add joint angles relative to root rather than head angles
        # offset angles (counter-clockwise is positive)
        for i in range(len(self.joints)):
            angle_tmp = self.joints[i].angle
            # intuitive to use negative x (length) and positive y
            # flip angle_tmp by the BOTH x AND y axis before output (to set clockwise as positive # to set the x axis from positive to negative domain)

            if i == 0:
                angle_tmp -= pi / 4
            elif i == len(self.joints) - 1:
                angle_tmp += pi / 4

            # probably no need to cap angle_tmp range

            obs = np.concatenate((obs, angle_tmp), axis = None)
            obs = np.concatenate((obs, self.joints[i].speed), axis = None)

        # complement the body's x position and the relative target position
        obs = np.concatenate((obs, self.body.position[0]), axis = None)
        obs = np.concatenate((obs, self.head_target_location - np.array(self.body.position)),
                              axis = None)

        obs = np.float32(obs)
        assert self.observation_space.contains(obs)
        return obs

    def step(self, action):
        assert self.action_space.contains(action)
        # self.world.Step(self.timeStep, self.vel_iters, self.pos_iters)
        # Framework handles this differently
        # Referenced bipedal_walker
        # self.world.Step(1.0 / 50, 6 * 30, 2 * 30)
        self.world.Step(1.0 / FPS, self.vel_iters, self.pos_iters)
        obs = self._get_obs()

        # VELOCITY OR MOTOR CONTROL
        for i in range(len(self.joints)):
            # "action space" should reflect real joint speed derived using Jacobian
            if self.velocity_control:
                self.joints[i].motorSpeed = np.clip(action[i], float(- MAX_JOINT_SPEED), float(MAX_JOINT_SPEED)))
            else:
                self.joints[i].motorSpeed = float(MAX_JOINT_SPEED * (VELOCITY_WEIGHT ** i) * np.sign(action[i]))
                self.joints[i].maxMotorTorque = float(
                    MAX_JOINT_TORQUE * np.clip(np.abs(action[i]), 0, 1)
                )

        reward = self.reward_function()

        done = False
        info = {}
        return obs, reward, done, info


"""
Misc functions that may be useful for this environment
"""
def calc_end_effector_pos(theta):
    """
    theta: 1-D array of joint angles starting from those
    closest to the body to the those furthest from the body
    """
    end_effector = np.tile(
        np.array([- BODY_WIDTH, BODY_HEIGHT]).astype(np.float),
        (len(theta), 1))

    angle_cumul = 0
    coef = 1
    for i in range(len(theta)):
        angle_cumul += theta[i]
        coef = 2 * LIMB_WIDTH
        if i == len(theta) - 1:
            coef = HEAD_WIDTH
        # flip angle_tmp by the y axis before output
        end_effector[i:, :] -= coef * \
            np.array([np.cos(angle_cumul), np.sin(angle_cumul)])

    return end_effector


def get_jacobian(theta):
    n = len(theta)
    Jacobian = np.zeros((2, n))
    # additional variables
    coef = 1
    theta_j_sum = 0
    # masked array for sum of theta excluding l differentiator
    masked_theta = np.ma.array(theta, mask = False)
    for l in range(n):
        masked_theta.mask[l] = True
        for i in range(l, n):
            # determine coefficient (length of limb assigned to joint)
            coef = 2 * LIMB_WIDTH
            if i == len(theta) - 1:
                coef = HEAD_WIDTH
            # pre-calculate sum of angles excluding l
            theta_j_sum = masked_theta[:i + 1].sum()
            # sinusoidal gradients
            Jacobian[0][l] = coef * ( \
                np.sin(theta[l]) * np.cos(theta_j_sum) - \
                np.cos(theta[l]) * np.sin(theta_j_sum))
            Jacobian[1][l] = coef * ( \
                np.sin(theta[l]) * np.sin(theta_j_sum) - \
                np.cos(theta[l]) * np.cos(theta_j_sum))
        # revert the masked_theta masks
        masked_theta.mask[l] = False
    return Jacobian
