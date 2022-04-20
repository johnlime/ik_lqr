from env.pigeon_gym import PigeonEnv3Joints, BODY_WIDTH, BODY_HEIGHT, LIMB_WIDTH, LIMB_HEIGHT, HEAD_WIDTH, ANGLE_FREEDOM, MAX_JOINT_TORQUE, MAX_JOINT_SPEED, VELOCITY_WEIGHT, LIMB_DENSITY, LIMB_FRICTION

import numpy as np
from math import pi, sqrt
from Box2D import *
from copy import copy, deepcopy

class IKPigeon(PigeonEnv3Joints):
    def __init__(self,
                 body_speed = 0,
                 reward_code = "head_stable_manual_reposition",
                 max_offset = 0.5,
                 joints = 3):

        self.num_joints = joints - 1
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
            # turn default angle from 0 to pi/2 since everything is pointing to negative
            # optional move is to redesign the model to face the positive direction
            angle_tmp += pi

            if i == 0:
                angle_tmp -= pi / 4
            elif i == len(self.joints) - 1:
                angle_tmp += pi / 4

            # probably no need to cap angle_tmp range

            obs = np.concatenate((obs, angle_tmp), axis = None)
            obs = np.concatenate((obs, self.joints[i].speed), axis = None)

        # complement the body's x position and the target position
        obs = np.concatenate((obs, self.body.position[0]), axis = None)
        obs = np.concatenate((obs, self.head_target_location - np.array(self.body.position)),
                              axis = None)

        obs = np.float32(obs)
        assert self.observation_space.contains(obs)
        return obs
