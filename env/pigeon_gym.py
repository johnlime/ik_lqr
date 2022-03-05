from Box2D import *
import gym
from gym import spaces

from math import sin, pi, sqrt
import numpy as np
from copy import copy, deepcopy

# anatomical variables ("macros")
BODY_WIDTH = 10
BODY_HEIGHT = 5

LIMB_WIDTH = 5
LIMB_HEIGHT = 2

HEAD_WIDTH = 3

ANGLE_FREEDOM = 0.6

# control variables/macros
MAX_JOINT_TORQUE = 200 #70
MAX_JOINT_SPEED = 5 #10
VELOCITY_WEIGHT = 1.0 #0.9
LIMB_DENSITY = 0.1 ** 3
LIMB_FRICTION = 5

VIEWPORT_SCALE = 6.0
FPS = 60

HEAD_OFFSET_X = 10
HEAD_OFFSET_Y = 2

class PigeonEnv3Joints(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": FPS}

    def __init__(self,
                 body_speed = 0,
                 reward_code = "head_stable_manual_reposition",
                 max_offset = 0.5):
        """
        Action and Observation space
        """

        # 3-dim joints' torque ratios
        self.action_space = spaces.Box(
            np.array([-1.0] * 3).astype(np.float32),
            np.array([1.0] * 3).astype(np.float32),
        )
        # 2-dim head location;
        # 1-dim head angle;
        # 3x2-dim joint angle and angular velocity;
        # 1-dim x-axis of the body
        # [NEW] 2-dim target head location
        high = np.array([np.inf] * 12).astype(np.float32) # formally 10
        self.observation_space = spaces.Box(-high, high)

        """
        Box2D Pigeon Model Params and Initialization
        """
        self.world = b2World()                          # remove in Framework
        self.body = None
        self.joints = []
        self.head = None
        self.bodyRef = [] # for destruction
        self.body_speed = body_speed
        self._pigeon_model()

        """
        Box2D Simulation Params
        """
        self.timeStep = 1.0 / FPS
        self.vel_iters, self.pos_iters = 10, 10

        self.viewer = None

        """
        Assigning a Reward Function
        """
        self._assign_reward_func(reward_code, max_offset)

    """
    Define Reward Function and Necessary Parameters
    """
    def _assign_reward_func(self, reward_code, max_offset):
        if "head_stable_manual_reposition" in reward_code:
            self.max_offset = max_offset

            self.relative_repositioned_head_target_location = np.array(self.head.position) - np.array([0, HEAD_OFFSET_Y])
            self.head_target_location = self.relative_repositioned_head_target_location + np.array(self.body.position)
            self.head_target_angle = self.head.angle
            self.reward_function = self._head_stable_manual_reposition

            if "strict_angle" in reward_code:
                self.reward_function = self._head_stable_manual_reposition_strict_angle

        else:
            raise ValueError("Unknown reward_code")

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
        for i in range(2):
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

    def _destroy(self):
        for body in self.bodyRef:
            # all associated joints are destroyed implicitly
            self.world.DestroyBody(body)

    def _get_obs(self):
        # (self.head{relative}, self.joints -> obs) operation
        obs = np.array(self.head.position) - np.array(self.body.position)
        obs = np.concatenate((obs, self.head.angle), axis = None)
        for i in range(len(self.joints)):
            obs = np.concatenate((obs, self.joints[i].angle), axis = None)
            obs = np.concatenate((obs, self.joints[i].speed), axis = None)
        obs = np.concatenate((obs, self.body.position[0]), axis = None)

        # complement a target position
        obs = np.concatenate((obs, self.head_target_location - np.array(self.body.position)),
                              axis = None)

        obs = np.float32(obs)
        assert self.observation_space.contains(obs)
        return obs

    def reset(self):
        self._destroy()
        self._pigeon_model()
        return self._get_obs()

    def _head_target_reposition_mechanism(self):
        # detect whether the target head position is behind the body edge or not
        if self.head_target_location[0] > self.body.position[0] - float(BODY_WIDTH + HEAD_OFFSET_X):
            self.head_target_location = np.array(self.body.position) + \
                self.relative_repositioned_head_target_location

        head_dif_loc = np.linalg.norm(np.array(self.head.position) - \
                self.head_target_location)
        head_dif_ang = abs(self.head.angle - self.head_target_angle)
        return head_dif_loc, head_dif_ang

    """
    Modular Reward Functions
    """
    def _head_stable_manual_reposition(self):
        # This method is separated from step(), since there are variables used
        # that are only defined in with this strain of reward functions
        head_dif_loc, head_dif_ang = self._head_target_reposition_mechanism()

        reward = 0
        # threshold reward function with static offset
        if head_dif_loc < self.max_offset:
            reward += 1 - head_dif_loc/self.max_offset

            if head_dif_ang < np.pi / 6: # 30 deg
                reward += 1 - head_dif_ang/ np.pi

        return reward

    def _head_stable_manual_reposition_strict_angle(self):
        head_dif_loc, head_dif_ang = self._head_target_reposition_mechanism()

        reward = 0
        # threshold reward function with static offset
        if head_dif_loc < self.max_offset:
            if head_dif_ang < np.pi / 6: # 30 deg
                reward += 1 - head_dif_ang/ np.pi

        return reward

    def step(self, action):
        assert self.action_space.contains(action)
        # self.world.Step(self.timeStep, self.vel_iters, self.pos_iters)
        # Framework handles this differently
        # Referenced bipedal_walker
        # self.world.Step(1.0 / 50, 6 * 30, 2 * 30)
        self.world.Step(1.0 / FPS, self.vel_iters, self.pos_iters)
        obs = self._get_obs()

        # MOTOR CONTROL
        for i in range(len(self.joints)):
            # Copied from bipedal_walker
            self.joints[i].motorSpeed = float(MAX_JOINT_SPEED * (VELOCITY_WEIGHT ** i) * np.sign(action[i]))
            self.joints[i].maxMotorTorque = float(
                MAX_JOINT_TORQUE * np.clip(np.abs(action[i]), 0, 1)
            )

        reward = self.reward_function()

        done = False
        info = {}
        return obs, reward, done, info

    def render(self, mode = "human"):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)

            # Set ORIGIN POINT relative to camera
            self.camera_trans = b2Vec2(-250, -200) \
            + VIEWPORT_SCALE * self.bodyRef[0].position # camera moves with body

            ## Needs head_stable_manual_reposition reward function to execute
            try:
                # init visualize max_offset
                render_target_area = rendering.make_circle( \
                    radius=VIEWPORT_SCALE * self.max_offset,
                    res=30,
                    filled=True)
                target_translate = rendering.Transform(
                    translation = VIEWPORT_SCALE * self.head_target_location - self.camera_trans,
                    rotation = 0.0,
                    scale = VIEWPORT_SCALE * np.ones(2)
                )
                render_target_area.add_attr(self.target_translate)
                render_target_area.set_color(0.0, 1.0, 0.0)
                self.viewer.add_geom(render_target_area)
            except:
                pass

            # init translation and rotation for each limb
            self.render_polygon_list = []
            self.render_polygon_rotate_list = []
            self.render_polygon_translate_list = []
            for body in self.bodyRef:
                polygon = rendering.FilledPolygon(
                    body.fixtures[0].shape.vertices
                )
                rotate = rendering.Transform(
                    translation = (0.0, 0.0),
                    rotation = body.angle,
                )
                translate = rendering.Transform(
                    translation = VIEWPORT_SCALE * body.position - self.camera_trans,
                    rotation = 0.0,
                    scale = VIEWPORT_SCALE * np.ones(2)
                )
                polygon.set_color(1.0, 0.0, 0.0)
                polygon.add_attr(rotate)
                polygon.add_attr(translate)
                self.render_polygon_list.append(polygon)
                self.render_polygon_rotate_list.append(rotate)
                self.render_polygon_translate_list.append(translate)
                self.viewer.add_geom(polygon)

        # Update ORIGIN POINT relative to camera
        self.camera_trans = b2Vec2(-250, -200) \
        + VIEWPORT_SCALE * self.bodyRef[0].position # camera moves with body

        ## Needs head_stable_manual_reposition reward function to execute
        try:
            # update max_offset shape translation
            new_target_translate = VIEWPORT_SCALE * self.head_target_location - self.camera_trans
            self.target_translate.set_translation(new_target_translate[0], new_target_translate[1])
        except:
            pass

        # update body rotation and translation
        for i, body in enumerate(self.bodyRef):
            self.render_polygon_rotate_list[i].set_rotation(body.angle)
            new_body_translate = VIEWPORT_SCALE * body.position - self.camera_trans
            self.render_polygon_translate_list[i].set_translation(new_body_translate[0], new_body_translate[1])

        return self.viewer.render(return_rgb_array = mode == "rgb_array")

    def close(self):
        # self._destroy()
        # self.world = None

        if self.viewer:
            self.viewer.close()
            self.viewer = None
