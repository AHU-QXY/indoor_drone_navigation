import os
from enum import Enum
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math

from envs.customized.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType, ActionType, ObservationType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class BaseSingleAgentAviary(BaseAviary):
    """Base single drone environment class for reinforcement learning."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 use_random_start: bool = True,
                 use_random_goal: bool = True
                 ):
        """Initialization of a generic single agent RL environment.

        Attribute `num_drones` is automatically set to 1; `vision_attributes`
        are selected based on the choice of `obs` and `act`; `obstacles` is
        set to True and overridden with landmarks for vision applications;
        `user_debug_gui` is set to False for performance.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = 8
        # self.goalPos = goal
        self.use_random_start = use_random_start  # TODO

        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
            else:
                print(
                    "[ERROR] in BaseSingleAgentAviary.__init()__, no controller is available for the specified drone_model")
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=True,  # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False,  # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         use_random_start=use_random_start,
                         use_random_goal=use_random_goal
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000 / 3600)


    ################################################################################
    def generate_random_point(self):
        while True:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            return [x, y, 0.75]

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            p.loadURDF("block.urdf",
                       [1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("cube_small.urdf",
                       [0, 1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("duck_vhacd.urdf",
                       [-1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("teddy_vhacd.urdf",
                       [0, -1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
        else:
            self.obstacleId = []

            # 边框
            base_pos_1 = np.array([2.8, 4.095, 1])
            objectId_1 = p.loadURDF("block1.urdf", base_pos_1, p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                    physicsClientId=self.CLIENT)
            self.obstacleId.append(objectId_1)
            base_pos_2 = np.array([2.8, 0.105, 1])
            objectId_2 = p.loadURDF("block1.urdf", base_pos_2, p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True,
                                    physicsClientId=self.CLIENT)
            self.obstacleId.append(objectId_2)
            base_pos_3 = np.array([0.105, 2.1, 1])
            objectId_3 = p.loadURDF("block2.urdf", base_pos_3, p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True,
                                    physicsClientId=self.CLIENT)
            self.obstacleId.append(objectId_3)
            base_pos_4 = np.array([5.475, 2.1, 1])
            objectId_4 = p.loadURDF("block2.urdf", base_pos_4, p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True,
                                    physicsClientId=self.CLIENT)
            self.obstacleId.append(objectId_4)
            base_pos_5 = np.array([0.665, 1.715, 1])
            objectId_5 = p.loadURDF("blockin1.urdf", base_pos_5, p.getQuaternionFromEuler([0, 0, 0]),
                                    useFixedBase=True,
                                    physicsClientId=self.CLIENT)
            self.obstacleId.append(objectId_5)
            base_pos_6 = np.array([2.1, 2.2, 1])
            objectId_6 = p.loadURDF("U5.urdf", base_pos_6, p.getQuaternionFromEuler([0, 0, np.pi/2]),
                                    useFixedBase=True,
                                    physicsClientId=self.CLIENT)
            self.obstacleId.append(objectId_6)
            base_pos_7 = np.array([4, 1.8, 1])
            objectId_7 = p.loadURDF("U5.urdf", base_pos_7, p.getQuaternionFromEuler([0, 0, np.pi/2]),
                                    useFixedBase=True,
                                    physicsClientId=self.CLIENT)
            self.obstacleId.append(objectId_7)
            base_pos_8 = np.array([3.2, 2.2, 1])
            objectId_8 = p.loadURDF("U5.urdf", base_pos_8, p.getQuaternionFromEuler([0, 0, 0]),
                                    useFixedBase=True,
                                    physicsClientId=self.CLIENT)
            self.obstacleId.append(objectId_8)
            # base_pos_6 = np.array([0.665, 2.485, 1])
            # objectId_6 = p.loadURDF("blockin1.urdf", base_pos_6, p.getQuaternionFromEuler([0, 0, 0]),
            #                         useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_6)
            # base_pos_7 = np.array([4.875, 1.715, 1])
            # objectId_7 = p.loadURDF("blockin1.urdf", base_pos_7, p.getQuaternionFromEuler([0, 0, 0]),
            #                         useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_7)
            base_pos_8 = np.array([4.875, 2.485, 1])
            objectId_8 = p.loadURDF("blockin1.urdf", base_pos_8, p.getQuaternionFromEuler([0, 0, 0]),
                                    useFixedBase=True,
                                    physicsClientId=self.CLIENT)
            self.obstacleId.append(objectId_8)

            # base_pos_9 = np.array([3.255, 2.485, 1])
            # objectId_9 = p.loadURDF("blockin2.urdf", base_pos_9, p.getQuaternionFromEuler([0, 0, 0]),
            #                         useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_9)
            base_pos_9 = np.array([3.655, 2.485, 1])
            objectId_9 = p.loadURDF("blockin3.urdf", base_pos_9, p.getQuaternionFromEuler([0, 0, 0]),
                                    useFixedBase=True,
                                    physicsClientId=self.CLIENT)
            self.obstacleId.append(objectId_9)
            # base_pos_10 = np.array([3.255, 1.715, 1])
            # objectId_10 = p.loadURDF("blockin2.urdf", base_pos_10, p.getQuaternionFromEuler([0, 0, 0]),
            #                         useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_10)
            #
            # base_pos_5 = np.array([1.665, 1.4, 1])
            # objectId_5 = p.loadURDF("blockin5.urdf", base_pos_5, p.getQuaternionFromEuler([0, 0, np.pi/2]),
            #                         useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_5)
            # base_pos_6 = np.array([2.3, 3.6, 1])
            # objectId_6 = p.loadURDF("blockin6.urdf", base_pos_6, p.getQuaternionFromEuler([0, 0, np.pi/2]),
            #                         useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_6)
            # base_pos_7 = np.array([4, 2.8, 1])
            # objectId_7 = p.loadURDF("blockin7.urdf", base_pos_7, p.getQuaternionFromEuler([0, 0, 0]),
            #                         useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_7)
            # base_pos_8 = np.array([3.6, 1.8, 1])
            # objectId_8 = p.loadURDF("blockin8.urdf", base_pos_8, p.getQuaternionFromEuler([0, 0, 0]),
            #                         useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_8)
            # base_pos_9 = np.array([3.6, 1.8, 1])
            # objectId_9 = p.loadURDF("blockin8.urdf", base_pos_9, p.getQuaternionFromEuler([0, 0, np.pi/2]),
            #                         useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_9)
            #
            # base_pos_10 = np.array([0.6, 1.715, 1])
            # objectId_10 = p.loadURDF("blockin9.urdf", base_pos_10, p.getQuaternionFromEuler([0, 0, 0]),
            #                         useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_10)
            # #
            # base_pos_11 = np.array([1.925, 2.485, 1])
            # objectId_11 = p.loadURDF("blockin3.urdf", base_pos_11, p.getQuaternionFromEuler([0, 0, 0]),
            #                         useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_11)
            base_pos_12 = np.array([1.925, 1.715, 1])
            objectId_12 = p.loadURDF("blockin3.urdf", base_pos_12, p.getQuaternionFromEuler([0, 0, 0]),
                                    useFixedBase=True,
                                    physicsClientId=self.CLIENT)
            self.obstacleId.append(objectId_12)
            # #
            # base_pos_13 = np.array([1.925, 3.29, 1])
            # objectId_13 = p.loadURDF("blockin4.urdf", base_pos_13, p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_13)
            base_pos_14 = np.array([1.925, 0.91, 1])
            objectId_14 = p.loadURDF("blockin4.urdf", base_pos_14, p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True,
                                    physicsClientId=self.CLIENT)
            self.obstacleId.append(objectId_14)
            base_pos_15 = np.array([3.605, 3.29, 1])
            objectId_15 = p.loadURDF("blockin4.urdf", base_pos_15, p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True,
                                    physicsClientId=self.CLIENT)
            self.obstacleId.append(objectId_15)
            # base_pos_16 = np.array([3.605, 0.91, 1])
            # objectId_16 = p.loadURDF("blockin4.urdf", base_pos_16, p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_16)
            # base_pos_17 = np.array([1.925, 2.25, 1])
            # objectId_17 = p.loadURDF("blockin5.urdf", base_pos_17, p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_17)
            # base_pos_18 = np.array([3.605, 1.95, 1])
            # objectId_18 = p.loadURDF("blockin5.urdf", base_pos_18, p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_18)
            # base_pos_19 = np.array([4.875, 2.25, 1])
            # objectId_19 = p.loadURDF("blockin5.urdf", base_pos_19, p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_19)
            # base_pos_20 = np.array([0.665, 1.95, 1])
            # objectId_20 = p.loadURDF("blockin5.urdf", base_pos_20, p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True,
            #                         physicsClientId=self.CLIENT)
            # self.obstacleId.append(objectId_20)

            # add goal
            # TODO:if you want to set fixed Point -> self.goalPos = []
            # TODO:if you want to set random point -> self.goalPos = np.random.uniform(low= , high=, size=(1,3))
            if self.use_random_goal:
                self.goalPos = []
                x = np.random.uniform(low=4.15, high=4.25)
                y = np.random.uniform(low=2.65, high=2.9)
                self.goalPos.append(x)  # 将x坐标添加到列表中
                self.goalPos.append(y)  # 将y坐标添加到列表中
                self.goalPos.append(0.5)  # z
                self.goalPos1 = [1.82, 2.1, 0.5]
                self.goalPos2 = [3, 2.1, 0.5]
                self.goalPos3 = [4.1, 2.1, 0.5]
                self.wall1 = [4.41, 2.48]
                self.wall2 = [3.92, 2.48]

                # self.goalPos = np.expand_dims(self.goalPos, axis=0)  # Format
            else:
                self.goalPos1 = [1.82, 2.1, 0.5]
                self.goalPos2 = [3, 2.1, 0.5]
                self.goalPos3 = [4.2, 2.1, 0.5]
                self.goalPos = [4.15, 2.65, 0.5]
                self.wall1 = [4.41, 2.48]
                self.wall2 = [3.92, 2.48]
            self.goalId = p.loadURDF('envs/customized/urdf/sphere.urdf', self.goalPos, useFixedBase=True,
                                     physicsClientId=self.CLIENT)
            if self.use_random_start:
                self.INIT_XYZS = []
                x = np.random.uniform(low=1.3, high=1.5)  # 生成一个随机的x坐标，范围在[-0.5, 0]
                y = np.random.uniform(low=1, high=1.4)  # 根据条件 x + y = -0.5 计算y坐标
                self.INIT_XYZS.append(x)  # 将x坐标添加到列表中
                self.INIT_XYZS.append(y)  # 将y坐标添加到列表中
                self.INIT_XYZS.append(0.5)  # z
                self.INIT_XYZS = np.expand_dims(self.INIT_XYZS, axis=0)  # Format
            else:
                self.INIT_XYZS = []
                self.INIT_XYZS.append(-0.25)  # x
                self.INIT_XYZS.append(-0.25)  # y
                self.INIT_XYZS.append(0.02)  # z
                self.INIT_XYZS = np.expand_dims(self.INIT_XYZS, axis=0)  # Format

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        ndarray
            A Box() of size 1, 3, 4, or 6 depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE == ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseSingleAgentAviary._actionSpace()")
            exit()
        return spaces.Box(low=-1 * np.ones(size),
                          # return spaces.Box(low=np.zeros(size),  # Alternative action space, see PR #32
                          high=np.ones(size),
                          dtype=np.float32
                          )

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, 4, or 6 and represent
        RPMs, desired thrust and torques, the next target position to reach
        using PID control, a desired velocity vector, new PID coefficients, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        if self.ACT_TYPE == ActionType.RPM:
            return np.array(self.HOVER_RPM * (1 + 0.05 * action))
        elif self.ACT_TYPE == ActionType.PID:
            state = self._getDroneStateVector(0)
            next_pos = self._calculateNextStep(
                current_position=state[0:3],
                destination=action,
                step_size=1,
            )
            rpm, _, _ = self.ctrl.computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                 cur_pos=state[0:3],
                                                 cur_quat=state[3:7],
                                                 cur_vel=state[10:13],
                                                 cur_ang_vel=state[13:16],
                                                 target_pos=next_pos
                                                 )
            return rpm
        elif self.ACT_TYPE == ActionType.VEL:
            state = self._getDroneStateVector(0)
            if np.linalg.norm(action[0:3]) != 0:
                v_unit_vector = action[0:3] / np.linalg.norm(action[0:3])
            else:
                v_unit_vector = np.zeros(3)
            rpm, _, _ = self.ctrl.computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                 cur_pos=state[0:3],
                                                 cur_quat=state[3:7],
                                                 cur_vel=state[10:13],
                                                 cur_ang_vel=state[13:16],
                                                 target_pos=state[0:3],  # same as the current position
                                                 target_rpy=np.array([0, 0, state[9]]),  # keep current yaw
                                                 target_vel=self.SPEED_LIMIT * np.abs(action[3]) * v_unit_vector
                                                 # target the desired velocity vector
                                                 )
            return rpm
        elif self.ACT_TYPE == ActionType.ONE_D_RPM:
            return np.repeat(self.HOVER_RPM * (1 + 0.05 * action), 4)
        elif self.ACT_TYPE == ActionType.ONE_D_PID:
            state = self._getDroneStateVector(0)
            rpm, _, _ = self.ctrl.computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                 cur_pos=state[0:3],
                                                 cur_quat=state[3:7],
                                                 cur_vel=state[10:13],
                                                 cur_ang_vel=state[13:16],
                                                 target_pos=state[0:3] + 0.1 * np.array([0, 0, action[0]])
                                                 )
            return rpm
        else:
            print("[ERROR] in BaseSingleAgentAviary._preprocessAction()")

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.IMG_RES[1], self.IMG_RES[0], 4),
                              dtype=np.uint8
                              )
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
            # obs_lower_bound = np.array([-1,      -1,      0,      -1,  -1,  -1,  -1,  -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,      -1,           -1,           -1,           -1])
            # obs_upper_bound = np.array([1,       1,       1,      1,   1,   1,   1,   1,      1,      1,      1,       1,       1,       1,       1,       1,       1,            1,            1,            1])
            # return spaces.Box( low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32 )
            ############################################################
            # OBS SPACE OF SIZE 15
            return spaces.Box(#low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]),
                              # high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                              # low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]),
                              # high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                              low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0]),
                              high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                              dtype=np.float32
                              )

            ############################################################
        else:
            print("[ERROR] in BaseSingleAgentAviary._observationSpace()")

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
                self.rgb[0], self.dep[0], self.seg[0] = self._getDroneImages(0,
                                                                             segmentation=False
                                                                             )
                #### Printing observation to PNG frames example ############
                if self.RECORD:
                    self._exportImage(img_type=ImageType.RGB,
                                      img_input=self.rgb[0],
                                      path=self.ONBOARD_IMG_PATH,
                                      frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ)
                                      )
            return self.rgb[0]
        elif self.OBS_TYPE == ObservationType.KIN:
            obs = self._getDroneStateVector(0)
            # ret = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], self.goalPos]).reshape(15, )
            ret = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], self.goalPos, self.goalPos1, self.goalPos2, self.goalPos3]).reshape(24, )
            return ret.astype('float32')
            ############################################################
        else:
            print("[ERROR] in BaseSingleAgentAviary._computeObs()")

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Must be implemented in a subclass.

        Parameters
        ----------
        state : ndarray
            Array containing the non-normalized state of a single drone.

        """
        raise NotImplementedError
