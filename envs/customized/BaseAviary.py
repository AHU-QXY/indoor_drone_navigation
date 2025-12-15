import os
from sys import platform
import time
import collections
from datetime import datetime
import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image
# import pkgutil
# egl = pkgutil.get_loader('eglRenderer')
import math
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType

from gym_pybullet_drones.envs.BaseAviary import BaseAviary as BaseAviary_Official


class BaseAviary(BaseAviary_Official):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,  # TODO
                 vision_attributes=False,
                 output_folder='results',
                 use_random_start: bool = True,
                 use_random_goal: bool = True
                 ):
        self.use_random_start = use_random_start  # TODO
        self.use_random_goal = use_random_goal
        # self.radia_results = np.zeros(16)
        super(BaseAviary, self).__init__(drone_model=drone_model,
                                         num_drones=num_drones,
                                         neighbourhood_radius=neighbourhood_radius,
                                         initial_xyzs=initial_xyzs,
                                         initial_rpys=initial_rpys,
                                         physics=physics,
                                         pyb_freq=pyb_freq,
                                         ctrl_freq=ctrl_freq,
                                         gui=gui,
                                         record=record,
                                         obstacles=obstacles,
                                         user_debug_gui=user_debug_gui,
                                         vision_attributes=vision_attributes,
                                         output_folder=output_folder
                                         )
        # self.goal1_flag_changed = False
        self.goal1_flag = False
        self.goal2_flag = False
        self.goal3_flag = False

    # def _radia(self):
    #     hitRayColor = [0, 1, 0]
    #     missRayColor = [1, 0, 0]
    #     rayLength = 0.3  # 激光长度
    #     rayNum = 16  # 激光数量
    #
    #     # 获取需要发射得rayNum激光的起点与终点
    #     begins, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[0], physicsClientId=self.CLIENT)
    #     theta = self.rpy[0, :][2]
    #     rayFroms = [begins for _ in range(rayNum)]
    #     rayTos = [
    #         [
    #             begins[0] + rayLength * math.cos(2 * math.pi * float(i) / rayNum + theta),
    #             begins[1] + rayLength * math.sin(2 * math.pi * float(i) / rayNum + theta),
    #             begins[2]
    #         ]
    #         for i in range(rayNum)]
    #
    #     # 调用激光探测函数
    #     results = p.rayTestBatch(rayFroms, rayTos)
    #     for index, result in enumerate(results):
    #         if result[0] == -1:
    #             self.radia_results[index] = 0
    #         else:
    #             # distance = np.linalg.norm(self.pos - np.array(result[3]), ord=2)
    #             self.radia_results[index] = 1
    #     # # 染色前清楚标记
    #     # p.removeAllUserDebugItems()
    #     # # 根据results结果给激光染色
    #     # for index, result in enumerate(results):
    #     #     if result[0] == -1:
    #     #         p.addUserDebugLine(rayFroms[index], rayTos[index], missRayColor)
    #     #         # print('safe----')
    #     #     else:
    #     #         p.addUserDebugLine(rayFroms[index], rayTos[index], hitRayColor)
    #     #         # print('hit----')

    def _radia(self):
        rayLength = 0.8  # 激光长度
        rayNum = 16  # 激光数量

        # 获取机器人的位置和旋转角度
        begins, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[0], physicsClientId=self.CLIENT)
        # _, self.rpy = p.getBasePositionAndOrientation(self.DRONE_IDS[0], physicsClientId=self.CLIENT)
        theta = self.rpy[0, :][2]

        # 初始化保存距离的列表
        self.radia_results = []

        # 发射激光并检测碰撞
        for i in range(rayNum):
            # 计算每个射线的角度
            # angle = theta + (math.pi / (rayNum-1)) * i - (math.pi / 2)  # 从-90度开始扫描，所以要减去90度的角度
            # angle = theta
            angle = 2 * math.pi * float(i) / rayNum + theta

            # 计算激光的起点和终点
            rayFrom = begins
            rayTo = [
                begins[0] + rayLength * math.cos(angle),
                begins[1] + rayLength * math.sin(angle),
                begins[2]
            ]

            # 检测激光与环境中物体的碰撞
            result = p.rayTest(rayFrom, rayTo, physicsClientId=self.CLIENT)

            # 如果没有碰撞，将距离设为激光的最大长度
            if result[0][0] == -1:
                distance = rayLength
                # p.addUserDebugLine(rayFrom, rayTo, [0, 1, 0], lifeTime=0.2)
            else:
                distance = min(result[0][2], rayLength)  # 获取与物体碰撞的距离
                # p.addUserDebugLine(rayFrom, rayTo, [1, 0, 0], lifeTime=0.2)

            # 将距离保存到列表中
            self.radia_results.append(distance)


    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        # TODO : initialize random number generator with seed

        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        # self._radia()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        # self.goal1_flag_changed = False
        self.goal1_flag = False
        self.goal2_flag = False
        self.goal3_flag = False

        return initial_obs, initial_info

    def step(self,
             action
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeTerminated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is trunacted, always false.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                     height=self.VID_HEIGHT,
                                                     shadow=1,
                                                     viewMatrix=self.CAM_VIEW,
                                                     projectionMatrix=self.CAM_PRO,
                                                     renderer=p.ER_TINY_RENDERER,
                                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                     physicsClientId=self.CLIENT
                                                     )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
            if self.VISION_ATTR:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                    #### Printing observation to PNG frames example ############
                    self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                    img_input=self.rgb[i],
                                    path=self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/",
                                    frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                    )
        #### Read the GUI's input parameters #######################
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter%(self.PYB_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = [p.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                          physicsClientId=self.CLIENT
                                                          ) for i in range(self.NUM_DRONES)]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.PYB_STEPS_PER_CTRL):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            for i in range (self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        # self._radia()
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        return obs, reward, terminated, truncated, info