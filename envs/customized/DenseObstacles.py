import numpy as np
import pybullet
import config
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from envs.customized.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary


class DenseObstacles(BaseSingleAgentAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,  # TODO: if you want to set a fixed start pos, set as there, format as[[x, y, z]]
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=True,  # NOTE: There need to be set as False if you want to gain time
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.VEL,
                 use_random_start: bool = True,  # TODO
                 use_random_goal: bool = True
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

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
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        # TODO: Add Obstacle
        goal : list of (x, y, z), i think it x and y in workspace, and z can set as constant.
        """
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         use_random_start=use_random_start,
                         use_random_goal=use_random_goal
                         )
        # self.goal1_flag_changed = False
        self.goal1_flag = False
        self.goal2_flag = False
        self.goal3_flag = False
        self.prev_state = [0, 0, 0]


    def _checkCollision(self):

        collision2obstacles = []
        # check collision with obstalce，这种检测碰撞不一定靠谱，考虑改成欧几里得距离
        for i in range(len(self.obstacleId)):
            close_info = pybullet.getClosestPoints(self.DRONE_IDS[0], self.obstacleId[i], distance=0.06,
                                                   physicsClientId=self.CLIENT)
            if close_info:
                collision2obstacles.append(True)  # make collision
            else:
                collision2obstacles.append(False)  # no collision

        return any(collision2obstacles)

    def _computeReward(self):
        """Computes the current reward value.
        Returns
        -------
        float
            The reward.
        """
        state = self._getDroneStateVector(0)
        collisionFlag = self._checkCollision()
        checkgoal1 = self._checkgoal1()
        checkgoal2 = self._checkgoal2()
        checkgoal3 = self._checkgoal3()
        # pybullet.resetDebugVisualizerCamera(
        #     cameraDistance=4,
        #     cameraYaw=0,
        #     cameraPitch=-115,
        #     cameraTargetPosition=state[0:3],
        #     physicsClientId=0
        # )

        # distance_reward = max(0, 5 - np.linalg.norm(np.array(self.goalPos) - state[0:3]))
        # if (2000*(np.linalg.norm(np.array(self.goalPos) - self.prev_state[0:3]) - (np.linalg.norm(np.array(self.goalPos1) - state[0:3])))) < 0.5:
        #     distance_reward = distance_reward - 0.8
        # print("distance_reward", distance_reward)
        if config.testmode:
            distance_reward = max(0, 4.5 - np.linalg.norm(np.array(self.goalPos) - state[0:3]))

            distance_reward = max(0, 2 - np.linalg.norm(np.array(self.goalPos1) - state[0:3]))
            if (2000*(np.linalg.norm(np.array(self.goalPos1) - self.prev_state[0:3]) - (np.linalg.norm(np.array(self.goalPos1) - state[0:3])))) < 0.5:
                distance_reward = distance_reward - 0.4
            if checkgoal1:
                distance_reward = 4.9 - 2*(np.linalg.norm(np.array(self.goalPos2) - state[0:3]))
                if (2000*(np.linalg.norm(np.array(self.goalPos2) - self.prev_state[0:3]) - (np.linalg.norm(np.array(self.goalPos2) - state[0:3])))) < 0.8:
                    distance_reward = distance_reward - 0.8

            if checkgoal2:
                distance_reward = 6.8 - (np.linalg.norm(np.array(self.goalPos3) - state[0:3]))
                if (2000*(np.linalg.norm(np.array(self.goalPos3) - self.prev_state[0:3]) - (np.linalg.norm(np.array(self.goalPos3) - state[0:3])))) < 0.8:
                    distance_reward = distance_reward - 1

            if checkgoal3:
                distance_reward = 10.3 - (np.linalg.norm(np.array(self.goalPos) - state[0:3]))
                if (2000*(np.linalg.norm(np.array(self.goalPos) - self.prev_state[0:3]) - (np.linalg.norm(np.array(self.goalPos) - state[0:3])))) < 0.8:
                    distance_reward = distance_reward - 2.5
                if (2000 * (np.linalg.norm(np.array(self.goalPos) - self.prev_state[0:3]) - (np.linalg.norm(np.array(self.goalPos) - state[0:3])))) < 0.2:
                    distance_reward = distance_reward - 1
        else:
            distance_reward = max(0, 4.5 - np.linalg.norm(np.array(self.goalPos) - state[0:3]))

        self.prev_state = state[0:3]
        # print("checkgoal2", checkgoal2)
        if collisionFlag:
            collision_penalty = -10
        else:
            collision_penalty = 0
        if np.linalg.norm(np.array(self.goalPos) - state[0:3]) < 0.1:
            signal_reward = 50000
        else:
            signal_reward = 0
        reward = signal_reward + distance_reward + collision_penalty
        if not config.testmode:
            reward = distance_reward + collision_penalty
        return reward

    ################################################################################
    def _checkgoal1(self):
        state = self._getDroneStateVector(0)
        if not self.goal1_flag:
            if np.linalg.norm(np.array(self.goalPos1) - state[0:3]) < 0.2:
                self.goal1_flag = True
                return self.goal1_flag
        return self.goal1_flag

    def _checkgoal2(self):
        state = self._getDroneStateVector(0)
        if not self.goal2_flag:
            if np.linalg.norm(np.array(self.goalPos2) - state[0:3]) < 0.1:
                self.goal2_flag = True
                return self.goal2_flag
        return self.goal2_flag

    def _checkgoal3(self):
        state = self._getDroneStateVector(0)
        if not self.goal3_flag:
            if np.linalg.norm(np.array(self.goalPos3) - state[0:3]) < 0.1:
                self.goal3_flag = True
                return self.goal3_flag
        return self.goal3_flag

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        collisionFlag = self._checkCollision()
        if np.linalg.norm(np.array(self.goalPos) - state[0:3]) < 0.1:
            return True
        elif collisionFlag:
            return True
        elif state[2] > 1:
            return True
        else:
            return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        if self.step_counter > 8000:
            return True
        else:
            return False


    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years

