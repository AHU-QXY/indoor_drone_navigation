from argparse import Namespace
from envs.drones_vec_env import DummyVecEnv_Drones, SubprocVecEnv_Drones

REGISTRY_VEC_ENV = {
    "Dummy_Drone": DummyVecEnv_Drones,
    # multiprocess #
    "Subproc_Drone": SubprocVecEnv_Drones,
}


def make_envs(config: Namespace):
    def _thunk():
        from envs.drones_env import Drones_Env
        env = Drones_Env(config)

        return env

    if config.vectorize in REGISTRY_VEC_ENV.keys():
        return REGISTRY_VEC_ENV[config.vectorize]([_thunk for _ in range(config.parallels)])
    elif config.vectorize == "NOREQUIRED":
        return _thunk()
    else:
        raise NotImplementedError

