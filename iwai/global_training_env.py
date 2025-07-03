import os

import pandas as pd

import utils
from agent.components.es_registry import ServiceType
from iwai.dqn_trainer import QR_DATA_QUALITY_STEP, CV_DATA_QUALITY_STEP
from iwai.lgbn_training_env import LGBNTrainingEnv
from iwai.proj_types import ESServiceAction

MAX_CORES = int(utils.get_env_param("MAX_CORES", 8))

"""
Environment you can "sample" from. Gymnasium compatible
"""


# INVALID_ACTION_PUNISHMENT = 5

class GlobalTrainingEnv:
    def __init__(
            self, env_qr: LGBNTrainingEnv, env_cv: LGBNTrainingEnv, max_cores=MAX_CORES
    ):
        self.env_qr = env_qr
        self.env_cv = env_cv
        self.max_cores = max_cores

    def reset(self):
        state_qr, _ = self.env_qr.reset()
        state_cv, _ = self.env_cv.reset()
        total_used_cores_after = state_qr.cores + state_cv.cores
        free_cores = self.max_cores - total_used_cores_after

        state_qr = state_qr._replace(free_cores=free_cores)
        state_cv = state_cv._replace(free_cores=free_cores)

        return state_qr, state_cv

    def step(self, action_qr: ESServiceAction, action_cv: ESServiceAction):
        # Execute both actions, but apply shared resource logic
        # total_used_cores_before = self.env_qr.state.cores + self.env_cv.state.cores

        old_state_qr = self.env_qr.state
        old_state_cv = self.env_cv.state

        total_used_cores_before = old_state_qr.cores + old_state_cv.cores
        free_cores_before = self.max_cores - total_used_cores_before

        # Apply actions
        self.env_qr.state = self.env_qr.state._replace(free_cores=free_cores_before)
        next_state_qr, reward_qr, done_qr, _, _ = self.env_qr.step(action_qr)
        self.env_cv.state = self.env_cv.state._replace(free_cores=next_state_qr.free_cores)
        next_state_cv, reward_cv, done_cv, _, _ = self.env_cv.step(action_cv)

        done = done_qr or done_cv

        joint_reward = reward_qr + reward_cv

        return (next_state_qr, next_state_cv), joint_reward, done


if __name__ == "__main__":
    ROOT = os.path.dirname(__file__)
    df = pd.read_csv(ROOT + "/../share/metrics/LGBN.csv")

    env_qr = LGBNTrainingEnv(ServiceType.QR, step_data_quality=QR_DATA_QUALITY_STEP)
    env_qr.reload_lgbn_model(df)

    env_cv = LGBNTrainingEnv(ServiceType.CV, step_data_quality=CV_DATA_QUALITY_STEP)
    env_cv.reload_lgbn_model(df)

    # Wrap in joint environment
    joint_env = GlobalTrainingEnv(env_qr, env_cv, max_cores=8)
    joint_env.reset()

    for _ in range(10):
        print(joint_env.step(ESServiceAction.INC_CORES, ESServiceAction.DEC_CORES))
