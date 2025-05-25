import utils
from iwai.LGBN_Training_Env import LGBN_Training_Env

MAX_CORES = int(utils.get_env_param('MAX_CORES', 8))

class JointTrainingEnv:
    def __init__(self, env_qr: LGBN_Training_Env, env_cv: LGBN_Training_Env, max_cores=MAX_CORES):
        self.env_qr = env_qr
        self.env_cv = env_cv
        self.max_cores = max_cores

    def reset(self):
        state_qr, _ = self.env_qr.reset()
        state_cv, _ = self.env_cv.reset()
        return state_qr, state_cv

    def step(self, action_qr, action_cv):
        # Execute both actions, but apply shared resource logic
        # total_used_cores_before = self.env_qr.state.cores + self.env_cv.state.cores

        # Apply actions
        next_state_qr, reward_qr, done_qr, _, _ = self.env_qr.step(action_qr)
        next_state_cv, reward_cv, done_cv, _, _ = self.env_cv.step(action_cv)

        total_used_cores_after = next_state_qr.cores + next_state_cv.cores
        overuse = max(0, total_used_cores_after - self.max_cores)

        # Shared penalty if resource overuse occurred
        penalty = -1.0 * overuse  # or more complex penalty function
        joint_reward = reward_qr + reward_cv + penalty

        done = done_qr or done_cv
        return (next_state_qr, next_state_cv), joint_reward, done
