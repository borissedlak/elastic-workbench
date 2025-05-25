import utils
from iwai.LGBN_Training_Env import LGBN_Training_Env, INVALID_ACTION_PUNISHMENT

MAX_CORES = int(utils.get_env_param('MAX_CORES', 8))


class GlobalTrainingEnv:
    def __init__(self, env_qr: LGBN_Training_Env, env_cv: LGBN_Training_Env, max_cores=MAX_CORES):
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

    def step(self, action_qr, action_cv):
        # Execute both actions, but apply shared resource logic
        # total_used_cores_before = self.env_qr.state.cores + self.env_cv.state.cores

        # Apply actions
        next_state_qr, reward_qr, done_qr, _, _ = self.env_qr.step(action_qr)
        next_state_cv, reward_cv, done_cv, _, _ = self.env_cv.step(action_cv)

        total_used_cores_after = next_state_qr.cores + next_state_cv.cores
        free_cores = self.max_cores - total_used_cores_after
        next_state_qr = next_state_qr._replace(free_cores=free_cores)
        next_state_cv = next_state_cv._replace(free_cores=free_cores)

        overuse = total_used_cores_after > self.max_cores

        done = done_qr or done_cv
        penalty = 0
        if overuse:  # Shared penalty if resource overuse occurred
            penalty = INVALID_ACTION_PUNISHMENT
            done = True

        joint_reward = reward_qr + reward_cv + penalty

        return (next_state_qr, next_state_cv), joint_reward, done
