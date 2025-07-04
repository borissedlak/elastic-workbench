import logging
import os

import pandas as pd
from matplotlib import pyplot as plt

import utils
from agent.components.es_registry import ServiceType
from iwai.dqn_trainer import (
    DQN,
    STATE_DIM,
    ACTION_DIM_QR,
    ACTION_DIM_CV,
    QR_DATA_QUALITY_STEP,
    CV_DATA_QUALITY_STEP,
    NO_EPISODES,
    EPISODE_LENGTH, PC_DATA_QUALITY_STEP, ACTION_DIM_PC,
)
from iwai.global_training_env import GlobalTrainingEnv
from iwai.lgbn_training_env import LGBNTrainingEnv
from iwai.proj_types import ESServiceAction

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("multiscale")
logger.setLevel(logging.DEBUG)

# Add handler only if it doesn't exist yet
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)


class JointDQNTrainer:
    def __init__(self, dqn_qr: DQN, dqn_cv: DQN, dqn_pc: DQN, joint_env: GlobalTrainingEnv):

        self.dqn_qr = dqn_qr
        self.dqn_cv = dqn_cv
        self.dqn_pc = dqn_pc
        self.env = joint_env
        self.max_episodes = NO_EPISODES
        self.episode_length = EPISODE_LENGTH

    @utils.print_execution_time
    def train(self):
        score_list = []
        for ep in range(self.max_episodes):
            state_qr, state_cv, state_pc = self.env.reset()
            ep_score = 0

            for t in range(self.episode_length):
                # Choose actions
                action_qr = self.dqn_qr.choose_action(state_qr.to_np_ndarray(True))
                action_cv = self.dqn_cv.choose_action(state_cv.to_np_ndarray(True))
                action_pc = self.dqn_pc.choose_action(state_pc.to_np_ndarray(True))

                # Step joint env
                (next_state_qr, next_state_cv, next_state_pc), reward, done = self.env.step(
                    ESServiceAction(action_qr), ESServiceAction(action_cv), ESServiceAction(action_pc)
                )

                # Store transitions in each agent's buffer
                self.dqn_qr.memory.put(
                    (
                        state_qr.to_np_ndarray(True),
                        action_qr,
                        reward,
                        next_state_qr.to_np_ndarray(True),
                        done,
                    )
                )
                self.dqn_cv.memory.put(
                    (
                        state_cv.to_np_ndarray(True),
                        action_cv,
                        reward,
                        next_state_cv.to_np_ndarray(True),
                        done,
                    )
                )
                self.dqn_pc.memory.put(
                    (
                        state_pc.to_np_ndarray(True),
                        action_pc,
                        reward,
                        next_state_pc.to_np_ndarray(True),
                        done,
                    )
                )

                state_qr, state_cv, state_pc = next_state_qr, next_state_cv, next_state_pc
                ep_score += reward

                # Train if enough samples
                if self.dqn_qr.memory.size() > self.dqn_qr.batch_size:
                    self.dqn_qr.train_batch()
                if self.dqn_cv.memory.size() > self.dqn_cv.batch_size:
                    self.dqn_cv.train_batch()
                if self.dqn_pc.memory.size() > self.dqn_pc.batch_size:
                    self.dqn_pc.train_batch()

                if done:
                    break

            # Decay epsilon
            self.dqn_qr.epsilon = max(
                self.dqn_qr.epsilon * self.dqn_qr.epsilon_decay, self.dqn_qr.epsilon_min
            )
            self.dqn_cv.epsilon = max(
                self.dqn_cv.epsilon * self.dqn_cv.epsilon_decay, self.dqn_cv.epsilon_min
            )
            self.dqn_pc.epsilon = max(
                self.dqn_pc.epsilon * self.dqn_pc.epsilon_decay, self.dqn_pc.epsilon_min
            )

            score_list.append(ep_score)
            logger.info(f"Final state for QR Env: {self.env.env_qr.state}")
            logger.info(f"Final state for CV Env: {self.env.env_cv.state}")
            logger.info(f"Final state for PC Env: {self.env.env_pc.state}")
            logger.info(
                f"[EP {ep + 1:3d}] Score: {ep_score:.2f} | Epsilon QR: {self.dqn_qr.epsilon:.4f}, CV: {self.dqn_cv.epsilon:.4f}, PC: {self.dqn_pc.epsilon:.4f}"
            )

        plt.plot(score_list)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Joint DQN Training")
        plt.show()

        # Save networks
        self.dqn_qr.store_dqn_as_file(suffix="QR_joint")
        self.dqn_cv.store_dqn_as_file(suffix="CV_joint")
        self.dqn_pc.store_dqn_as_file(suffix="PC_joint")


def train_joint_q_networks(nn_folder=None):
    df = pd.read_csv(ROOT + "/../share/metrics/LGBN_v2.csv")

    env_qr = LGBNTrainingEnv(ServiceType.QR, step_data_quality=QR_DATA_QUALITY_STEP)
    env_qr.reload_lgbn_model(df)

    env_cv = LGBNTrainingEnv(ServiceType.CV, step_data_quality=CV_DATA_QUALITY_STEP)
    env_cv.reload_lgbn_model(df)

    env_pc = LGBNTrainingEnv(ServiceType.PC, step_data_quality=PC_DATA_QUALITY_STEP)
    env_pc.reload_lgbn_model(df)

    # Wrap in joint environment
    joint_env = GlobalTrainingEnv(env_qr, env_cv, env_pc, max_cores=8)

    # Create DQNs
    if nn_folder is None:
        dqn_qr = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_QR)
        dqn_cv = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_CV)
        dqn_pc = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_PC)
    else:
        dqn_qr = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_QR, nn_folder=nn_folder)
        dqn_cv = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_CV, nn_folder=nn_folder)
        dqn_pc = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_CV, nn_folder=nn_folder)

    # Train jointly
    trainer = JointDQNTrainer(dqn_qr, dqn_cv, dqn_pc, joint_env)
    trainer.train()


if __name__ == "__main__":
    train_joint_q_networks()
