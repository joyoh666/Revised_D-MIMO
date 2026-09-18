import unittest

import numpy as np
import pandas as pd
import torch

from models.RUscheduler.DDQNscheduler import GRUQNetwork
from models.RUscheduler.test_environment import (
    EnvConfig as TestEnvConfig,
    RUWindowEnv as TestRUWindowEnv,
)
from models.RUscheduler.train_environment import (
    EnvConfig as TrainEnvConfig,
    RUWindowEnv as TrainRUWindowEnv,
)


class _SyntheticStore:
    def __init__(self):
        self._gains = (
            np.arange(80 * 6, dtype=np.float32).reshape(80, 6) + 1.0
        ) / 100.0
        self.manifest = pd.DataFrame({"dataset": ["synthetic"]})

    def get_window_gains(self, idx: int) -> np.ndarray:
        if int(idx) != 0:
            raise IndexError(idx)
        return self._gains.copy()


class RUSchedulerEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.store = _SyntheticStore()

    def test_train_environment_uses_training_reward(self):
        cfg = TrainEnvConfig(
            seg_len_ticks=20,
            handover_penalty=0.5,
            inter_ru_phase_noise_std=0.0,
            two_ru_penalty=2.0,
            noise_power=1e-3,
            reward_snr_db=10.0,
        )
        env = TrainRUWindowEnv(self.store, [0], cfg, training=True)
        env.reset(ep_idx=0)
        env._conn_state = 0
        env._conn_hist = [0]

        state_power = env._all_state_power_from_segment(1)
        _, reward, _, info = env.step(3)

        expected = float(state_power[3]) - 0.5 - 2.0 - 3.0
        self.assertAlmostEqual(reward, expected)
        self.assertEqual(info["handover_attempt"], 1)

    def test_test_environment_uses_test_reward(self):
        cfg = TestEnvConfig(
            seg_len_ticks=20,
            handover_penalty=0.5,
            inter_ru_phase_noise_std=0.0,
            noise_power=1e-3,
        )
        env = TestRUWindowEnv(self.store, cfg)
        env.reset(ep_idx=0)
        env._conn_state = 0
        env._conn_hist = [0]

        state_power = env.all_state_power_segment(1)
        _, reward, _, info = env.step(3)

        expected = float(state_power[3]) - 0.5
        self.assertAlmostEqual(reward, expected)
        self.assertEqual(info["handover_attempt"], 1)

    def test_gru_hidden_state_api_matches_callers(self):
        model = GRUQNetwork(obs_dim=15, hidden_size=8, n_actions=6)
        hidden = model.init_hidden(batch=4, device=torch.device("cpu"))
        self.assertEqual(tuple(hidden.shape), (4, 8))


if __name__ == "__main__":
    unittest.main()
