import os
import time
import warnings
from datetime import datetime
from collections import Counter, deque
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

from custom_racing_env import RacingGateAviary
from racing_utils import RacingWrapper
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

class StatsCallback(BaseCallback):
    """
    自定义统计回调：统计失败原因和平均步长。
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.reasons = deque(maxlen=100) # 记录最近 100 个回合的死因
        self.lengths = deque(maxlen=100) # 记录最近 100 个回合的步长

    def _on_step(self) -> bool:
        # 检查每个并行环境是否在这一步结束了
        for info in self.locals['infos']:
            if 'terminal_reason' in info and info['terminal_reason'] != "none":
                self.reasons.append(info['terminal_reason'])
                self.lengths.append(info.get('episode_steps', 0))
        
        # 每 2000 步打印一次统计
        if self.n_calls % 2000 == 0:
            if len(self.reasons) > 0:
                counts = Counter(self.reasons)
                avg_len = np.mean(self.lengths)
                print(f"\n>>> [训练统计] 过去 100 回合分析:")
                print(f"    平均寿命: {avg_len:.1f} 步")
                for reason, count in counts.items():
                    print(f"    原因 {reason:15}: {count}%")
                print("-" * 30)
        return True

def make_env(gui=False):
    def _init():
        env = RacingGateAviary(
            gui=gui,
            obs=ObservationType.KIN,
            act=ActionType.VEL
        )
        env = RacingWrapper(env)
        return env
    return _init

def train():
    n_envs = 25
    print(f"\n[启动诊断训练] 正在开启 {n_envs} 个环境...")
    
    train_env = SubprocVecEnv([make_env(gui=False) for i in range(n_envs)])
    eval_env = make_env(gui=False)()

    output_dir = "results/racing_rl_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu"
    )
    # model = PPO.load("results/racing_rl_20260309_204430/best_model.zip", env=train_env, device="cpu") # 加载预训练模型继续训练 (如果有的话)

    # 组合回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    stats_callback = StatsCallback()

    total_timesteps = 10000000
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, stats_callback])

    model_path = os.path.join(output_dir, "final_racing_model.zip")
    model.save(model_path)
    return model_path

if __name__ == "__main__":
    m_path = train()
    print(f"\n[任务结束] 模型存放在: {m_path}")
