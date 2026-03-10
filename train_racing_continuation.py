import os
import time
import warnings
import csv
from datetime import datetime
from collections import Counter, deque
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

from custom_racing_env import RacingGateAviary
from racing_utils import RacingWrapper
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

class SuccessRateCallback(BaseCallback):
    """
    成功率停止回调：
    统计最近 100 个回合的成功率，达到目标（如 80%）时停止训练。
    同时保存奖励最高和耗时最短的模型。
    """
    def __init__(self, target_rate=0.8, window_size=100, log_path="results/flight_history.csv", verbose=0):
        super().__init__(verbose)
        self.target_rate = target_rate
        self.window_size = window_size
        self.log_path = log_path
        self.pyb_freq = 240.0 # 物理模拟频率
        
        # 使用 deque 记录最近回合的成功情况 (1 为成功，0 为失败)
        self.success_window = deque(maxlen=window_size)
        self.total_successes = 0
        self.best_success_reward = -np.inf
        self.best_success_time = np.inf # 记录最短耗时 (秒)
        self.last_save_time = 0 # 保存限流
        
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        file_exists = os.path.isfile(self.log_path)
        file_empty = os.path.getsize(self.log_path) == 0 if file_exists else True
        
        if not file_exists or file_empty:
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'Steps', 'Time_s', 'Reason', 'Total_R', 
                    'Progress_R', 'Speed_R', 'Gate_R', 'Finish_R', 'Height_R', 'Survival_R', 
                    'Crash_P', 'Tilt_P', 'Success_Count', 'Success_Rate'
                ])

    def _on_step(self) -> bool:
        output_dir = os.path.dirname(self.log_path)
        current_real_time = time.time()
        
        for info in self.locals['infos']:
            reason = info.get('terminal_reason', 'none')
            if reason != "none":
                steps = info.get('episode_steps', 0)
                time_s = steps / self.pyb_freq # 换算物理耗时
                stats = info.get('run_stats', {})
                total_r = info.get('total_reward', 0)
                
                # 记录成功情况
                is_success = 1 if reason == "success" else 0
                self.success_window.append(is_success)
                
                if is_success:
                    self.total_successes += 1
                    
                    # 1. 保存奖励最高的模型 (最佳综合表现)
                    if total_r > self.best_success_reward:
                        self.best_success_reward = total_r
                        self.model.save(os.path.join(output_dir, "best_model.zip"))
                        print(f"\n[NEW BEST REWARD] 奖励刷新: {total_r:.2f}")

                    # 2. 保存耗时最短的模型 (最快模型)
                    if time_s < self.best_success_time:
                        self.best_success_time = time_s
                        self.model.save(os.path.join(output_dir, "fastest_model.zip"))
                        print(f"\n[NEW RECORD] 速度记录刷新: {time_s:.3f} 秒 ({steps} 步)")

                    # 3. 普通成功模型备份 (30秒限流)
                    if current_real_time - self.last_save_time > 30:
                        self.model.save(os.path.join(output_dir, "last_success_model.zip"))
                        self.last_save_time = current_real_time
                
                # 计算当前胜率
                current_rate = sum(self.success_window) / len(self.success_window) if len(self.success_window) > 0 else 0
                
                # 写入 CSV
                with open(self.log_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%H:%M:%S"),
                        steps,
                        f"{time_s:.3f}",
                        reason, 
                        f"{total_r:.2f}",
                        f"{stats.get('reward_progress', 0):.2f}",
                        f"{stats.get('reward_speed', 0):.2f}",
                        f"{stats.get('reward_gate', 0):.2f}",
                        f"{stats.get('reward_finish', 0):.2f}",
                        f"{stats.get('reward_height', 0):.2f}",
                        f"{stats.get('reward_survival', 0):.2f}",
                        f"{stats.get('penalty_crash', 0):.2f}",
                        f"{stats.get('penalty_tilt', 0):.2f}",
                        self.total_successes,
                        f"{current_rate:.2%}"
                    ])
                
                if is_success:
                    print(f"\n[!!!] 成功！耗时: {time_s:.3f}s | 胜率: {current_rate:.2%} (窗口: {len(self.success_window)})")

        # 检查是否达到停止条件：窗口已满且胜率达标
        if len(self.success_window) >= self.window_size:
            current_rate = sum(self.success_window) / self.window_size
            if current_rate >= self.target_rate:
                print(f"\n[任务达成] 最近 {self.window_size} 回合胜率已达 {current_rate:.2%}，停止训练。")
                return False 
            
        return True

def make_env(gui=False):
    def _init():
        env = RacingGateAviary(gui=gui, obs=ObservationType.KIN, act=ActionType.VEL)
        env = RacingWrapper(env)
        return env
    return _init

def get_latest_best_model():
    results_dir = "results"
    subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if d.startswith("racing_rl_")]
    if not subdirs: return None
    latest_dir = max(subdirs, key=os.path.getmtime)
    # best_model = os.path.join(latest_dir, "best_model.zip")
    best_model = os.path.join(latest_dir, "fastest_model.zip")
    return best_model if os.path.exists(best_model) else None

def continue_training():
    model_path = get_latest_best_model()
    if not model_path: return

    print(f"\n[启动高稳健训练] 加载: {model_path}")
    n_envs = 10
    train_env = SubprocVecEnv([make_env(gui=False) for _ in range(n_envs)])
    eval_env = make_env(gui=False)()

    model = PPO.load(model_path, env=train_env, device="cpu", n_steps=2048)
    output_dir = os.path.dirname(model_path)
    
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=output_dir, log_path=output_dir, 
        eval_freq=max(1000, 10000 // n_envs), deterministic=True, render=False
    )
    
    log_file = os.path.join(output_dir, "continuation_log.csv")
    # 设置目标胜率为 80%
    success_callback = SuccessRateCallback(target_rate=0.9, window_size=1000, log_path=log_file)
    
    model.learn(
        total_timesteps=100000000, 
        callback=CallbackList([eval_callback, success_callback]), 
        reset_num_timesteps=False
    )

    model.save(os.path.join(output_dir, "best_model.zip"))
    model.save(os.path.join(output_dir, "high_reliability_model.zip"))
    print(f"\n[任务达成] 最终高可靠性模型已保存至: {output_dir}")

if __name__ == "__main__":
    continue_training()
