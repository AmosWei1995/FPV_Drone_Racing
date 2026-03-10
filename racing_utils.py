import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RacingWrapper(gym.Wrapper):
    """
    强化竞速任务包装器 (收敛优化版)：
    1. 大幅缩减奖励量级，防止梯度爆炸。
    2. 优化高度和生存奖励。
    3. 修复动作空间映射。
    """
    def __init__(self, env):
        super().__init__(env)
        self.gate_passed = False
        self.step_count = 0
        self.last_action = np.zeros(4)
        self.frame_skip = 1 
        
        # 统计项
        self.accumulated_info = {
            'reward_progress': 0,
            'reward_speed': 0,
            'reward_gate': 0,
            'reward_finish': 0,
            'reward_height': 0,
            'reward_survival': 0,
            'penalty_crash': 0,
            'penalty_tilt': 0,
            'penalty_smooth': 0,
            'penalty_track': 0
        }
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        low = np.concatenate([self.env.observation_space.low.flatten(), np.array([-10, -10, -10, 0])])
        high = np.concatenate([self.env.observation_space.high.flatten(), np.array([10, 10, 10, 1])])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.gate_passed = False
        self.step_count = 0
        self.last_action = np.zeros(4)
        self.accumulated_info = {k: 0 for k in self.accumulated_info}
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment_obs(obs), info

    def step(self, action):
        accumulated_reward = 0
        final_terminated = False
        final_truncated = False
        
        action_flat = action.flatten()
        smoothed_action = 0.8 * self.last_action + 0.2 * action_flat
        p_smooth = -0.01 * np.linalg.norm(action_flat - self.last_action) # 降低惩罚量级
        self.last_action = smoothed_action
        
        action_to_env = smoothed_action.reshape(1, 4)

        for _ in range(self.frame_skip):
            self.step_count += 1
            state_before = self.env._getDroneStateVector(0)
            pos_before = state_before[0:3]
            
            obs, _, terminated, truncated, info = self.env.step(action_to_env)
            
            state_after = self.env._getDroneStateVector(0)
            pos_after = state_after[0:3]
            rpy_after = state_after[7:10]
            vel_after = state_after[10:13]
            
            # --- 核心奖励缩放 ---
            # 1. 生存奖励：从 0.5 降到 0.05
            r_survival = 0.05
            
            # 2. 高度奖励：从 2.0 降到 0.2
            target_h = 0.5 # 与门中心高度对齐
            dist_h = abs(pos_after[2] - target_h)
            r_height = 0.2 * np.exp(-(dist_h**2) / 0.1) - 0.05
            
            # 3. 进度奖励：从 800.0 大幅降到 20.0 (最关键)
            target = self.env.FINISH_POS if self.gate_passed else self.env.GATE_POS
            origin = self.env.GATE_POS if self.gate_passed else self.env.START_POS
            
            dist_before = np.linalg.norm(target - pos_before)
            dist_after = np.linalg.norm(target - pos_after)
            r_progress = (dist_before - dist_after) * 20.0 
            
            # 4. 速度奖励：从 3.0 降到 0.2 (平衡安全性)
            target_dir = (target - pos_after) / (dist_after + 1e-6)
            speed_towards_target = np.dot(vel_after, target_dir)
            r_speed = max(0, speed_towards_target) * 0.2
            
            # 5. 偏离航线惩罚：从 -0.5 增加到 -2.0 (强制走直线)
            line_vec = target - origin
            line_len = np.linalg.norm(line_vec)
            if line_len > 1e-6:
                line_unit = line_vec / line_len
                vec_pos = pos_after - origin
                off_track_vec = vec_pos - np.dot(vec_pos, line_unit) * line_unit
                p_track = -2.0 * np.linalg.norm(off_track_vec)
            else:
                p_track = 0

            # 6. 倾斜惩罚：从 -1.0 降到 -0.1
            tilt = np.linalg.norm(rpy_after[0:2])
            p_tilt = -0.1 * tilt
            
            # 7. 事件奖励：从 500/3000 降到 50/200
            r_gate = 0
            r_finish = 0
            p_crash = 0
            
            info['terminal_reason'] = "none"
            
            if not self.gate_passed:
                if np.linalg.norm(pos_after - self.env.GATE_POS) < 0.6:
                    self.gate_passed = True
                    r_gate = 50.0 # 奖励缩减
                    print("\n>>> [成功] 穿过第一个门！")

            if self.gate_passed and np.linalg.norm(pos_after - self.env.FINISH_POS) < 0.8:
                r_finish = 200.0 # 奖励缩减
                terminated = True 
                info['terminal_reason'] = "success"
                print("\n>>> [成功] 到达终点！")

            # 8. 终止与惩罚：从 -10.0 增加到 -100.0 (重罚失败)
            if self.step_count > 30:
                if pos_after[2] < 0.05: 
                    p_crash = -100.0 
                    terminated = True 
                    info['terminal_reason'] = "crash_ground"
                
                if tilt > np.deg2rad(80): 
                    p_crash = -100.0
                    terminated = True
                    info['terminal_reason'] = "crash_tilt"

            if pos_after[2] > 3.8:
                p_crash = -50.0
                terminated = True
                info['terminal_reason'] = "crash_ceiling"
                
            if truncated or self.step_count > 10000:
                terminated = True
                info['terminal_reason'] = "timeout"

            step_r = r_progress + r_speed + r_gate + r_finish + r_height + r_survival + p_tilt + p_crash + p_track + p_smooth
            accumulated_reward += step_r
            
            # 更新统计
            self.accumulated_info['reward_progress'] += r_progress
            self.accumulated_info['reward_speed'] += r_speed
            self.accumulated_info['reward_gate'] += r_gate
            self.accumulated_info['reward_finish'] += r_finish
            self.accumulated_info['reward_height'] += r_height
            self.accumulated_info['reward_survival'] += r_survival
            self.accumulated_info['penalty_crash'] += p_crash
            self.accumulated_info['penalty_tilt'] += p_tilt
            self.accumulated_info['penalty_smooth'] += p_smooth
            self.accumulated_info['penalty_track'] += p_track
            
            if terminated:
                final_terminated = True
                info['run_stats'] = self.accumulated_info
                info['episode_steps'] = self.step_count
                break 
            
        return self._augment_obs(obs), accumulated_reward, final_terminated, truncated, info

    def _augment_obs(self, obs):
        state = self.env._getDroneStateVector(0)
        pos = state[0:3]
        target = self.env.FINISH_POS if self.gate_passed else self.env.GATE_POS
        rel_pos = target - pos
        gate_flag = np.array([1.0 if self.gate_passed else 0.0])
        return np.concatenate([obs.flatten(), rel_pos, gate_flag]).astype('float32')
