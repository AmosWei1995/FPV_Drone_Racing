import time
import numpy as np
import pybullet as p
from custom_racing_env import RacingGateAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

def run():
    """
    通过键盘实时操控无人机的虚拟目标点。
    """
    # 1. 创建环境 (必须开启 GUI)
    env = RacingGateAviary(gui=True)
    
    # 2. 初始化 PID 控制器
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

    # 3. 重置环境并获取初始位置
    env.reset()
    target_pos = env.START_POS.copy()
    target_yaw = 0.0
    
    print("\n" + "="*30)
    print("   键盘操控模式已启动")
    print("="*30)
    print(" W / S : 前进 / 后退")
    print(" A / D : 向左 / 向右")
    print(" ↑ / ↓ : 升高 / 降低")
    print(" Q / E : 旋转 (Yaw)")
    print(" ESC   : 退出程序")
    print("="*30)
    print("\n提示：请先点击一下 PyBullet 的 3D 仿真窗口，然后再使用键盘。")

    # 仿真循环
    for i in range(20000): 
        # 获取键盘事件
        events = p.getKeyboardEvents(physicsClientId=env.CLIENT)
        
        # 定义移动步长
        step = 0.05
        yaw_step = 0.05
        
        # 键盘映射 (ASCII 码或 PyBullet 常量)
        # W=119, S=115, A=97, D=100, Q=113, E=101
        # Up=65297, Down=65298, ESC=27
        
        if 119 in events and events[119] & p.KEY_IS_DOWN: # W
            target_pos[0] += step
        if 115 in events and events[115] & p.KEY_IS_DOWN: # S
            target_pos[0] -= step
        if 97 in events and events[97] & p.KEY_IS_DOWN: # A
            target_pos[1] += step
        if 100 in events and events[100] & p.KEY_IS_DOWN: # D
            target_pos[1] -= step
        if 65297 in events and events[65297] & p.KEY_IS_DOWN: # Up
            target_pos[2] += step
        if 65298 in events and events[65298] & p.KEY_IS_DOWN: # Down
            target_pos[2] -= (step if target_pos[2] > 0.05 else 0) # 防止切地
        if 113 in events and events[113] & p.KEY_IS_DOWN: # Q
            target_yaw += yaw_step
        if 101 in events and events[101] & p.KEY_IS_DOWN: # E
            target_yaw -= yaw_step
            
        if 27 in events: # ESC
            break

        # 获取当前状态并计算 PID
        state = env._getDroneStateVector(0)
        action_rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=target_pos,
            target_rpy=np.array([0, 0, target_yaw])
        )

        # 转换为环境接收的归一化动作
        action_normalized = (action_rpm / env.HOVER_RPM - 1) / 0.05
        action_normalized = np.clip(action_normalized, -1, 1)
        
        # 步进
        obs, reward, terminated, truncated, info = env.step(action_normalized.reshape(1, 4))

        if terminated or truncated:
            env.reset()
            target_pos = env.START_POS.copy()
            target_yaw = 0.0
            
        time.sleep(env.CTRL_TIMESTEP)

    env.close()

if __name__ == "__main__":
    run()
