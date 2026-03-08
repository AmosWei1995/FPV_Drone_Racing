import time
import numpy as np
from custom_racing_env import RacingGateAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

def run():
    """
    使用平滑航点追踪策略的 PID 控制运行程序。
    """
    # 1. 创建环境 (开启 GUI)
    env = RacingGateAviary(gui=True)
    
    # 2. 初始化 PID 控制器
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

    # 3. 重置环境
    env.reset()
    
    # 航点列表：门 -> 终点
    waypoints = [
        env.GATE_POS,
        env.FINISH_POS
    ]
    wp_idx = 0
    
    print(f"\n[任务开始] 当前目标: 穿越门 {waypoints[0]}")

    # 仿真循环
    for i in range(5000): 
        # 获取无人机当前状态
        state = env._getDroneStateVector(0)
        curr_pos = state[0:3]
        
        # --- 核心改进：平滑目标追踪 ---
        # 不要直接瞄准远处的航点，而是沿着航点方向设置一个近处的“虚拟目标”
        final_target = waypoints[wp_idx]
        dir_vec = final_target - curr_pos
        dist_to_final = np.linalg.norm(dir_vec)

        # 每一帧的瞄准点距离当前位置不超过 0.2 米
        lookahead_dist = 0.4
        if dist_to_final > lookahead_dist:
            target_pos = curr_pos + (dir_vec / dist_to_final) * lookahead_dist
        else:
            target_pos = final_target

        # 检查是否真正接近了大航点以切换下一个
        if dist_to_final < 0.25:
            if wp_idx < len(waypoints) - 1:
                wp_idx += 1
                print(f"[航点达成] 切换目标: 终点 {waypoints[wp_idx]}")

        # 4. 计算 PID 要求的 RPM
        action_rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=target_pos, # 使用平滑后的虚拟目标
            target_rpy=np.array([0, 0, 0])
        )

        # 5. 转换为环境接收的归一化动作 [-1, 1] 并增加安全限幅
        action_normalized = (action_rpm / env.HOVER_RPM - 1) / 0.05
        action_normalized = np.clip(action_normalized, -1, 1)
        
        # 执行物理步进
        obs, reward, terminated, truncated, info = env.step(action_normalized.reshape(1, 4))

        if terminated:
            print("[任务成功] 无人机已抵达终点！")
            break
        if truncated:
            print("[任务失败] 无人机飞出边界或超时。")
            break
            
        # 控制仿真速度与真实时间同步
        time.sleep(env.CTRL_TIMESTEP)

    # env.close()

if __name__ == "__main__":
    run()
