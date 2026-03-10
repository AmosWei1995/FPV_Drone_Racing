import os
import time
import numpy as np
from stable_baselines3 import PPO
from train_racing import make_env

def get_latest_model_path():
    """自动获取最新的模型文件夹，优先加载最近成功的模型。"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return None
    subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if d.startswith("racing_rl_")]
    if not subdirs:
        return None
    latest_dir = max(subdirs, key=os.path.getmtime)
    
    # 最快的模型
    fastest_model = os.path.join(latest_dir, "fastest_model.zip")
    if os.path.exists(fastest_model):
        return fastest_model

    # 极高优先级：加载最近一次成功的模型
    last_success = os.path.join(latest_dir, "last_success_model.zip")
    if os.path.exists(last_success):
        return last_success

    # 优先级 2：加载评估出的最佳模型
    best_model = os.path.join(latest_dir, "best_model.zip")
    if os.path.exists(best_model):
        return best_model
    
    

    
    
    # 优先级 3：最终保存的模型
    final_model = os.path.join(latest_dir, "final_racing_model.zip")
    if os.path.exists(final_model):
        return final_model
    return None

def test():
    model_path = get_latest_model_path()
    if not model_path:
        print("[错误] 未找到任何训练好的模型！")
        return

    print(f"\n[启动测试] 正在加载模型: {model_path}")
    
    # 创建带 GUI 的环境
    env = make_env(gui=True)() 
    model = PPO.load(model_path)
    
    # 运行 10 个回合以捕捉 50% 胜率下的成功瞬间
    for episode in range(10):
        obs, info = env.reset()
        print(f"\n--- 第 {episode + 1} 回合开始 ---")
        
        for i in range(2000): 
            # 关闭确定性预测，允许模型带一点随机探索，这通常能发挥出 50% 胜率的真实水平
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            
            time.sleep(0.01)
            
            # --- 核心改进：使用显式的 terminal_reason ---
            reason = info.get('terminal_reason', 'none')
            
            if terminated or truncated:
                if reason == "success":
                    print(f"🎉 [成功] 无人机已抵达终点！(用时: {i} 步)")
                elif reason == "crash_ground":
                    print(f"❌ [失败] 撞地坠毁。")
                elif reason == "crash_ceiling":
                    print(f"❌ [失败] 飞得太高(冲天)。")
                elif reason == "crash_tilt":
                    print(f"❌ [失败] 姿态翻转。")
                elif reason == "timeout":
                    print(f"🕒 [结束] 飞行超时。")
                else:
                    print(f"🏁 [结束] 回合终止 (原因: {reason})")
                
                time.sleep(1)
                break
                
    env.close()

if __name__ == "__main__":
    test()
