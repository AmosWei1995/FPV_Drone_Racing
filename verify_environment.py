import torch
import pybullet as p

def verify_ubuntu_env():
    print("--- 硬件与 CUDA 状态检测 ---")
    if not torch.cuda.is_available():
        print("[错误] PyTorch 无法检测到 CUDA。请检查 Ubuntu 驱动或 PyTorch 版本。")
        return

    device_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_id)
    print(f"[通过] CUDA 正常运行。当前 GPU: {gpu_name}")
    
    try:
        print("\n--- 显存分配测试 ---")
        # 申请 1GB 显存测试 (GTX 1650S 极限)
        dummy_tensor = torch.zeros((256, 1024, 1024), dtype=torch.float32, device="cuda")
        allocated = torch.cuda.memory_allocated(device_id) / (1024**2)
        print(f"[通过] 显存分配成功。当前已占用: {allocated:.2f} MB")
        del dummy_tensor
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"[错误] 显存分配失败 (OOM 风险): {e}")

    print("\n--- 物理引擎测试 ---")
    try:
        # 在 Ubuntu 服务器/无桌面环境下，DIRECT 模式是唯一选择
        physicsClient = p.connect(p.DIRECT) 
        print("[通过] PyBullet 无头模式 (DIRECT) 连接成功。")
        p.disconnect()
    except Exception as e:
        print(f"[错误] PyBullet 连接失败: {e}")

if __name__ == "__main__":
    verify_ubuntu_env()