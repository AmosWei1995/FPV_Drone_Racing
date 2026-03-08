import time
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

class RacingGateAviary(BaseRLAviary):
    """
    自定义的无人机竞速环境。
    继承自 BaseRLAviary，方便后续直接接入 Stable-Baselines3。
    """
    def __init__(self, **kwargs):
        super().__init__(
            obs=ObservationType.KIN, # 使用运动学向量作为状态输入 (非图像)
            act=ActionType.RPM,      # 动作空间输出为 4 个电机的 RPM
            **kwargs
        )

    def _addObstacles(self):
        """
        覆盖父类方法：在环境中动态生成穿越门。
        每次环境 reset() 时都会调用此方法，非常适合做赛道的随机化。
        """
        # super()._addObstacles() # 不加载默认障碍物，地面已在 BaseAviary._housekeeping 中加载

        # --- 门 1 的参数设置 ---
        gate_x = 1.0     
        gate_y = 0.0     
        gate_z = 0.5     
        
        width = 0.8      # 门框内径宽度 0.8 米
        height = 0.8     # 门框内径高度 0.8 米
        thickness = 0.05 # 门框边柱的厚度 5 厘米

        # 颜色设置为亮红色 (RGBA)
        red_color = [1, 0, 0, 1]

        # 1. 创建左侧立柱
        p.createMultiBody(
            baseMass=0, # 质量为 0 代表它是静态障碍物
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[thickness, thickness, height/2], physicsClientId=self.CLIENT),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[thickness, thickness, height/2], rgbaColor=red_color, physicsClientId=self.CLIENT),
            basePosition=[gate_x, gate_y - width/2, gate_z],
            physicsClientId=self.CLIENT
        )
        
        # 2. 创建右侧立柱
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[thickness, thickness, height/2], physicsClientId=self.CLIENT),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[thickness, thickness, height/2], rgbaColor=red_color, physicsClientId=self.CLIENT),
            basePosition=[gate_x, gate_y + width/2, gate_z],
            physicsClientId=self.CLIENT
        )
        
        # 3. 创建顶部横梁
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[thickness, width/2 + thickness, thickness], physicsClientId=self.CLIENT),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[thickness, width/2 + thickness, thickness], rgbaColor=red_color, physicsClientId=self.CLIENT),
            basePosition=[gate_x, gate_y, gate_z + height/2],
            physicsClientId=self.CLIENT
        )
        
        # 记录门中心的绝对坐标，稍后会被用于奖励函数的计算
        self.target_gate_pos = np.array([gate_x, gate_y, gate_z])

    def _computeReward(self):
        """计算奖励函数。"""
        # 简单示例：根据距离门的距离计算奖励
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(self.target_gate_pos - state[0:3])
        return -dist

    def _computeTerminated(self):
        """判断是否达到终止条件。"""
        return False

    def _computeTruncated(self):
        """判断是否达到截断条件。"""
        # 例如：飞行时间超过 10 秒或飞出边界
        if self.step_counter / self.PYB_FREQ > 10:
            return True
        return False

    def _computeInfo(self):
        """返回额外信息。"""
        return {}

if __name__ == "__main__":
    # 测试运行代码 (仅用于视觉验证，开启 GUI)
    env = RacingGateAviary(gui=True)
    
    # 关闭多余的侧边栏 UI，只看 3D 画面
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.CLIENT)
    
    # 调整相机视角，从侧后方俯视无人机和门
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-30, cameraPitch=-30, cameraTargetPosition=[0.5, 0, 0.5], physicsClientId=env.CLIENT)

    env.reset()
    
    print("\n[环境加载完毕] 你应该能看到正前方有一个红色的矩形门。")
    print("当前为怠速状态（Action=0），无人机会尝试悬停或缓慢下落。")

    # 运行 5 秒钟的物理仿真循环
    # for i in range(500): 
    while True:
        # 输入一个全为 0 的动作 (对应 HOVER_RPM)
        env.step(np.zeros((1, 4))) 
        time.sleep(0.01)
        
    env.close()