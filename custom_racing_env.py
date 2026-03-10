import time
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

class RacingGateAviary(BaseRLAviary):
    """
    带三屏独立绿色 HUD 显示器的单机竞速环境。
    """
    def __init__(self, **kwargs):
        # 1. 定义赛道核心位置 (世界坐标)
        self.START_POS = np.array([-1.5, 0, 0.2])       # 起点 z 从 0.1 提升到 0.2
        self.GATE_POS  = np.array([2.0, 0.0, 0.5])   # 穿越门中心
        self.FINISH_POS = np.array([6.5, 0.0, 0.1])  # 终点
        
        # 2. 定义活动空间限制 [x_min, x_max, y_min, y_max, z_min, z_max]
        self.BOUNDS = np.array([-2.0, 8.0, -4.0, 4.0, 0.0, 4.0])

        # 初始化 HUD IDs (三个显示器)
        self.hud_ids = [-1, -1, -1]

        # 如果没有提供 initial_xyzs，默认从 START_POS 出发
        if 'initial_xyzs' not in kwargs:
            kwargs['initial_xyzs'] = self.START_POS.reshape(1, 3)

        # 确保速度限制足够大
        super().__init__(
            pyb_freq=480, # 提升物理频率
            ctrl_freq=240, # 提升控制频率
            **kwargs
        )
        self.SPEED_LIMIT = 10.0 # 目标 10 m/s

    def step(self, action):
        """覆盖 step 方法以实时更新 HUD。"""
        obs, reward, terminated, truncated, info = super().step(action)
        if self.GUI:
            self._update_hud()
        return obs, reward, terminated, truncated, info

    def _update_hud(self):
        """在 3D 场景中显示三个独立的绿色实时显示器。"""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = np.linalg.norm(state[10:13])
        dist = np.linalg.norm(self.FINISH_POS - pos)
        
        # 三项核心指标
        metrics = [
            f"HEIGHT: {pos[2]:.2f} m",
            f"SPEED : {vel:.2f} m/s",
            f"DIST  : {dist:.2f} m"
        ]
        
        # 显示器堆叠参数
        start_xyz = [self.START_POS[0], self.START_POS[1] - 2, 3.5]
        vertical_spacing = 0.4
        
        for i, text in enumerate(metrics):
            text_pos = [start_xyz[0], start_xyz[1], start_xyz[2] - i * vertical_spacing]
            
            if self.hud_ids[i] == -1:
                self.hud_ids[i] = p.addUserDebugText(text, 
                                                text_pos, 
                                                textColorRGB=[0, 1, 0], # 亮绿色
                                                textSize=2.5, 
                                                physicsClientId=self.CLIENT)
            else:
                self.hud_ids[i] = p.addUserDebugText(text, 
                                                text_pos, 
                                                textColorRGB=[0, 1, 0], # 亮绿色
                                                textSize=2.5, 
                                                replaceItemUniqueId=self.hud_ids[i], 
                                                physicsClientId=self.CLIENT)

    def _addObstacles(self):
        """添加赛道视觉元素。"""
        # 起点 (绿)
        vs_start = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.01, rgbaColor=[0, 1, 0, 0.6], physicsClientId=self.CLIENT)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_start, 
                         basePosition=[self.START_POS[0], self.START_POS[1], 0.02], physicsClientId=self.CLIENT)

        # 穿越门 (红)
        self._build_gate(self.GATE_POS, width=0.8, height=0.8)

        # 终点 (蓝)
        vs_finish = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.01, rgbaColor=[0, 0, 1, 0.6], physicsClientId=self.CLIENT)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_finish, 
                         basePosition=[self.FINISH_POS[0], self.FINISH_POS[1], 0.02], physicsClientId=self.CLIENT)

        if self.GUI:
            self._draw_bounds()

    def _build_gate(self, pos, width, height):
        x, y, z = pos
        thick = 0.05
        red = [1, 0, 0, 1]
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[thick, thick, height/2]),
                         p.createVisualShape(p.GEOM_BOX, halfExtents=[thick, thick, height/2], rgbaColor=red),
                         [x, y - width/2, z], physicsClientId=self.CLIENT)
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[thick, thick, height/2]),
                         p.createVisualShape(p.GEOM_BOX, halfExtents=[thick, thick, height/2], rgbaColor=red),
                         [x, y + width/2, z], physicsClientId=self.CLIENT)
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[thick, width/2 + thick, thick]),
                         p.createVisualShape(p.GEOM_BOX, halfExtents=[thick, width/2 + thick, thick], rgbaColor=red),
                         [x, y, z + height/2], physicsClientId=self.CLIENT)

    def _draw_bounds(self):
        b = self.BOUNDS
        color = [1, 0, 0] 
        for start, end in [([b[0],b[2],b[4]], [b[1],b[2],b[4]]), ([b[1],b[2],b[4]], [b[1],b[3],b[4]]), 
                           ([b[1],b[3],b[4]], [b[0],b[3],b[4]]), ([b[0],b[3],b[4]], [b[0],b[2],b[4]]),
                           ([b[0],b[2],b[5]], [b[1],b[2],b[5]]), ([b[1],b[2],b[5]], [b[1],b[3],b[5]]),
                           ([b[1],b[3],b[5]], [b[0],b[3],b[5]]), ([b[0],b[3],b[5]], [b[0],b[2],b[5]]),
                           ([b[0],b[2],b[4]], [b[0],b[2],b[5]]), ([b[1],b[2],b[4]], [b[1],b[2],b[5]]),
                           ([b[1],b[3],b[4]], [b[1],b[3],b[5]]), ([b[0],b[3],b[4]], [b[0],b[3],b[5]])]:
            p.addUserDebugLine(start, end, color, physicsClientId=self.CLIENT)

    def _computeReward(self):
        return 0

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        return {}

if __name__ == "__main__":
    env = RacingGateAviary(gui=True)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.CLIENT)
    p.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=-35, cameraPitch=-35, cameraTargetPosition=[2, 0, 1], physicsClientId=env.CLIENT)

    env.reset()
    # for i in range(1500): 
    while True:
        obs, reward, terminated, truncated, info = env.step(np.zeros((1, 4))) 
        if terminated or truncated:
            env.reset()
        time.sleep(0.01)
    env.close()
