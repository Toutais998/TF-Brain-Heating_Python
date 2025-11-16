"""
全局参数配置模块
存储所有光学、机械、热学和几何参数
"""
import numpy as np


class SimulationParams:
    """仿真参数类"""
    
    def __init__(self):
        # 光学参数和介质属性
        self.opt = {
            'wavelength': 510,           # 波长 nm (Yaroslavsky 2002)
            'plotWavelength': None,      # 绘图波长 nm
            'tissueModelType': 3,        # 组织模型类型
            'absTissue': 0.04,          # 吸收系数 1/mm (532nm需要更高的吸收系数)
            'scatterTissue': 10.6,       # 散射系数 1/mm (532nm散射系数略高)
            # 这三个参数是Tianyu用的920参数
            # 'wavelength': 920,
            # 'absTissue': 0.039,
            # 'scatterTissue': 6.66,
            # 'gTissue': 0.9,             # 各向异性因子
            'gTissue': 0.88,
            'nTissue': 1.36,
            'absGlass': None,           # 玻璃吸收系数
            'nGlass': None,             # 玻璃折射率
            'absWater': None,           # 水吸收系数
            'nWater': 1.328,            # 水折射率
        }
        
        # 散射系数函数（人类数据，Bevilacqua et al 2000）
        self.opt['scatterFrontCort'] = lambda x: 10.9 * ((x/500)**(-0.334))
        self.opt['scatterTempCort'] = lambda x: 11.6 * ((x/500)**(-0.601))
        
        # 介质机械属性
        self.mech = {
            'pTissue': 1.04e-6,         # 脑组织密度 kg/mm³
            'pBlood': 1.06e-6,          # 血液密度 kg/mm³
            'wBlood': 8.5e-3,           # 血液灌注率 /s
            'pGlass': 2.23e-6,          # 玻璃密度 kg/mm³
            'pWater': 1e-6,             # 水密度 kg/mm³
            'pBoneCancellous': 1.178e-6,  # 松质骨密度 kg/mm³
            'pBoneCortical': 1.908e-6,    # 皮质骨密度 kg/mm³
        }
        
        # 介质热学属性
        self.therm = {
            'cTissue': 3.65e6,          # 脑组织比热 mJ/kg/°C
            'kTissue': 0.527,           # 脑组织导热率 mW/mm/°C
            'cBlood': 3.6e6,            # 血液比热 mJ/kg/°C
            'qBrain': 9.7e-3,           # 脑代谢产热 mW/mm³
            'cGlass': 0.647e6,          # 玻璃比热 mJ/kg/°C
            'kGlass': 0.8,              # 玻璃导热率 mW/mm/°C
            'cWater': 4.184e6,          # 水比热 mJ/kg/°C
            'kWater': 0.6,              # 水导热率 mW/mm/°C
            'cBoneCancellous': 2.274e6,   # 松质骨比热 mJ/kg/°C
            'kBoneCancellous': 0.31,      # 松质骨导热率 mW/mm/°C
            'cBoneCortical': 1.313e6,     # 皮质骨比热 mJ/kg/°C
            'kBoneCortical': 0.32,        # 皮质骨导热率 mW/mm/°C
            'uStep': 0.03,              # 热传导仿真最小长度单位 mm
        }
        
        # 样品几何参数
        self.geo = {
            'dstep': None,              # 离散化步长
            'zrange': None,             # 深度范围
            'rmax': None,               # 最大径向范围
            'd_glass': 0.16,            # 盖玻片厚度 mm
            'r_glass': 2,               # 盖玻片半径 mm
            'd_water': None,            # 浸水层厚度 mm
            'd_skull': 0.14,            # 颅骨厚度 mm
            'd_boneCortical': 0.01,     # 颅骨皮质层厚度 mm
            'NA': 1.05,                 # 物镜数值孔径
            'f': 7.2,                   # 物镜焦距 mm
            'wd': 2,                    # 物镜工作距离 mm
            'w0': 5.3,                  # 物镜后孔径处光束1/e²半径 mm
            'FOV': 0.3,                 # 焦点扫描线性视场 mm
            'focalDepthTissue': 0,      # 组织内焦点深度 mm
            'surfIllumRadius': None,    # 光锥与脑组织交线圆的半径
        }
        
        # 初始温度场（可选）
        self.u_start = None


# 全局参数实例
params = SimulationParams()
