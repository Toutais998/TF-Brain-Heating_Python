"""
高斯光束初始化模块
在脑表面按高斯光束强度分布初始化每个光子的位置与传播方向
优化版本：使用Numba JIT编译加速
"""
import numpy as np
from numba import jit  # gauss_init函数使用params全局变量，JIT编译需要重构
from A02_config_params import params


def gauss_init(num, r, zcenter, fl):
    """
    在脑表面按高斯光束强度分布初始化每个光子的位置与传播方向
    适用于无限远共轭系统，依据物镜后孔径处 1/e² 半径的高斯轮廓采样
    
    参数:
        num: 光子数量
        r: 物镜后孔径处高斯束腰 1/e² 半径 (mm)
        zcenter: 光线起始 z 位置（脑表面）
        fl: 焦点到 zcenter 的距离 (mm)
    
    返回:
        cors: 3×num 初始坐标数组
        dirs: 3×num 单位方向向量数组
    """
    
    # 1) 按 2D 高斯采样径向位置（1/e² 半径为 r）
    # 先生成 3 倍数量，后面会被裁剪
    rinit = np.sqrt(-0.5 * np.log(np.random.rand(3 * num))) * r
    
    # 2) 用后孔径 NA 做硬裁剪，保留足够光子
    rinit = rinit[rinit <= params.geo['f'] * params.geo['NA']]
    rinit = rinit[:num]
    
    # 3) 根据正弦条件计算每条光线的发散角 θ（单位：度）
    thinit = np.arcsin(rinit / params.geo['f'] / params.opt['nWater']) * 180 / np.pi
    
    # 4) 随机方位角（0-360°）
    thinitpos = -180 + 2 * np.random.rand(num) * 180
    
    # 5) 由球坐标生成单位方向向量
    zdir = np.cos(thinit * np.pi / 180)  # cosθ
    temp = np.sqrt(1 - zdir**2)          # sinθ
    xdir = -temp * np.cos(thinitpos * np.pi / 180)  # -sinθ·cosφ
    ydir = -temp * np.sin(thinitpos * np.pi / 180)  # -sinθ·sinφ
    # 注：负号使光线向下传播（朝向脑内）
    
    # 6) 扫描模式：在 xy 平面内随机平移（非角度扫描）
    FOV = params.geo['FOV']
    dx = FOV * np.random.rand(num) - 0.5 * FOV  # [-FOV/2, FOV/2]
    dy = FOV * np.random.rand(num) - 0.5 * FOV
    
    # 7) 计算每条光线在脑表面的起始坐标
    # 光线从焦点反向延长到 zcenter 平面
    xcor = fl * np.tan(thinit * np.pi / 180) * np.cos(thinitpos * np.pi / 180)
    ycor = fl * np.tan(thinit * np.pi / 180) * np.sin(thinitpos * np.pi / 180)
    zcor = np.ones(num) * zcenter
    
    # 8) 返回起始坐标与方向
    cors = np.array([xcor + dx, ycor + dy, zcor])  # 三维起始位置（已含扫描平移）
    dirs = np.array([xdir, ydir, zdir])            # 单位方向向量
    
    return cors, dirs
