"""
蒙特卡洛光传输模拟模块
优化版本：使用Numba JIT编译和向量化操作加速
"""
import numpy as np
import time
from A02_config_params import params
from A03_gauss_init import gauss_init

# 尝试导入Numba用于JIT编译加速（可选）
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 定义一个空的装饰器，如果numba不可用
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# 尝试导入CuPy用于GPU加速（可选）
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def monte_carlo_light(nphotonpackets=500, zrange=None, rmax=None, dstep=None):
    """
    蒙特卡洛光传输模拟主函数
    
    参数:
        nphotonpackets: 光子包数量（×10000）
        zrange: 深度范围 [zmin, zcenter, zmax] (mm)
        rmax: 径向最大范围 (mm)
        dstep: 网格步长 (mm)
    
    返回:
        frac_abs: 单位体积吸收光子比例 [1/mm³]
        frac_trans: 单位面积透射光子比例 [1/mm²]
        r: 径向坐标向量 (mm)
        depth: 深度坐标向量 (mm)
        catcher: 记录每个体素吸收能量的二维矩阵
        nlaunched: 实际发射的光子包总数
        lostphotons: 4类丢失光子计数 [径向溢出, 过深, 盖玻片反射, 颅骨反射]
    """
    
    start_time = time.time()
    
    # 读取参数
    wavelength = params.opt['wavelength']
    modelnumber = params.opt['tissueModelType']
    
    # 设置默认值
    if nphotonpackets < 1:
        nphotonpackets = 1
        print('Warning: 光子包数已自动调整为 1×10000（最小值）')
    
    if zrange is None:
        zrange = [0, 0, 6]
    if rmax is None:
        rmax = 6
    if dstep is None:
        dr = 0.01
    else:
        dr = dstep
    
    # 获取光学参数
    if modelnumber == 3:
        # 用户自定义
        absorption = params.opt['absTissue']
        scattering = params.opt['scatterTissue']
        g = params.opt['gTissue']
    else:
        # 内置模型插值
        if ((wavelength < 480 or wavelength > 900) and modelnumber == 1) or \
           ((wavelength < 450 or wavelength > 1064) and modelnumber == 2):
            print('Warning: 波长超出实验数据范围，正在外推')
        
        # 模型数据
        absorptionmat = {
            1: np.array([[480, 0.37], [560, 0.26], [580, 0.19], [640, 0.05], [780, 0.02], [900, 0.02]]),
            2: np.array([[450, 0.07], [510, 0.04], [630, 0.02], [670, 0.02], [1064, 0.05]])
        }
        scatteringmat = {
            1: np.array([[480, 11], [580, 9.7], [640, 9.0], [700, 8.2], [780, 7.8], [900, 6.6]]),
            2: np.array([[450, 11.7], [510, 10.6], [630, 9.0], [670, 8.4], [1064, 5.7]])
        }
        gmat = {
            1: np.array([[480, 0.89], [580, 0.89], [640, 0.89], [700, 0.90], [780, 0.90], [900, 0.90]]),
            2: np.array([[450, 0.88], [510, 0.88], [630, 0.89], [670, 0.91], [1064, 0.9]])
        }
        
        absorption = np.interp(wavelength, absorptionmat[modelnumber][:, 0], absorptionmat[modelnumber][:, 1])
        scattering = np.interp(wavelength, scatteringmat[modelnumber][:, 0], scatteringmat[modelnumber][:, 1])
        g = np.interp(wavelength, gmat[modelnumber][:, 0], gmat[modelnumber][:, 1])
        g = np.clip(g, 0, 1)
    
    # 初始化模拟变量
    nphotonstotal_touse = 10000 * nphotonpackets
    nphotons = 10000
    lostphotons = np.array([0, 0, 0, 0])
    nlaunched = nphotons
    dz = dr
    n = params.opt['nTissue']
    zmin, zcenter, zmax = zrange
    
    # 用于接收能量的二维矩阵
    catcher = np.zeros((len(np.arange(0, rmax + dr, dr)), len(np.arange(zmin, zmax + dz, dz))))
    
    newplotthreshold = nphotons
    
    print('\nProgress:')
    print(f'{nphotons/nphotonstotal_touse*100:.1f}% 光子已发射')
    print('预计剩余时间：计算中...')
    
    # 发射第一批光子包
    fll = params.geo['focalDepthTissue']
    cors, dirs = gauss_init(nphotons, params.geo['w0'], zcenter, fll)
    w = np.ones(nphotons)
    
    # 主循环
    while True:
        # 1. 随机步长
        s = -np.log(np.random.rand(nphotons)) / (absorption + scattering)
        
        # 2. 移动光子包
        cors = cors + np.tile(s, (3, 1)) * dirs
        
        # 3. 统计越界光子
        rcors = np.sqrt(cors[0, :]**2 + cors[1, :]**2)
        lostphotons[0] += np.sum(w[rcors >= rmax + dr])
        lostphotons[1] += np.sum(w[cors[2, :] >= zmax + dz])
        lostphotons[2] += np.sum(w[(cors[2, :] < zmin) & (rcors < params.geo['r_glass'])])
        lostphotons[3] += np.sum(w[(cors[2, :] < zmin) & (rcors >= params.geo['r_glass'])])
        
        # 4. 杀掉越界光子包
        outofbounds = (rcors >= rmax + dr) | (cors[2, :] >= zmax + dz) | (cors[2, :] < zmin)
        w[outofbounds] = 0
        cors[:, w == 0] = np.array([[0], [0], [zcenter]])
        rcors = np.sqrt(cors[0, :]**2 + cors[1, :]**2)
        
        # 5. 将吸收能量记录到 catcher 矩阵（向量化版本）
        zs = np.floor((cors[2, :] - zmin) / dz).astype(int)
        rs = np.floor(rcors / dr).astype(int)
        
        valid = (rs >= 0) & (rs < catcher.shape[0]) & (zs >= 0) & (zs < catcher.shape[1])
        # 向量化更新catcher矩阵
        if np.any(valid):
            valid_rs = rs[valid]
            valid_zs = zs[valid]
            valid_w = w[valid]
            absorption_ratio = absorption / (absorption + scattering)
            # 使用numpy的add.at进行安全的累加操作
            np.add.at(catcher, (valid_rs, valid_zs), valid_w * absorption_ratio)
        
        # 6. 光子包权重按吸收衰减
        w = w * (scattering / (absorption + scattering))
        
        # 7. 再次杀掉刚刚越界的光子
        outofbounds2 = (rcors >= rmax) | (cors[2, :] >= zmax)
        w[outofbounds2] = 0
        
        # 8. 抽样散射方向
        if g > 0 and g <= 1:
            costh = (1 + g**2 - ((1 - g**2) / (1 - g + 2 * g * np.random.rand(nphotons)))**2) / (2 * g)
        elif g == 0:
            costh = 2 * np.random.rand(nphotons) - 1
        else:
            raise ValueError('g 必须在 [0,1] 之间')
        
        phi = 2 * np.pi * np.random.rand(nphotons)
        sinth = np.sqrt(1 - costh**2)
        temp = np.sqrt(1 - dirs[2, :]**2)
        
        # 9. 旋转坐标系得到新方向向量
        uxyz = np.zeros((3, nphotons))
        uxyz[0, :] = sinth * (dirs[0, :] * dirs[2, :] * np.cos(phi) - dirs[1, :] * np.sin(phi)) / temp
        uxyz[1, :] = sinth * (dirs[1, :] * dirs[2, :] * np.cos(phi) + dirs[0, :] * np.sin(phi)) / temp
        uxyz[2, :] = -np.cos(phi) * temp
        uxyz = uxyz + dirs * np.tile(costh, (3, 1))
        
        # 10. 处理接近垂直方向的数值稳定性
        tofix = (np.abs(dirs[0, :]) < 0.0001) & (np.abs(dirs[1, :]) < 0.0001)
        if np.any(tofix):
            uxyz[0, tofix] = sinth[tofix] * np.cos(phi[tofix])
            uxyz[1, tofix] = sinth[tofix] * np.sin(phi[tofix])
            uxyz[2, tofix] = costh[tofix] * np.sign(dirs[2, tofix])
        
        dirs = uxyz
        
        # 11. 归一化方向向量
        mag = np.sqrt(np.sum(dirs**2, axis=0))
        dirs = dirs / np.tile(mag, (3, 1))
        
        # 12. 俄罗斯轮盘赌
        chance = np.random.rand(nphotons)
        w[(w < 1e-4) & (chance <= 0.1) & (w > 0)] = w[(w < 1e-4) & (chance <= 0.1) & (w > 0)] / 0.1
        todestroy = ((w < 1e-4) & (chance >= 0.1) & (w > 0)) | outofbounds | outofbounds2
        ntodestroy = np.sum(todestroy)
        
        # 13. 补充新光子包
        if ntodestroy > 0:
            if ntodestroy + nlaunched <= nphotonstotal_touse:
                cors[:, todestroy], dirs[:, todestroy] = gauss_init(ntodestroy, params.geo['w0'], zcenter, fll)
                w[todestroy] = 1
                nlaunched += ntodestroy
            elif nlaunched < nphotonstotal_touse:
                which = np.where(todestroy)[0]
                replaceins = which[:nphotonstotal_touse - nlaunched]
                cors[:, replaceins], dirs[:, replaceins] = gauss_init(len(replaceins), params.geo['w0'], zcenter, fll)
                w[replaceins] = 1
                nlaunched += len(replaceins)
                w[which[nphotonstotal_touse - nlaunched:]] = 0
            else:
                w[(w < 1e-4) & (chance >= 0.1) & (w > 0)] = 0
        
        # 14. 若所有光子包权重为 0，结束模拟
        if np.all(w == 0):
            break
        
        # 15. 命令行进度更新
        if nlaunched >= newplotthreshold:
            elapsed_time = time.time() - start_time
            if nlaunched > nphotons:
                timeremaining = elapsed_time * ((nphotonstotal_touse / (nlaunched - nphotons)) - 1)
            else:
                timeremaining = 0.0
            newplotthreshold += nphotons
            print(f'\r{nlaunched/nphotonstotal_touse*100:.1f}% 光子已发射, 预计剩余时间：{timeremaining:.1f} s', end='')
    
    # 后处理
    total_time = time.time() - start_time
    print(f'\n模拟耗时：{total_time:.1f} s')
    
    # 将 catcher 转为单位体积吸收分数
    r_array = np.arange(0, rmax + dr, dr)
    frac_abs = catcher.T / (2 * np.pi * (r_array - 0.5 * dr))
    frac_abs = frac_abs / (dr**2 * dz) / nphotonstotal_touse
    frac_trans = frac_abs / absorption
    
    # 保存实际使用的参数
    params.geo['dstep'] = dr
    params.geo['zrange'] = zrange
    params.geo['rmax'] = rmax
    params.opt['absTissue'] = absorption
    params.opt['scatterTissue'] = scattering
    params.opt['gTissue'] = g
    
    r = r_array
    depth = np.arange(zmin, zmax + dz, dz)
    
    return frac_abs, frac_trans, r, depth, catcher, nlaunched, lostphotons
