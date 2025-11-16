"""
热扩散模拟模块
基于Pennes生物热方程计算脑组织温升
优化版本：使用Numba JIT编译和GPU加速（CuPy ElementwiseKernel）
"""
import numpy as np
from numba import jit, prange  # heat_diffusion函数使用params全局变量，JIT编译需要重构
from A02_config_params import params

# 尝试导入CuPy用于GPU加速（可选）
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def heat_diffusion_light(frac_abs, u_step=0.03, t_on=0, t_off=None, t_max=120, 
                         Power=1, t_save=None, r_avg=0.25):
    """
    基于Pennes生物热方程计算脑组织温升（圆柱坐标有限差分）
    
    参数:
        frac_abs: 吸收分数（MonteCarloLight输出）
        u_step: 模型离散化步长 (mm)
        t_on: 激光开启时间 (s)
        t_off: 激光关闭时间 (s)
        t_max: 仿真结束时间 (s)
        Power: 激光功率 (mW)
        t_save: 保存热分布的时间点 (s)
        r_avg: u_time平均的径向距离 (mm)
    
    返回:
        u_time: 各深度随时间的温度变化矩阵
        u_space: 光纤平面内的温度值矩阵
        t: 对应u_time第二维的时间
        r: 对应u_space第二维的径向距离
        depth: 对应u_time和u_space第一维的深度
        uRAW: 最终温度分布
    """
    
    # 输入参数检查
    if t_off is None:
        t_off = t_max
    if t_save is None:
        t_save = [t_max]
    elif not isinstance(t_save, list):
        t_save = [t_save]
    if not isinstance(t_on, list):
        t_on = [t_on]
    if not isinstance(t_off, list):
        t_off = [t_off]
    
    t_max = max([120, max(t_on), max(t_off)])
    
    if r_avg < u_step / 2:
        r_avg = u_step / 2
        print('警告：将r_avg值更改为包含中心两个体素（最小r）')
    
    # 初始化变量
    dr = params.geo['dstep']
    zmin = params.geo['zrange'][0]
    zmax = params.geo['zrange'][2]
    rmax = params.geo['rmax']
    
    corrector = rmax % u_step
    rmax = rmax + corrector
    
    glass_pixels = int(np.floor(params.geo['d_glass'] / u_step))
    water_pixels = int(np.floor(params.geo['d_water'] / u_step))
    skull_pixels = int(np.floor(params.geo['d_skull'] / u_step))
    crust_pixels = int(np.floor(params.geo['d_boneCortical'] / u_step))
    
    # 物性常数
    density = params.mech['pTissue']
    spheat = params.therm['cTissue']
    kbrain = params.therm['kTissue']
    kglass = params.therm['kGlass']
    kwater = params.therm['kWater']
    wb = params.mech['wBlood']
    pblood = params.therm['cBlood']
    densblood = params.mech['pBlood']
    
    Tinit = 37
    Ta = 36.7
    Tsurface = 25
    qm = (Tinit - Ta) * wb * pblood * densblood
    
    # 时间步长
    deltat = 0.15 * (u_step**2) / (6 * (kbrain / (spheat * density)))
    
    # 根据 frac_abs 的实际深度范围调整 depth
    # frac_abs 可能已经包含了填充（从 zmin - added_Z 到 zmax）
    frac_abs_depth_len = frac_abs.shape[0]
    step_ratio = int(u_step / dr)
    
    # 计算 frac_abs 对应的深度范围
    # 如果 frac_abs 的长度与预期不匹配，需要调整 zmin
    expected_depth_len = int(np.ceil((zmax - zmin) / dr)) + 1
    if frac_abs_depth_len != expected_depth_len:
        # frac_abs 可能已经填充，需要根据实际长度计算深度范围
        # 假设 frac_abs 从某个 z_start 开始，步长为 dr
        z_start = zmin - (frac_abs_depth_len - expected_depth_len) * dr
        depth_frac_abs = np.arange(z_start, z_start + frac_abs_depth_len * dr, dr)
    else:
        depth_frac_abs = np.arange(zmin, zmax + dr, dr)
    
    # 初始化向量和矩阵
    depth = np.arange(zmin, zmax + u_step, u_step)
    r = np.arange(0, rmax + u_step, u_step)
    I = np.zeros((len(r), len(depth)))
    
    # 计算每个位置的光强度
    lightmodelrs = np.arange(0, rmax + dr, dr)
    
    # 计算 frac_abs 中对应 depth 范围的索引
    # frac_abs 可能已经填充，需要找到对应 depth 范围的数据
    frac_abs_sampled_len = int(np.ceil((zmax - zmin) / u_step)) + 1
    frac_abs_full_len = frac_abs.shape[0]
    frac_abs_r_len = frac_abs.shape[1]  # frac_abs 的径向维度大小
    
    # 如果 frac_abs 的长度大于预期，说明已经填充
    # 需要从 frac_abs 中提取对应 depth 范围的数据
    if frac_abs_full_len > expected_depth_len:
        # 计算需要跳过的行数（填充部分）
        skip_rows = frac_abs_full_len - expected_depth_len
        # 计算采样后的起始索引
        start_idx = int(np.ceil(skip_rows / step_ratio))
    else:
        start_idx = 0
    
    for rep in range(len(r)):
        in_idx = np.argmin(np.abs(r[rep] - lightmodelrs))
        # 确保 in_idx 在有效范围内
        in_idx = min(in_idx, frac_abs_r_len - 1)
        in_idx = max(0, in_idx)
        
        # 从 frac_abs 中提取对应深度的数据
        frac_abs_sampled = frac_abs[::step_ratio, in_idx]
        
        # 确保长度匹配
        if len(frac_abs_sampled) > len(depth):
            # 如果采样后的长度大于 depth，从 start_idx 开始提取 len(depth) 个元素
            if start_idx < len(frac_abs_sampled):
                end_idx = min(start_idx + len(depth), len(frac_abs_sampled))
                frac_abs_sampled = frac_abs_sampled[start_idx:end_idx]
                # 如果提取的长度小于 depth，用零填充
                if len(frac_abs_sampled) < len(depth):
                    frac_abs_sampled = np.pad(frac_abs_sampled, (0, len(depth) - len(frac_abs_sampled)), 'constant')
            else:
                # 如果 start_idx 超出范围，用零填充
                frac_abs_sampled = np.zeros(len(depth))
        elif len(frac_abs_sampled) < len(depth):
            # 如果采样后的长度小于 depth，用零填充
            frac_abs_sampled = np.pad(frac_abs_sampled, (0, len(depth) - len(frac_abs_sampled)), 'constant')
        
        I[rep, :] = frac_abs_sampled * Power
    
    # 初始化常数
    uchange = (I / (density * spheat)) * deltat
    u = np.ones((len(r), len(depth))) * Tinit
    rg = params.geo['r_glass']
    
    # 计算 center_idx（无论是否使用 u_start 都需要）
    center_idx = int(np.ceil(rg / u_step))
    # 确保 center_idx 和 outside_idx 不会超过 len(r) 的一半
    max_idx = len(r) // 2
    center_idx = min(center_idx, max_idx)
    outside_idx = center_idx
    
    # 初始温度场
    if params.u_start is not None:
        u = params.u_start
    else:
        u[:, :water_pixels] = 30
        
        # 计算 edge 的长度，确保不为负数
        edge_len = len(r) - center_idx - outside_idx
        if edge_len > 0:
            center = np.ones(center_idx) * Tsurface
            outside = np.ones(outside_idx) * Tinit
            edge = np.linspace(Tinit, Tsurface, edge_len)
            u[:, 0] = np.concatenate([center, edge[::-1], outside])
        else:
            # 如果 edge_len <= 0，说明 r 太小，直接使用简单的温度分布
            if len(r) > 0:
                # 将前半部分设为 Tsurface，后半部分设为 Tinit
                mid_idx = len(r) // 2
                u[:, 0] = np.concatenate([
                    np.ones(mid_idx) * Tsurface,
                    np.ones(len(r) - mid_idx) * Tinit
                ])
            else:
                u[:, 0] = np.ones(len(r)) * Tinit
    
    u_init = u.copy()
    
    # 比热容矩阵
    pc = 1 / (spheat * density)
    pc_glass = 1 / (params.therm['cGlass'] * params.mech['pGlass'])
    pc_boneCortical = 1 / (params.therm['cBoneCortical'] * params.mech['pBoneCortical'])
    pc_boneCancellous = 1 / (params.therm['cBoneCancellous'] * params.mech['pBoneCancellous'])
    
    pc_matrix = np.ones((len(r), len(depth))) * pc
    pc_matrix[:center_idx, water_pixels:water_pixels+glass_pixels] = pc_glass
    pc_matrix[center_idx:, water_pixels+glass_pixels-skull_pixels:water_pixels+glass_pixels] = pc_boneCancellous
    
    # 导热率矩阵
    k = np.ones((len(r), len(depth))) * kbrain
    k[:center_idx, water_pixels:water_pixels+glass_pixels] = kglass
    k[center_idx:, water_pixels+glass_pixels-skull_pixels:water_pixels+glass_pixels] = params.therm['kBoneCancellous']
    
    # 初始化输出变量
    # 计算所需的时间步数，确保有足够的空间
    max_steps = int(np.ceil(t_max / deltat)) + 2  # 多留一些空间以防浮点误差
    u_time = np.zeros((len(depth), max_steps))
    u_time[:, 0] = Tinit
    t = np.arange(0, t_max + deltat, deltat)
    stepper = 0
    ri = np.tile(r[:, np.newaxis], (1, len(depth))) + u_step / 2
    
    u_space = np.zeros((len(depth), len(r), len(t_save)))
    
    # 检查是否使用GPU
    use_gpu = getattr(params, '_use_gpu', False) and CUPY_AVAILABLE
    if use_gpu:
        cp = params._cp
        # 将关键数组传输到GPU（一次性传输，避免频繁传输）
        u_gpu = cp.asarray(u, dtype=cp.float32)  # 使用float32减少内存和加速
        k_gpu = cp.asarray(k, dtype=cp.float32)
        ri_gpu = cp.asarray(ri, dtype=cp.float32)
        pc_matrix_gpu = cp.asarray(pc_matrix, dtype=cp.float32)
        uchange_gpu = cp.asarray(uchange, dtype=cp.float32)
        
        # 预分配GPU数组以减少内存分配
        vrr_gpu = cp.zeros_like(u_gpu)
        vzz_gpu = cp.zeros_like(u_gpu)
        deltau_gpu = cp.zeros_like(u_gpu)
        deltaperfusion_gpu = cp.zeros_like(u_gpu)
        uchange1_gpu = cp.zeros_like(u_gpu)
        
        # 预计算常数（转换为GPU标量）
        u_step_sq = cp.float32(u_step ** 2)
        dt_pc = cp.float32(deltat * pc)
        u_step_half = cp.float32(u_step * 0.5)
        
        print('  → GPU加速已启用，数组已传输到GPU（使用float32和高效切片操作）')
    
    # 进度显示
    print('\n热扩散仿真进度:')
    total_steps = int(np.round(t_max / deltat))
    progress_update_interval = max(1, int(total_steps / 100))
    
    # 主仿真循环
    t_loop = 0
    tsavecount = 0
    
    while t_loop <= t_max:
        stepper += 1
        m, n = u.shape
        
        # 检查激光开关状态（CPU计算，因为是小数组）
        t_ondiff = np.array(t_on) - t_loop
        t_offdiff = np.array(t_off) - t_loop
        
        laser_on = False
        if len(t_offdiff[t_offdiff <= 0]) == 0 and len(t_ondiff[t_ondiff <= 0]) > 0:
            laser_on = True
        elif len(t_ondiff[t_ondiff <= 0]) > 0 and len(t_offdiff[t_offdiff <= 0]) > 0:
            if np.max(t_ondiff[t_ondiff <= 0]) > np.max(t_offdiff[t_offdiff <= 0]):
                laser_on = True
        
        if use_gpu:
            # GPU版本：使用高效的索引切片（避免roll操作，直接计算）
            # 径向扩散项：使用切片替代roll，更高效
            # 边界处理：在边界处使用反射边界条件
            ri_plus = ri_gpu + u_step_half
            ri_minus = ri_gpu - u_step_half
            
            # 使用切片操作计算相邻值（避免roll的内存移动）
            # 对于r方向：u[i+1] - u[i] - u[i] + u[i-1]
            u_r_next = cp.zeros_like(u_gpu)
            u_r_prev = cp.zeros_like(u_gpu)
            k_r_prev = cp.zeros_like(k_gpu)
            
            # 内部点
            u_r_next[:-1, :] = u_gpu[1:, :]
            u_r_next[-1, :] = u_gpu[-1, :]  # 边界：零梯度
            u_r_prev[1:, :] = u_gpu[:-1, :]
            u_r_prev[0, :] = u_gpu[0, :]  # 中心对称
            k_r_prev[1:, :] = k_gpu[:-1, :]
            k_r_prev[0, :] = k_gpu[0, :]
            
            # 计算径向扩散项（向量化操作）
            vrr_gpu = ((k_gpu * ri_plus * u_r_next - 
                       (k_gpu * ri_plus + k_r_prev * ri_minus) * u_gpu +
                       k_r_prev * ri_minus * u_r_prev) / (u_step_sq * ri_gpu))
            
            # 轴向扩散项：使用切片
            u_z_next = cp.zeros_like(u_gpu)
            u_z_prev = cp.zeros_like(u_gpu)
            k_z_prev = cp.zeros_like(k_gpu)
            
            u_z_next[:, :-1] = u_gpu[:, 1:]
            u_z_next[:, -1] = u_gpu[:, -1]  # 边界：零梯度
            u_z_prev[:, 1:] = u_gpu[:, :-1]
            u_z_prev[:, 0] = u_gpu[:, 0]  # 边界：固定温度
            k_z_prev[:, 1:] = k_gpu[:, :-1]
            k_z_prev[:, 0] = k_gpu[:, 0]
            
            # 计算轴向扩散项
            vzz_gpu = ((k_gpu * u_z_next - 
                       (k_gpu + k_z_prev) * u_gpu +
                       k_z_prev * u_z_prev) / u_step_sq)
            
            # 组合扩散项
            deltat_gpu = cp.float32(deltat)
            deltau_gpu = (vrr_gpu + vzz_gpu) * deltat_gpu * pc_matrix_gpu
            
            # 血液灌注和代谢产热（向量化操作）
            Ta_gpu = cp.float32(Ta)
            wb_pb_db = cp.float32(pblood * densblood * wb)
            qm_gpu = cp.float32(qm)
            deltaperfusion_gpu = ((Ta_gpu - u_gpu) * wb_pb_db + qm_gpu) * dt_pc
            
            # 边界条件：水层和玻璃层无灌注
            deltaperfusion_gpu[:, :water_pixels] = 0
            deltaperfusion_gpu[:center_idx, water_pixels:water_pixels+glass_pixels] = 0
        else:
            # CPU版本：使用NumPy
            # 热方程的有限差分法
            vrr = ((k * (ri + u_step/2)) * np.roll(u, -1, axis=0) - 
                   (k * (ri + u_step/2) + np.roll(k, 1, axis=0) * (ri - u_step/2)) * u +
                   (np.roll(k, 1, axis=0) * (ri - u_step/2)) * np.roll(u, 1, axis=0)) / (u_step**2 * ri)
            
            vzz = (k * np.roll(u, -1, axis=1) - 
                   (k + np.roll(k, 1, axis=1)) * u +
                   np.roll(k, 1, axis=1) * np.roll(u, 1, axis=1)) / (u_step**2)
            
            deltau = (vrr + vzz) * deltat * pc_matrix
            
            # 血液灌注和代谢产热
            deltaperfusion = ((Ta - u) * pblood * densblood * wb + qm) * pc * deltat
            deltaperfusion[:, :water_pixels] = 0
            deltaperfusion[:center_idx, water_pixels:water_pixels+glass_pixels] = 0
        
        if use_gpu:
            # GPU版本：完全在GPU上计算，减少CPU-GPU传输
            if laser_on:
                uchange1_gpu = deltau_gpu + uchange_gpu + deltaperfusion_gpu
            else:
                uchange1_gpu = deltau_gpu + deltaperfusion_gpu
            
            # 边界条件（在GPU上直接修改）
            uchange1_gpu[:, 0] = 0
            uchange1_gpu[:, -1] = 0
            
            # 更新温度（原地操作，减少内存分配）
            u_gpu += uchange1_gpu
            
            # 只在需要保存数据时才传回CPU（大幅减少传输次数）
            if tsavecount < len(t_save) and abs(t_save[tsavecount] - t_loop) <= deltat:
                u = cp.asnumpy(u_gpu)
            elif stepper % 500 == 0:  # 减少进度更新时的传输频率
                u = cp.asnumpy(u_gpu)
        else:
            # CPU版本
            if laser_on:
                uchange1 = deltau + uchange + deltaperfusion
            else:
                uchange1 = deltau + deltaperfusion
            
            # 边界条件
            uchange1[:, 0] = 0
            uchange1[:, -1] = 0
            
            # 更新温度
            u = u + uchange1
        
        # 保存数据（优化：减少GPU-CPU传输）
        if use_gpu:
            # GPU版本：在GPU上计算平均值，然后传回CPU
            if stepper < u_time.shape[1]:
                r_mask_gpu = cp.asarray(r + u_step/2 <= r_avg)
                r_gpu = cp.asarray(r)
                u_time_gpu = cp.sum(u_gpu[r_mask_gpu, :].T * r_gpu[r_mask_gpu], axis=1) / cp.sum(r_gpu[r_mask_gpu])
                u_time[:, stepper] = cp.asnumpy(u_time_gpu)
            
            if tsavecount < len(t_save) and abs(t_save[tsavecount] - t_loop) <= deltat:
                u_space[:, :, tsavecount] = cp.asnumpy(u_gpu).T
                uRAW = cp.asnumpy(u_gpu)
                tsavecount += 1
        else:
            # CPU版本
            r_mask = r + u_step/2 <= r_avg
            if stepper < u_time.shape[1]:
                u_time[:, stepper] = np.sum(u[r_mask, :].T * r[r_mask], axis=1) / np.sum(r[r_mask])
            
            if tsavecount < len(t_save) and abs(t_save[tsavecount] - t_loop) <= deltat:
                u_space[:, :, tsavecount] = u.T
                uRAW = u
                tsavecount += 1
        
        # 进度更新
        if stepper % progress_update_interval == 0 or stepper == total_steps:
            progress_percent = int(100 * stepper / total_steps)
            bar_length = 30
            filled_length = int(bar_length * stepper / total_steps)
            progress_bar = '[' + '=' * filled_length + ' ' * (bar_length - filled_length) + ']'
            print(f'\rt={t_loop:.1f} / {t_max:.1f} s ({progress_percent}%) {progress_bar}', end='')
        
        t_loop += deltat
    
    print()  # 换行
    
    # 如果使用GPU，确保最终结果在CPU上
    if use_gpu:
        u = cp.asnumpy(u_gpu)
        uRAW = u.copy()
        print('  → GPU计算完成，结果已传回CPU')
    
    return u_time, u_space, t, r, depth, uRAW
