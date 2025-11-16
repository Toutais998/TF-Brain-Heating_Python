"""
主程序：510nm双光子成像深度-功率扫描
计算小鼠大脑皮层在颅窗和浸水下510nm双光子成像时
不同扫描深度和功率下的温度分布

优化版本：支持并行计算和GPU加速
"""
import numpy as np
import os
import pickle
import time
from functools import partial
from A02_config_params import params
from A05_monte_carlo_light import monte_carlo_light
from A04_heat_diffusion import heat_diffusion_light
from A06_visualization import light_heat_plotter, plot_temperature_map, plot_delta_temperature_map
import matplotlib.pyplot as plt
from A07_performance_profiler import profiler, profile_function

# 尝试导入GPU加速版本
try:
    from A05_monte_carlo_light_gpu import monte_carlo_light_gpu
    GPU_MC_AVAILABLE = True
except ImportError:
    GPU_MC_AVAILABLE = False
    monte_carlo_light_gpu = monte_carlo_light

# 尝试导入优化的热扩散版本
try:
    from A04_heat_diffusion_optimized import heat_diffusion_light as heat_diffusion_light_opt
    GPU_HEAT_AVAILABLE = True
except ImportError:
    GPU_HEAT_AVAILABLE = False
    heat_diffusion_light_opt = heat_diffusion_light

# 尝试导入并行计算支持
try:
    from multiprocessing import Pool, cpu_count, Manager
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    cpu_count = lambda: 1


def process_single_depth(args):
    """
    处理单个深度的函数（用于并行处理）
    
    参数:
        args: 元组 (i, depth, depthArray, powerPercent, utot, output_dir, 
                     SAVE_PLOTS, SAVE_DATA, USE_GPU)
    
    返回:
        result: 字典，包含该深度的所有结果
    """
    import numpy as np
    import pickle
    import time
    import os
    import matplotlib.pyplot as plt
    from A02_config_params import SimulationParams
    from A05_monte_carlo_light import monte_carlo_light
    try:
        from A05_monte_carlo_light_gpu import monte_carlo_light_gpu
        gpu_mc_available = True
    except:
        gpu_mc_available = False
        monte_carlo_light_gpu = monte_carlo_light
    from A04_heat_diffusion import heat_diffusion_light
    from A06_visualization import light_heat_plotter
    
    i, depth, depthArray, powerPercent, utot, output_dir, SAVE_PLOTS, SAVE_DATA, USE_GPU = args
    
    # 创建独立的参数实例（避免多进程间共享状态）
    local_params = SimulationParams()
    # 复制全局参数
    for key in ['opt', 'mech', 'therm', 'geo']:
        if hasattr(params, key):
            setattr(local_params, key, getattr(params, key).copy() if isinstance(getattr(params, key), dict) else getattr(params, key))
    
    # 设置GPU标志
    if USE_GPU:
        try:
            import cupy as cp
            local_params._use_gpu = True
            local_params._cp = cp
        except:
            local_params._use_gpu = False
            local_params._cp = None
    
    # 临时替换全局params（在函数内部）
    import A02_config_params
    original_params = A02_config_params.params
    A02_config_params.params = local_params
    
    try:
        depth_start_time = time.time()
        print(f"\n[进程 {os.getpid()}] 处理深度 {i+1}/{len(depthArray)}: {depth:.4f} mm ({1000*i/utot:.1f} μm)")
        
        local_params.geo['focalDepthTissue'] = depth
        
        # 运行蒙特卡洛光传输模拟
        print(f"[进程 {os.getpid()}] 1. 运行蒙特卡洛光传输模拟...")
        frac_abs, frac_trans, r1, d1, catcher, nlaunched, lostphotons = monte_carlo_light()
        
        absPortion = np.sum(catcher) / nlaunched
        lostPortions = lostphotons / nlaunched
        
        # 保存光分布数据
        if SAVE_DATA:
            with open(f'{output_dir}/light_510nm_{i}attleng.pkl', 'wb') as f:
                pickle.dump({
                    'frac_abs': frac_abs,
                    'frac_trans': frac_trans,
                    'r1': r1,
                    'd1': d1,
                    'catcher': catcher
                }, f)
        
        # 绘制光分布图
        if SAVE_PLOTS:
            local_params.opt['plotWavelength'] = None
            fig, ax = light_heat_plotter(r1, d1, frac_trans / np.max(frac_trans), 
                                         contours=[0.01, 0.1, 0.5],
                                         title=f'光分布 510nm, 深度={i}衰减长度')
            fig.savefig(f'{output_dir}/light_510nm_{i}attleng.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # 热扩散仿真参数
        local_params.geo['d_water'] = np.ceil((local_params.geo['wd'] - local_params.geo['d_glass'] - depth) / 
                                               local_params.geo['dstep']) * local_params.geo['dstep']
        added_Z = local_params.geo['d_water'] + local_params.geo['d_glass']
        local_params.geo['zrange'] = [-added_Z, 0, 6]
        
        times = [60]
        maxPower = 12 * np.exp(depth * utot)
        powers = powerPercent * maxPower
        
        # 扩展仿真体积
        frac_abs_padded = np.vstack([
            np.zeros((int(np.ceil(added_Z / local_params.geo['dstep'])), frac_abs.shape[1])),
            frac_abs
        ])
        
        # 初始化热场
        print(f"[进程 {os.getpid()}] 2. 初始化热场...")
        local_params.u_start = None
        _, _, _, _, _, u_start = heat_diffusion_light(
            frac_abs_padded, 0.03, 0, 60, 60, 0, [60], 0.1
        )
        local_params.u_start = u_start
        
        # 模拟不同功率
        print(f"[进程 {os.getpid()}] 3. 模拟不同功率...")
        u_time_all = []
        u_space_all = []
        t_all = []
        r2_all = []
        depth_all = []
        maxTemp_row = np.zeros(len(powerPercent))
        
        for power_idx, power in enumerate(powers):
            if power > 500:
                continue
            
            u_time, u_space, t, r2, depth_vals, _ = heat_diffusion_light(
                frac_abs_padded, 0.03, 0, max(times), max(times), 
                power, times, 0.1
            )
            
            step = max(1, (u_time.shape[1] - 1) // 2000)
            u_time = u_time[:, 1::step]
            
            maxTemp_row[power_idx] = np.percentile(u_space[:, :, 0].flatten(), 99)
            
            u_time_all.append(u_time)
            u_space_all.append(u_space)
            t_all.append(t)
            r2_all.append(r2)
            depth_all.append(depth_vals)
        
        # 保存热分布数据
        if SAVE_DATA:
            with open(f'{output_dir}/heat_510nm_{i}attleng.pkl', 'wb') as f:
                pickle.dump({
                    'u_time': u_time_all,
                    'u_space': u_space_all,
                    't': t_all,
                    'r2': r2_all,
                    'depth': depth_all
                }, f)
        
        # 绘制温度分布
        if SAVE_PLOTS:
            print(f"[进程 {os.getpid()}] 4. 生成温度分布图...")
            local_params.opt['plotWavelength'] = 'hot'
            
            for power_idx, power in enumerate(powers):
                if power > 500:
                    continue
                
                for time_idx, time_point in enumerate(times):
                    fig, ax = light_heat_plotter(
                        r2_all[power_idx], depth_all[power_idx], 
                        u_space_all[power_idx][:, :, time_idx],
                        contours=np.arange(25, 101, 1),
                        title=f'510nm, {int(power)}mW, {int(time_point)}s'
                    )
                    ax.set_ylim([-added_Z, 6])
                    ax.set_xlim([-6, 6])
                    filename = f'{output_dir}/TempMaps/510nm_{i}le_{int(power)}mW_{int(time_point)}s.png'
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    delta_T = u_space_all[power_idx][:, :, time_idx] - u_start.T
                    fig, ax = light_heat_plotter(
                        r2_all[power_idx], depth_all[power_idx], delta_T,
                        contours=np.arange(0, 71, 1),
                        title=f'ΔT 510nm, {int(power)}mW, {int(time_point)}s'
                    )
                    ax.set_ylim([-added_Z, 6])
                    ax.set_xlim([-6, 6])
                    filename = f'{output_dir}/DeltaTempMaps/deltaT_510nm_{i}le_{int(power)}mW_{int(time_point)}s.png'
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
        
        depth_time = time.time() - depth_start_time
        print(f"[进程 {os.getpid()}] 深度 {i+1} 完成，耗时: {depth_time:.2f} 秒")
        
        return {
            'i': i,
            'maxTemp': maxTemp_row,
            'absPortion': absPortion,
            'lostPortions': lostPortions
        }
    
    finally:
        # 恢复全局params
        A02_config_params.params = original_params


def main():
    """主函数"""
    
    print("="*60)
    print("510nm 双光子成像深度-功率扫描仿真")
    print("="*60)
    
    # 性能优化提示
    if PARALLEL_AVAILABLE:
        num_cores = cpu_count()
        print(f"\n检测到 {num_cores} 个CPU核心")
    
    # 检查Numba是否可用并验证
    numba_active = False
    try:
        import numba
        numba_active = True
        print(f"✓ Numba已安装 (版本: {numba.__version__})")
        # 检查JIT是否启用
        if numba.config.DISABLE_JIT == 0:
            print("  → JIT编译已启用")
        else:
            print("  ⚠ JIT编译被禁用")
    except ImportError:
        print("⚠ Numba未安装，建议安装以获得更好性能: pip install numba")
    
    # 检查CuPy是否可用并验证
    cupy_active = False
    try:
        import cupy as cp
        cupy_active = True
        print(f"✓ CuPy已安装 (版本: {cp.__version__})")
        try:
            # 尝试获取GPU信息（修复版本）
            device = cp.cuda.Device(0)
            device.use()
            mem_info = device.mem_info
            total_mem = mem_info[1] / 1e9
            free_mem = mem_info[0] / 1e9
            print(f"  → GPU设备: {device.id} (计算能力: {device.compute_capability})")
            print(f"  → GPU内存: 总计 {total_mem:.2f} GB, 可用 {free_mem:.2f} GB")
            print("  → GPU将用于蒙特卡洛和热扩散计算加速")
        except Exception as e:
            print(f"  ⚠ 无法访问GPU: {e}")
            cupy_active = False
    except ImportError:
        print("⚠ CuPy未安装，如需GPU加速请安装: pip install cupy-cuda12x")
    
    print("\n优化提示：代码已使用向量化操作加速关键计算")
    print("性能监控：CPU/GPU使用率低是正常的，因为代码主要是单线程顺序执行")
    
    # 注意：510nm波长需要更新组织光学参数
    # 这里使用的是估计值，实际应用需要实验测量或文献查找
    print("\n警告：510nm波长的组织光学参数需要根据实际情况调整！")
    print(f"当前使用参数：")
    print(f"  吸收系数: {params.opt['absTissue']} 1/mm")
    print(f"  散射系数: {params.opt['scatterTissue']} 1/mm")
    print(f"  各向异性因子: {params.opt['gTissue']}")
    print()
    
    # 模拟不同焦点深度的光传播
    utot = params.opt['scatterTissue'] + params.opt['absTissue']
    depthArray = np.arange(0, 8) / utot
    depthArray[0] = 0.01  # 避免零深度时的计算错误
    
    # 性能优化配置（在depthArray定义之后）
    USE_PARALLEL = PARALLEL_AVAILABLE and len(depthArray) > 1  # 是否使用并行
    USE_GPU = cupy_active  # 是否使用GPU加速（仅用于热扩散，需要CuPy）
    USE_GPU_MC = False  # 蒙特卡洛GPU加速（不推荐，通常更慢）
    SAVE_PLOTS = True  # 是否保存图片（设为False可大幅加速）
    SAVE_DATA = True   # 是否保存数据文件
    
    if USE_GPU:
        print(f"\n✓ GPU加速已启用，热扩散计算将在GPU上执行")
        if USE_GPU_MC:
            print(f"  ⚠ 蒙特卡洛GPU加速已启用（实验性，可能更慢）")
        else:
            print(f"  → 蒙特卡洛使用CPU版本（推荐，通常更快）")
        # 设置全局GPU标志
        try:
            import cupy as cp
            params._use_gpu = True
            params._cp = cp
        except:
            USE_GPU = False
            params._use_gpu = False
    else:
        params._use_gpu = False
        params._cp = None
    
    if USE_PARALLEL:
        num_cores_use = min(cpu_count(), len(depthArray))
        print(f"\n将使用 {num_cores_use} 个CPU核心并行处理深度扫描")
    else:
        print("\n使用单线程顺序执行")
        if not SAVE_PLOTS:
            print("⚠ 图片保存已禁用，可大幅加速")
    
    # 功率百分比数组
    powerPercent = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 
                            0.3, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0])
    
    maxTemp = np.zeros((len(powerPercent), len(depthArray)))
    absPortion = np.zeros(len(depthArray))
    lostPortions = np.zeros((4, len(depthArray)))
    
    # 创建输出目录
    output_dir = './510GaussDepthScan'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/TempMaps', exist_ok=True)
    os.makedirs(f'{output_dir}/DeltaTempMaps', exist_ok=True)
    
    # 主循环：遍历不同深度（支持并行处理）
    total_start_time = time.time()
    
    if USE_PARALLEL and len(depthArray) > 1:
        # 并行处理模式
        print(f"\n使用并行处理模式，{num_cores_use} 个进程")
        print("注意：GPU加速在并行模式下可能不可用（每个进程需要独立的GPU上下文）")
        
        # 准备参数列表
        process_args = [
            (i, depth, depthArray, powerPercent, utot, output_dir, 
             SAVE_PLOTS, SAVE_DATA, False)  # 并行模式下暂时禁用GPU
            for i, depth in enumerate(depthArray)
        ]
        
        # 使用进程池并行处理
        with Pool(processes=num_cores_use) as pool:
            results = pool.map(process_single_depth, process_args)
        
        # 收集结果
        for result in results:
            i = result['i']
            maxTemp[:, i] = result['maxTemp']
            absPortion[i] = result['absPortion']
            lostPortions[:, i] = result['lostPortions']
    
    else:
        # 顺序处理模式（原逻辑）
        for i, depth in enumerate(depthArray):
            depth_start_time = time.time()
            print(f"\n{'='*60}")
            print(f"处理深度 {i+1}/{len(depthArray)}: {depth:.4f} mm ({1000*i/utot:.1f} μm)")
            print(f"{'='*60}")
            
            params.geo['focalDepthTissue'] = depth
            
            # 运行蒙特卡洛光传输模拟（CPU版本，GPU版本通常更慢）
            print("\n1. 运行蒙特卡洛光传输模拟...")
            mc_start = time.time()
            if USE_GPU_MC and GPU_MC_AVAILABLE:
                print("   使用GPU加速（实验性，可能更慢）")
                frac_abs, frac_trans, r1, d1, catcher, nlaunched, lostphotons = monte_carlo_light_gpu()
            else:
                frac_abs, frac_trans, r1, d1, catcher, nlaunched, lostphotons = monte_carlo_light()
            mc_time = time.time() - mc_start
            print(f"   蒙特卡洛模拟耗时: {mc_time:.2f} 秒")
            
            print(f"\n散射长度 = {1000*i/utot:.1f} μm")
            print(f"吸收分数 = {np.sum(catcher)/nlaunched:.6f}")
            
            absPortion[i] = np.sum(catcher) / nlaunched
            lostPortions[:, i] = lostphotons / nlaunched
            
            # 绘制光分布图（可选）
            if SAVE_PLOTS:
                plot_start = time.time()
                params.opt['plotWavelength'] = None
                fig, ax = light_heat_plotter(r1, d1, frac_trans / np.max(frac_trans), 
                                             contours=[0.01, 0.1, 0.5],
                                             title=f'光分布 510nm, 深度={i}衰减长度')
                fig.savefig(f'{output_dir}/light_510nm_{i}attleng.png', dpi=300, bbox_inches='tight')
                plt.close(fig)  # 关闭图形释放内存
                print(f"   绘图耗时: {time.time() - plot_start:.2f} 秒")
            
            # 保存光分布数据（可选）
            if SAVE_DATA:
                save_start = time.time()
                with open(f'{output_dir}/light_510nm_{i}attleng.pkl', 'wb') as f:
                    pickle.dump({
                        'frac_abs': frac_abs,
                        'frac_trans': frac_trans,
                        'r1': r1,
                        'd1': d1,
                        'catcher': catcher
                    }, f)
                print(f"   数据保存耗时: {time.time() - save_start:.2f} 秒")
            
            # 热扩散仿真参数
            params.geo['d_water'] = np.ceil((params.geo['wd'] - params.geo['d_glass'] - depth) / 
                                           params.geo['dstep']) * params.geo['dstep']
            added_Z = params.geo['d_water'] + params.geo['d_glass']
            params.geo['zrange'] = [-added_Z, 0, 6]
            
            times = [60]  # 时间点 (s)
            maxPower = 12 * np.exp(depth * utot)  # 焦点处12mW
            powers = powerPercent * maxPower
            
            # 扩展仿真体积到脑上方的玻璃和水
            frac_abs_padded = np.vstack([
                np.zeros((int(np.ceil(added_Z / params.geo['dstep'])), frac_abs.shape[1])),
                frac_abs
            ])
            
            # 初始化输出变量
            u_time_all = []
            u_space_all = []
            t_all = []
            r2_all = []
            depth_all = []
            
            # 初始化：运行模型达到稳态
            print("\n2. 初始化热场（达到稳态）...")
            init_start = time.time()
            params.u_start = None
            
            if USE_GPU and GPU_HEAT_AVAILABLE:
                _, _, _, _, _, u_start = heat_diffusion_light_opt(
                    frac_abs_padded, 0.03, 0, 60, 60, 0, [60], 0.1
                )
            else:
                _, _, _, _, _, u_start = heat_diffusion_light(
                    frac_abs_padded, 0.03, 0, 60, 60, 0, [60], 0.1
                )
            params.u_start = u_start
            print(f"   初始化耗时: {time.time() - init_start:.2f} 秒")
            
            # 模拟各种功率水平下的加热
            print("\n3. 模拟不同功率下的加热...")
            heat_start = time.time()
            for power_idx, power in enumerate(powers):
                if power > 500:
                    continue
                
                power_start = time.time()
                print(f"\n  功率 {power_idx+1}/{len(powers)}: {power:.2f} mW", end='')
                
                if USE_GPU and GPU_HEAT_AVAILABLE:
                    u_time, u_space, t, r2, depth_vals, _ = heat_diffusion_light_opt(
                        frac_abs_padded, 0.03, 0, max(times), max(times), 
                        power, times, 0.1
                    )
                else:
                    u_time, u_space, t, r2, depth_vals, _ = heat_diffusion_light(
                        frac_abs_padded, 0.03, 0, max(times), max(times), 
                        power, times, 0.1
                    )
                
                # 缩减时间序列矩阵
                step = max(1, (u_time.shape[1] - 1) // 2000)
                u_time = u_time[:, 1::step]
                
                # 计算最大温度（99百分位数）
                maxTemp[power_idx, i] = np.percentile(u_space[:, :, 0].flatten(), 99)
                
                u_time_all.append(u_time)
                u_space_all.append(u_space)
                t_all.append(t)
                r2_all.append(r2)
                depth_all.append(depth_vals)
                
                print(f" ({time.time() - power_start:.1f}s)")
            
            print(f"   总加热仿真耗时: {time.time() - heat_start:.2f} 秒")
            
            # 保存热分布数据（可选）
            if SAVE_DATA:
                save_start = time.time()
                with open(f'{output_dir}/heat_510nm_{i}attleng.pkl', 'wb') as f:
                    pickle.dump({
                        'u_time': u_time_all,
                        'u_space': u_space_all,
                        't': t_all,
                        'r2': r2_all,
                        'depth': depth_all
                    }, f)
                print(f"   数据保存耗时: {time.time() - save_start:.2f} 秒")
            
            # 绘制不同功率和时间下的温度分布（可选）
            if SAVE_PLOTS:
                print("\n4. 生成温度分布图...")
                plot_start = time.time()
                params.opt['plotWavelength'] = 'hot'
                
                for power_idx, power in enumerate(powers):
                    if power > 500:
                        continue
                    
                    for time_idx, time_point in enumerate(times):
                        # 温度图
                        fig, ax = light_heat_plotter(
                            r2_all[power_idx], depth_all[power_idx], 
                            u_space_all[power_idx][:, :, time_idx],
                            contours=np.arange(25, 101, 1),
                            title=f'510nm, {int(power)}mW, {int(time_point)}s'
                        )
                        ax.set_ylim([-added_Z, 6])
                        ax.set_xlim([-6, 6])
                        filename = f'{output_dir}/TempMaps/510nm_{i}le_{int(power)}mW_{int(time_point)}s.png'
                        fig.savefig(filename, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        
                        # 温度差图
                        delta_T = u_space_all[power_idx][:, :, time_idx] - u_start.T
                        fig, ax = light_heat_plotter(
                            r2_all[power_idx], depth_all[power_idx], delta_T,
                            contours=np.arange(0, 71, 1),
                            title=f'ΔT 510nm, {int(power)}mW, {int(time_point)}s'
                        )
                        ax.set_ylim([-added_Z, 6])
                        ax.set_xlim([-6, 6])
                        filename = f'{output_dir}/DeltaTempMaps/deltaT_510nm_{i}le_{int(power)}mW_{int(time_point)}s.png'
                        fig.savefig(filename, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                
                print(f"   绘图总耗时: {time.time() - plot_start:.2f} 秒")
            
            depth_time = time.time() - depth_start_time
            print(f"\n深度 {i+1} 总耗时: {depth_time:.2f} 秒")
            print(f"预计剩余时间: {(depth_time * (len(depthArray) - i - 1)) / 60:.1f} 分钟")
    
    # 保存最终参数和结果
    print("\n5. 保存最终结果...")
    with open(f'{output_dir}/params_510nm.pkl', 'wb') as f:
        pickle.dump({
            'params': params,
            'maxTemp': maxTemp,
            'absPortion': absPortion,
            'lostPortions': lostPortions,
            'depthArray': depthArray,
            'powerPercent': powerPercent
        }, f)
    
    total_time = time.time() - total_start_time
    print("\n" + "="*60)
    print("仿真完成！")
    print(f"总耗时: {total_time/60:.1f} 分钟 ({total_time:.1f} 秒)")
    print(f"结果保存在: {output_dir}")
    print("="*60)
    
    # 性能分析总结
    print("\n性能分析:")
    print(f"  平均每个深度耗时: {total_time/len(depthArray):.1f} 秒")
    print(f"  如果禁用绘图，预计可节省: ~30-50% 时间")
    print(f"  如果使用并行处理，预计可加速: ~{min(cpu_count(), len(depthArray))}x")
    print("\n优化建议:")
    print("  1. 设置 SAVE_PLOTS = False 可大幅加速（测试时）")
    print("  2. 减少 powerPercent 数组长度可减少计算量")
    print("  3. 减少深度点数可加快整体速度")
    print("="*60)


if __name__ == '__main__':
    main()
