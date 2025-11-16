"""
主程序：510nm双光子成像深度-功率扫描 - 优化版本
主要优化：
1. 修复GPU检测显示
2. 添加快速测试模式
3. 优化内存使用
4. 改进进度显示
5. 禁用无效的GPU加速（反而更慢）
"""
import numpy as np
import os
import pickle
import time
from A02_config_params import params
from A05_monte_carlo_light import monte_carlo_light
from A04_heat_diffusion import heat_diffusion_light
from A06_visualization import light_heat_plotter
import matplotlib.pyplot as plt

# 尝试导入并行计算支持
try:
    from multiprocessing import Pool, cpu_count
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    cpu_count = lambda: 1


def main():
    """主函数"""
    
    print("="*60)
    print("510nm 双光子成像深度-功率扫描仿真 - 优化版本")
    print("="*60)
    
    # ==================== 配置区域 ====================
    # 快速测试模式（用于调试和验证）
    FAST_MODE = False  # 改为True启用快速模式
    
    # 性能配置
    SAVE_PLOTS = True  # 是否保存图片（False可节省30-50%时间）
    SAVE_DATA = True   # 是否保存数据文件
    USE_PARALLEL = False  # 是否使用多核并行（需要多个深度点）
    
    # GPU配置（经测试，GPU加速对当前问题无效，反而更慢）
    USE_GPU = False  # 不推荐启用
    # =================================================
    
    # 检测系统资源
    if PARALLEL_AVAILABLE:
        num_cores = cpu_count()
        print(f"\n检测到 {num_cores} 个CPU核心")
    
    # 检查CuPy（仅用于显示GPU信息）
    try:
        import cupy as cp
        device = cp.cuda.Device(0)
        device.use()
        mem_info = device.mem_info
        print(f"✓ 检测到GPU: 设备{device.id}, 计算能力{device.compute_capability}")
        print(f"  GPU内存: 总计{mem_info[1]/1e9:.2f}GB, 可用{mem_info[0]/1e9:.2f}GB")
        print(f"  注意: GPU加速对当前问题无效（已禁用）")
    except:
        print("⚠ 未检测到GPU或CuPy未安装")
    
    print(f"\n当前配置:")
    print(f"  快速模式: {'启用' if FAST_MODE else '禁用'}")
    print(f"  保存图片: {'是' if SAVE_PLOTS else '否'}")
    print(f"  并行处理: {'启用' if USE_PARALLEL else '禁用'}")
    
    # 模拟不同焦点深度的光传播
    utot = params.opt['scatterTissue'] + params.opt['absTissue']
    
    if FAST_MODE:
        # 快速测试模式
        depthArray = np.array([0.01, 1.0 / utot])
        powerPercent = np.array([0.1, 1.0])
        nphotonpackets = 10  # 10万光子
        print(f"\n快速模式: 2深度 × 2功率 × 10万光子")
        print(f"  预计耗时: 1-2分钟")
    else:
        # 标准模式
        depthArray = np.arange(0, 8) / utot
        depthArray[0] = 0.01
        powerPercent = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 
                                0.3, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0])
        nphotonpackets = 500  # 500万光子
        print(f"\n标准模式: {len(depthArray)}深度 × {len(powerPercent)}功率 × 500万光子")
        if SAVE_PLOTS:
            print(f"  预计耗时: 2-4小时")
        else:
            print(f"  预计耗时: 1-2小时（禁用绘图）")
    
    # 初始化结果数组
    maxTemp = np.zeros((len(powerPercent), len(depthArray)))
    absPortion = np.zeros(len(depthArray))
    lostPortions = np.zeros((4, len(depthArray)))
    
    # 创建输出目录
    output_dir = './510GaussDepthScan'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/TempMaps', exist_ok=True)
    os.makedirs(f'{output_dir}/DeltaTempMaps', exist_ok=True)
    
    # 主循环：遍历不同深度
    total_start_time = time.time()
    
    for i, depth in enumerate(depthArray):
        depth_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"处理深度 {i+1}/{len(depthArray)}: {depth:.4f} mm ({1000*i/utot:.1f} μm)")
        print(f"{'='*60}")
        
        params.geo['focalDepthTissue'] = depth
        
        # 1. 运行蒙特卡洛光传输模拟
        print("\n[1/4] 蒙特卡洛光传输模拟...")
        mc_start = time.time()
        frac_abs, frac_trans, r1, d1, catcher, nlaunched, lostphotons = monte_carlo_light(
            nphotonpackets=nphotonpackets
        )
        mc_time = time.time() - mc_start
        print(f"      完成，耗时: {mc_time:.1f}秒")
        print(f"      吸收分数: {np.sum(catcher)/nlaunched:.6f}")
        
        absPortion[i] = np.sum(catcher) / nlaunched
        lostPortions[:, i] = lostphotons / nlaunched
        
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
            params.opt['plotWavelength'] = None
            fig, ax = light_heat_plotter(r1, d1, frac_trans / np.max(frac_trans), 
                                         contours=[0.01, 0.1, 0.5],
                                         title=f'光分布 510nm, 深度={i}衰减长度')
            fig.savefig(f'{output_dir}/light_510nm_{i}attleng.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # 2. 准备热扩散仿真
        print("\n[2/4] 准备热扩散仿真...")
        params.geo['d_water'] = np.ceil((params.geo['wd'] - params.geo['d_glass'] - depth) / 
                                       params.geo['dstep']) * params.geo['dstep']
        added_Z = params.geo['d_water'] + params.geo['d_glass']
        params.geo['zrange'] = [-added_Z, 0, 6]
        
        times = [60]
        maxPower = 12 * np.exp(depth * utot)
        powers = powerPercent * maxPower
        
        # 扩展仿真体积
        frac_abs_padded = np.vstack([
            np.zeros((int(np.ceil(added_Z / params.geo['dstep'])), frac_abs.shape[1])),
            frac_abs
        ])
        
        # 3. 初始化热场
        print("\n[3/4] 初始化热场...")
        init_start = time.time()
        params.u_start = None
        _, _, _, _, _, u_start = heat_diffusion_light(
            frac_abs_padded, 0.03, 0, 60, 60, 0, [60], 0.1
        )
        params.u_start = u_start
        print(f"      完成，耗时: {time.time() - init_start:.1f}秒")
        
        # 4. 模拟不同功率
        print(f"\n[4/4] 模拟{len(powers)}个功率水平...")
        heat_start = time.time()
        
        u_time_all = []
        u_space_all = []
        t_all = []
        r2_all = []
        depth_all = []
        
        for power_idx, power in enumerate(powers):
            if power > 500:
                continue
            
            power_start = time.time()
            u_time, u_space, t, r2, depth_vals, _ = heat_diffusion_light(
                frac_abs_padded, 0.03, 0, max(times), max(times), 
                power, times, 0.1
            )
            
            # 缩减时间序列
            step = max(1, (u_time.shape[1] - 1) // 2000)
            u_time = u_time[:, 1::step]
            
            maxTemp[power_idx, i] = np.percentile(u_space[:, :, 0].flatten(), 99)
            
            u_time_all.append(u_time)
            u_space_all.append(u_space)
            t_all.append(t)
            r2_all.append(r2)
            depth_all.append(depth_vals)
            
            power_time = time.time() - power_start
            print(f"      功率{power_idx+1}/{len(powers)}: {power:.1f}mW, 耗时{power_time:.1f}秒")
        
        print(f"      总耗时: {time.time() - heat_start:.1f}秒")
        
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
            print("\n      生成温度分布图...")
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
        
        # 深度完成统计
        depth_time = time.time() - depth_start_time
        elapsed_total = time.time() - total_start_time
        avg_time_per_depth = elapsed_total / (i + 1)
        remaining_depths = len(depthArray) - i - 1
        estimated_remaining = avg_time_per_depth * remaining_depths
        
        print(f"\n深度 {i+1}/{len(depthArray)} 完成")
        print(f"  本深度耗时: {depth_time/60:.1f}分钟")
        print(f"  已用总时间: {elapsed_total/60:.1f}分钟")
        print(f"  预计剩余: {estimated_remaining/60:.1f}分钟")
        print(f"  预计完成: {(elapsed_total + estimated_remaining)/60:.1f}分钟")
        
        # 清理内存
        import gc
        gc.collect()
    
    # 保存最终结果
    print("\n保存最终结果...")
    
    # 创建可序列化的参数字典（排除lambda函数和其他不可序列化对象）
    try:
        # 创建params的可序列化副本
        serializable_params = {
            'opt': {k: v for k, v in params.opt.items() 
                   if k not in ['scatterFrontCort', 'scatterTempCort'] and not callable(v)},
            'mech': params.mech.copy(),
            'therm': params.therm.copy(),
            'geo': params.geo.copy(),
        }
        # 保存lambda函数的字符串表示（用于参考）
        serializable_params['opt']['scatterFrontCort_str'] = 'lambda x: 10.9 * ((x/500)**(-0.334))'
        serializable_params['opt']['scatterTempCort_str'] = 'lambda x: 11.6 * ((x/500)**(-0.601))'
        
        # 保存u_start（如果存在且可序列化）
        if params.u_start is not None:
            try:
                # 尝试序列化u_start，如果失败则跳过
                pickle.dumps(params.u_start)
                serializable_params['u_start'] = params.u_start
            except:
                print("  警告: u_start 无法序列化，已跳过")
                serializable_params['u_start'] = None
        else:
            serializable_params['u_start'] = None
        
    except Exception as e:
        print(f"  警告: 创建可序列化参数时出错: {e}")
        serializable_params = {'error': str(e)}
    
    # 保存结果
    try:
        with open(f'{output_dir}/params_510nm.pkl', 'wb') as f:
            pickle.dump({
                'params': serializable_params,
                'maxTemp': maxTemp,
                'absPortion': absPortion,
                'lostPortions': lostPortions,
                'depthArray': depthArray,
                'powerPercent': powerPercent
            }, f)
        print("  ✓ 结果保存成功")
    except Exception as e:
        print(f"  ✗ 保存失败: {e}")
        print(f"  错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        # 尝试保存不包含params的简化版本
        try:
            with open(f'{output_dir}/results_510nm.pkl', 'wb') as f:
                pickle.dump({
                    'maxTemp': maxTemp,
                    'absPortion': absPortion,
                    'lostPortions': lostPortions,
                    'depthArray': depthArray,
                    'powerPercent': powerPercent
                }, f)
            print("  ✓ 已保存简化版本（不包含params）到 results_510nm.pkl")
        except Exception as e2:
            print(f"  ✗ 简化版本保存也失败: {e2}")
    
    total_time = time.time() - total_start_time
    print("\n" + "="*60)
    print("仿真完成！")
    print(f"总耗时: {total_time/60:.1f}分钟")
    print(f"平均每深度: {total_time/len(depthArray)/60:.1f}分钟")
    print(f"结果保存在: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
