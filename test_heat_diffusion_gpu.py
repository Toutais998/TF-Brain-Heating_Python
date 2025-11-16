"""
热扩散GPU加速性能测试
专注于测试热扩散计算的GPU加速效果
"""
import numpy as np
import time
import cupy as cp
from A02_config_params import params

print("="*60)
print("热扩散GPU加速性能测试")
print("="*60)

# 1. GPU设备信息
print("\n1. GPU设备信息:")
device = cp.cuda.Device(0)
device.use()
mem_info = device.mem_info
print(f"   GPU设备ID: {device.id}")
print(f"   计算能力: {device.compute_capability}")
print(f"   总内存: {mem_info[1]/1e9:.2f} GB")
print(f"   可用内存: {mem_info[0]/1e9:.2f} GB")

# 2. 设置测试参数
print("\n2. 准备测试数据...")
params.geo['d_water'] = 1.0
params.geo['d_glass'] = 0.16
params.geo['d_skull'] = 0.14
params.geo['d_boneCortical'] = 0.01
params.geo['r_glass'] = 2.0
params.geo['dstep'] = 0.01
params.geo['zrange'] = [-1.16, 0, 6]
params.geo['rmax'] = 6

# 创建测试数据（模拟光吸收分布）
test_size = 600
test_frac_abs = np.random.rand(test_size, test_size) * 0.01

print(f"   测试网格大小: {test_size}x{test_size}")
print(f"   模拟时间: 10秒")

# 3. 测试CPU版本
print("\n3. 测试CPU版本...")
params._use_gpu = False
from A04_heat_diffusion import heat_diffusion_light as heat_cpu

start = time.time()
u_time_cpu, u_space_cpu, t_cpu, r_cpu, depth_cpu, uRAW_cpu = heat_cpu(
    test_frac_abs, 0.03, 0, 10, 10, 1, [10], 0.1
)
cpu_time = time.time() - start
print(f"   CPU耗时: {cpu_time:.2f} 秒")
print(f"   最终温度范围: {uRAW_cpu.min():.2f}°C - {uRAW_cpu.max():.2f}°C")

# 4. 测试GPU版本
print("\n4. 测试GPU版本...")
params._use_gpu = True
params._cp = cp
from A04_heat_diffusion_optimized import heat_diffusion_light as heat_gpu

start = time.time()
u_time_gpu, u_space_gpu, t_gpu, r_gpu, depth_gpu, uRAW_gpu = heat_gpu(
    test_frac_abs, 0.03, 0, 10, 10, 1, [10], 0.1
)
gpu_time = time.time() - start
print(f"   GPU耗时: {gpu_time:.2f} 秒")
print(f"   最终温度范围: {uRAW_gpu.min():.2f}°C - {uRAW_gpu.max():.2f}°C")

# 5. 验证结果一致性
print("\n5. 验证结果一致性...")
max_diff = np.abs(uRAW_cpu - uRAW_gpu).max()
mean_diff = np.abs(uRAW_cpu - uRAW_gpu).mean()
print(f"   最大差异: {max_diff:.6f}°C")
print(f"   平均差异: {mean_diff:.6f}°C")

if max_diff < 0.1:
    print("   ✓ 结果一致性验证通过")
else:
    print("   ⚠ 结果存在较大差异，请检查")

# 6. 性能总结
print("\n" + "="*60)
print("性能测试总结:")
print("="*60)
speedup = cpu_time / gpu_time
print(f"CPU时间: {cpu_time:.2f} 秒")
print(f"GPU时间: {gpu_time:.2f} 秒")
print(f"加速比: {speedup:.2f}x")

if speedup > 3:
    print("\n✓ GPU加速效果显著！")
    print(f"  对于完整模拟（8深度×18功率），预计可节省:")
    print(f"  {(1 - 1/speedup) * 100:.0f}% 的热扩散计算时间")
elif speedup > 1.5:
    print("\n✓ GPU加速有效")
    print("  建议在大规模模拟中使用")
else:
    print("\n⚠ GPU加速效果不明显")
    print("  可能原因：")
    print("  - 网格规模较小")
    print("  - GPU启动开销")
    print("  - 数据传输开销")

print("\n推荐配置:")
print("  USE_GPU = True  # 启用GPU加速热扩散")
print("  USE_GPU_MC = False  # 禁用蒙特卡洛GPU加速")
print("="*60)
