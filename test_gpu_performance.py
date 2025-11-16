"""
GPU性能测试脚本
比较CPU vs GPU版本的性能差异
"""
import numpy as np
import time
import cupy as cp

print("="*60)
print("GPU性能测试")
print("="*60)

# 1. 测试GPU基本信息
print("\n1. GPU设备信息:")
device = cp.cuda.Device(0)
device.use()
mem_info = device.mem_info
print(f"   GPU设备ID: {device.id}")
print(f"   计算能力: {device.compute_capability}")
print(f"   总内存: {mem_info[1]/1e9:.2f} GB")
print(f"   可用内存: {mem_info[0]/1e9:.2f} GB")

# 2. 测试蒙特卡洛模拟性能
print("\n2. 测试蒙特卡洛光传输模拟:")
print("   准备测试数据...")

from A02_config_params import params
from A05_monte_carlo_light import monte_carlo_light
from A05_monte_carlo_light_gpu import monte_carlo_light_gpu

# 设置测试参数
params.geo['focalDepthTissue'] = 0.01
params.geo['d_water'] = 1.0
params.geo['d_glass'] = 0.16
params.geo['d_skull'] = 0.14
params.geo['d_boneCortical'] = 0.01
params.geo['r_glass'] = 2.0
params.geo['w0'] = 5.3
params.geo['FOV'] = 0.3
params.opt['absTissue'] = 0.04
params.opt['scatterTissue'] = 10.6
params.opt['gTissue'] = 0.88
params.opt['nWater'] = 1.328
params.opt['nTissue'] = 1.36

print("\n   测试CPU版本（小规模）...")
start = time.time()
frac_abs_cpu, _, _, _, _, _, _ = monte_carlo_light(nphotonpackets=10)  # 10万光子
cpu_time = time.time() - start
print(f"   CPU耗时: {cpu_time:.2f} 秒")

print("\n   测试GPU版本（小规模）...")
start = time.time()
frac_abs_gpu, _, _, _, _, _, _ = monte_carlo_light_gpu(nphotonpackets=10)  # 10万光子
gpu_time = time.time() - start
print(f"   GPU耗时: {gpu_time:.2f} 秒")

speedup = cpu_time / gpu_time
print(f"\n   加速比: {speedup:.2f}x")

# 3. 测试热扩散性能
print("\n3. 测试热扩散计算:")
print("   准备测试数据...")

# 创建测试数据
test_frac_abs = np.random.rand(600, 600) * 0.01

# CPU版本
print("\n   测试CPU版本...")
params._use_gpu = False
from A04_heat_diffusion import heat_diffusion_light as heat_cpu
start = time.time()
_, _, _, _, _, _ = heat_cpu(test_frac_abs, 0.03, 0, 10, 10, 1, [10], 0.1)
cpu_heat_time = time.time() - start
print(f"   CPU耗时: {cpu_heat_time:.2f} 秒")

# GPU版本
print("\n   测试GPU版本...")
params._use_gpu = True
params._cp = cp
from A04_heat_diffusion_optimized import heat_diffusion_light as heat_gpu
start = time.time()
_, _, _, _, _, _ = heat_gpu(test_frac_abs, 0.03, 0, 10, 10, 1, [10], 0.1)
gpu_heat_time = time.time() - start
print(f"   GPU耗时: {gpu_heat_time:.2f} 秒")

speedup_heat = cpu_heat_time / gpu_heat_time
print(f"\n   加速比: {speedup_heat:.2f}x")

# 4. 总结
print("\n" + "="*60)
print("性能测试总结:")
print("="*60)
print(f"蒙特卡洛模拟加速: {speedup:.2f}x")
print(f"热扩散计算加速: {speedup_heat:.2f}x")
print(f"预计总体加速: {(speedup + speedup_heat)/2:.2f}x")
print("\n建议：")
if speedup > 2:
    print("✓ GPU加速效果显著，建议使用GPU版本")
else:
    print("⚠ GPU加速效果不明显，可能受限于数据传输开销")
print("="*60)
