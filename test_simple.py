"""
简单测试脚本
快速验证各模块是否正常工作
"""
import numpy as np
from A02_config_params import params
from A03_gauss_init import gauss_init
from A05_monte_carlo_light import monte_carlo_light
from A04_heat_diffusion import heat_diffusion_light
from A06_visualization import light_heat_plotter
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def test_gauss_init():
    """测试高斯光束初始化"""
    print("测试1: 高斯光束初始化...")
    cors, dirs = gauss_init(1000, params.geo['w0'], 0, 1.0)
    
    assert cors.shape == (3, 1000), "坐标数组形状错误"
    assert dirs.shape == (3, 1000), "方向数组形状错误"
    
    # 检查方向向量是否归一化
    mag = np.sqrt(np.sum(dirs**2, axis=0))
    assert np.allclose(mag, 1.0, atol=1e-6), "方向向量未归一化"
    
    print("  ✓ 高斯光束初始化测试通过")


def test_monte_carlo_light():
    """测试蒙特卡洛光传输（快速版本）"""
    print("\n测试2: 蒙特卡洛光传输（使用少量光子）...")
    
    # 使用很少的光子包进行快速测试
    frac_abs, frac_trans, r, depth, catcher, nlaunched, lostphotons = \
        monte_carlo_light(nphotonpackets=1, zrange=[0, 0, 2], rmax=2, dstep=0.05)
    
    assert frac_abs.shape[0] > 0, "吸收分数数组为空"
    assert frac_trans.shape[0] > 0, "透射分数数组为空"
    assert len(r) > 0, "径向坐标为空"
    assert len(depth) > 0, "深度坐标为空"
    
    # 检查能量守恒（粗略检查）
    total_absorbed = np.sum(catcher) / nlaunched
    total_lost = np.sum(lostphotons) / nlaunched
    print(f"  吸收分数: {total_absorbed:.4f}")
    print(f"  丢失分数: {total_lost:.4f}")
    print(f"  总计: {total_absorbed + total_lost:.4f}")
    
    print("  ✓ 蒙特卡洛光传输测试通过")
    
    return frac_abs, r, depth


def test_heat_diffusion(frac_abs):
    """测试热扩散（快速版本）"""
    print("\n测试3: 热扩散模拟（短时间）...")
    
    # 设置参数
    params.geo['d_water'] = 0.5
    params.geo['zrange'] = [0, 0, 2]
    
    # 扩展吸收分数数组
    added_Z = params.geo['d_water'] + params.geo['d_glass']
    frac_abs_padded = np.vstack([
        np.zeros((int(np.ceil(added_Z / params.geo['dstep'])), frac_abs.shape[1])),
        frac_abs
    ])
    
    # 运行短时间模拟
    u_time, u_space, t, r, depth, uRAW = heat_diffusion_light(
        frac_abs_padded, u_step=0.05, t_on=0, t_off=5, t_max=5, 
        Power=10, t_save=[5], r_avg=0.1
    )
    
    assert u_time.shape[0] > 0, "时间序列温度为空"
    assert u_space.shape[0] > 0, "空间温度分布为空"
    assert len(t) > 0, "时间数组为空"
    
    # 检查温度的物理合理性
    max_temp = np.max(u_space)
    min_temp = np.min(u_space)
    print(f"  最高温度: {max_temp:.2f} °C")
    print(f"  最低温度: {min_temp:.2f} °C")
    
    assert min_temp > 0 and max_temp < 100, "温度超出物理合理范围"
    
    print("  ✓ 热扩散模拟测试通过")
    
    return u_space, r, depth


def test_visualization(u_space, r, depth):
    """测试可视化"""
    print("\n测试4: 可视化...")
    
    params.opt['plotWavelength'] = 'hot'
    fig, ax = light_heat_plotter(r, depth, u_space[:, :, 0], 
                                 contours=[30, 35, 40],
                                 title='测试温度分布')
    
    # 保存测试图
    fig.savefig('test_visualization.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print("  ✓ 可视化测试通过")
    print("  测试图已保存为: test_visualization.png")


def main():
    """运行所有测试"""
    print("="*60)
    print("开始运行简单测试...")
    print("="*60)
    
    try:
        # 测试1: 高斯光束初始化
        test_gauss_init()
        
        # 测试2: 蒙特卡洛光传输
        frac_abs, r, depth = test_monte_carlo_light()
        
        # 测试3: 热扩散
        u_space, r, depth = test_heat_diffusion(frac_abs)
        
        # 测试4: 可视化
        test_visualization(u_space, r, depth)
        
        print("\n" + "="*60)
        print("所有测试通过！✓")
        print("="*60)
        print("\n提示：")
        print("1. 可以运行 main_depth_power_scan_532nm.py 进行完整仿真")
        print("2. 完整仿真需要较长时间，请耐心等待")
        print("3. 记得根据实际情况调整532nm的组织光学参数")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
