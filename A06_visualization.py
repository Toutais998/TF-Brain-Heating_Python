"""
可视化模块
绘制光热分布图
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from A02_config_params import params

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 尝试设置中文字体（Windows系统）
try:
    import platform
    if platform.system() == 'Windows':
        # Windows系统常见中文字体
        font_list = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
        for font_name in font_list:
            try:
                matplotlib.font_manager.findfont(matplotlib.font_manager.FontProperties(family=font_name))
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                break
            except:
                continue
except:
    pass


def light_heat_plotter(r, depth, plot_mat, contours=None, title=''):
    """
    光热分布绘图函数
    
    参数:
        r: 径向距离向量 (mm)
        depth: 深度向量 (mm)
        plot_mat: r×深度矩阵
        contours: 等高线数值列表
        title: 图形标题
    
    返回:
        fig: 图形对象
        ax: 坐标轴对象
    """
    
    FOV = params.geo['FOV']
    wavelength = params.opt['plotWavelength']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 确保 plot_mat 的维度正确：应该是 (len(r), len(depth))
    # 如果输入是 (len(depth), len(r))，需要转置
    if plot_mat.shape[0] == len(depth) and plot_mat.shape[1] == len(r):
        plot_mat = plot_mat.T
    
    # 创建镜像数据（完整的径向分布）
    r_full = np.concatenate([-r[::-1][:-1], r])
    plot_mat_full = np.concatenate([plot_mat[::-1, :][:-1, :], plot_mat], axis=0)
    
    # 绘制三维表面图（以2D形式显示）
    X, Y = np.meshgrid(r_full, depth)
    
    # plot_mat_full 应该是 (len(r_full), len(depth))
    # 转置后应该是 (len(depth), len(r_full)) 以匹配 meshgrid 的输出
    plot_data = plot_mat_full.T
    
    # 选择色图
    if wavelength is None:
        cmap = 'gray'
    elif wavelength == 'hot':
        cmap = 'hot'
    else:
        cmap = 'gray'
    
    # 绘制主图
    im = ax.pcolormesh(X, Y, plot_data, cmap=cmap, shading='auto')
    
    # 绘制等高线
    if contours is not None:
        if wavelength == 'hot':
            contour_color = 'k'
        else:
            contour_color = 'y'
        cs = ax.contour(X, Y, plot_data, levels=contours, 
                       colors=contour_color, linewidths=1.5)
        ax.clabel(cs, inline=True, fontsize=8)
    
    # 绘制盖玻片边界
    rg = params.geo['r_glass']
    dg = params.geo['d_glass']
    ax.plot([-rg, rg, rg, -rg, -rg], [-dg, -dg, 0, 0, -dg], 
            'c-', linewidth=2.5, label='Cover Glass' if 'Cover Glass' in title else '盖玻片')
    
    # 绘制焦点区域边界
    focus_R = FOV / 2
    focus_Z = params.geo['focalDepthTissue']
    ax.plot([-focus_R, focus_R, focus_R, -focus_R, -focus_R], 
            [focus_Z, focus_Z, focus_Z, focus_Z, focus_Z], 
            'm-', linewidth=2.5, label='Focus Region' if 'Focus' in title else '焦点区域')
    
    # 设置坐标轴（使用中文标签）
    ax.set_xlabel('Radial Distance (mm)' if 'Radial' in str(ax.get_xlabel()) else '径向距离 (mm)', fontsize=12)
    ax.set_ylabel('Depth (mm)' if 'Depth' in str(ax.get_ylabel()) else '深度 (mm)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()  # 反转Y轴（深度向下为正）
    ax.set_aspect('equal')
    ax.legend()
    
    # 添加颜色条
    if wavelength is None:
        cbar_label = '强度'
    elif wavelength == 'hot':
        cbar_label = 'Temperature (°C)'
    else:
        cbar_label = '强度'
    plt.colorbar(im, ax=ax, label=cbar_label)
    
    plt.tight_layout()
    
    return fig, ax


def save_figure(fig, filename):
    """
    保存图形
    
    参数:
        fig: 图形对象
        filename: 文件名（包含路径）
    """
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_temperature_map(r, depth, u_space, power, time_point, wavelength, 
                        depth_index, output_dir):
    """
    绘制温度分布图
    
    参数:
        r: 径向距离向量
        depth: 深度向量
        u_space: 温度空间分布
        power: 功率 (mW)
        time_point: 时间点 (s)
        wavelength: 波长 (nm)
        depth_index: 深度索引
        output_dir: 输出目录
    """
    params.opt['plotWavelength'] = 'hot'
    
    fig, ax = light_heat_plotter(r, depth, u_space[:, :, 0], 
                                 contours=np.arange(25, 101, 1),
                                 title=f'{wavelength}nm, {power}mW, {time_point}s')
    
    # 设置温度范围
    ax.set_ylim([-0.35, 6])
    ax.set_xlim([-6, 6])
    
    filename = f'{output_dir}/TempMaps/{wavelength}nm_{depth_index}le_{int(power)}mW_{int(time_point)}s.png'
    save_figure(fig, filename)


def plot_delta_temperature_map(r, depth, u_space, u_start, power, time_point, 
                               wavelength, depth_index, output_dir):
    """
    绘制温度差ΔT分布图
    
    参数:
        r: 径向距离向量
        depth: 深度向量
        u_space: 温度空间分布
        u_start: 初始温度分布
        power: 功率 (mW)
        time_point: 时间点 (s)
        wavelength: 波长 (nm)
        depth_index: 深度索引
        output_dir: 输出目录
    """
    params.opt['plotWavelength'] = 'hot'
    
    delta_T = u_space[:, :, 0] - u_start.T
    
    fig, ax = light_heat_plotter(r, depth, delta_T, 
                                 contours=np.arange(0, 71, 1),
                                 title=f'ΔT {wavelength}nm, {power}mW, {time_point}s')
    
    # 设置温度范围
    ax.set_ylim([-0.35, 6])
    ax.set_xlim([-6, 6])
    
    filename = f'{output_dir}/DeltaTempMaps/deltaT_{wavelength}nm_{depth_index}le_{int(power)}mW_{int(time_point)}s.png'
    save_figure(fig, filename)
