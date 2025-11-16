"""
性能监控工具
用于检查Numba和CuPy的激活状态，以及性能分析
"""
import sys
import time
import numpy as np

def check_numba():
    """检查Numba是否安装和激活"""
    print("=" * 60)
    print("Numba 状态检查")
    print("=" * 60)
    
    try:
        import numba
        print(f"✓ Numba已安装")
        print(f"  版本: {numba.__version__}")
        print(f"  JIT状态: {'启用' if numba.config.DISABLE_JIT == 0 else '禁用'}")
        
        # 测试JIT编译
        from numba import jit
        
        @jit(nopython=True)
        def test_func(x):
            return x * 2 + 1
        
        # 第一次调用会编译
        start = time.time()
        result1 = test_func(np.array([1, 2, 3]))
        first_call_time = time.time() - start
        
        # 第二次调用应该更快（已编译）
        start = time.time()
        result2 = test_func(np.array([1, 2, 3]))
        second_call_time = time.time() - start
        
        print(f"  首次调用时间: {first_call_time*1000:.3f} ms (包含编译时间)")
        print(f"  二次调用时间: {second_call_time*1000:.3f} ms (已编译)")
        
        if second_call_time < first_call_time * 0.5:
            print("  ✓ JIT编译正常工作")
        else:
            print("  ⚠ JIT可能未正常工作")
        
        return True
    except ImportError:
        print("✗ Numba未安装")
        print("  安装命令: pip install numba")
        return False
    except Exception as e:
        print(f"✗ 检查Numba时出错: {e}")
        return False

def check_cupy():
    """检查CuPy是否安装和激活"""
    print("\n" + "=" * 60)
    print("CuPy 状态检查")
    print("=" * 60)
    
    try:
        import cupy as cp
        print(f"✓ CuPy已安装")
        print(f"  版本: {cp.__version__}")
        
        # 检查CUDA
        try:
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            print(f"  CUDA版本: {cuda_version // 1000}.{(cuda_version % 1000) // 10}")
        except:
            print("  ⚠ 无法获取CUDA版本")
        
        # 检查GPU设备
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"  GPU设备数量: {device_count}")
            
            for i in range(device_count):
                device = cp.cuda.Device(i)
                with device:
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    mempool = cp.get_default_memory_pool()
                    meminfo = mempool.get_limit()
                    print(f"  GPU {i}: {props['name'].decode()}")
                    print(f"    内存: {meminfo / 1e9:.2f} GB")
        except Exception as e:
            print(f"  ⚠ 无法访问GPU设备: {e}")
        
        # 测试GPU计算
        try:
            # CPU计算
            start = time.time()
            a_cpu = np.random.rand(1000, 1000)
            b_cpu = np.random.rand(1000, 1000)
            result_cpu = np.dot(a_cpu, b_cpu)
            cpu_time = time.time() - start
            
            # GPU计算
            start = time.time()
            a_gpu = cp.asarray(a_cpu)
            b_gpu = cp.asarray(b_cpu)
            result_gpu = cp.dot(a_gpu, b_gpu)
            cp.cuda.Stream.null.synchronize()  # 等待GPU完成
            gpu_time = time.time() - start
            
            print(f"\n  性能测试 (1000x1000矩阵乘法):")
            print(f"    CPU时间: {cpu_time*1000:.2f} ms")
            print(f"    GPU时间: {gpu_time*1000:.2f} ms")
            print(f"    加速比: {cpu_time/gpu_time:.2f}x")
            
            if gpu_time < cpu_time:
                print("  ✓ GPU加速正常工作")
            else:
                print("  ⚠ GPU可能未正常工作或数据太小")
        
        except Exception as e:
            print(f"  ⚠ GPU测试失败: {e}")
        
        return True
    except ImportError:
        print("✗ CuPy未安装")
        print("  安装命令: pip install cupy-cuda12x  (CUDA 12.x)")
        print("            pip install cupy-cuda11x  (CUDA 11.x)")
        return False
    except Exception as e:
        print(f"✗ 检查CuPy时出错: {e}")
        return False

def check_system():
    """检查系统信息"""
    print("\n" + "=" * 60)
    print("系统信息")
    print("=" * 60)
    
    import platform
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version.split()[0]}")
    
    try:
        import psutil
        print(f"CPU核心数: {psutil.cpu_count(logical=True)}")
        print(f"物理核心数: {psutil.cpu_count(logical=False)}")
        print(f"内存: {psutil.virtual_memory().total / 1e9:.2f} GB")
    except ImportError:
        print("安装psutil可查看更详细的系统信息: pip install psutil")
    
    print(f"NumPy版本: {np.__version__}")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("性能监控工具")
    print("=" * 60)
    
    check_system()
    numba_ok = check_numba()
    cupy_ok = check_cupy()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    
    if numba_ok:
        print("✓ Numba可用 - 可以用于JIT编译加速")
    else:
        print("✗ Numba不可用 - 建议安装以获得更好性能")
    
    if cupy_ok:
        print("✓ CuPy可用 - 可以用于GPU加速")
        print("  ⚠ 注意：当前代码未实际使用GPU，需要修改代码才能利用GPU")
    else:
        print("✗ CuPy不可用 - 如需GPU加速请安装")
    
    print("\n提示：")
    print("1. CPU/GPU使用率低是正常的，因为代码主要是单线程顺序执行")
    print("2. 要真正使用GPU，需要将numpy数组转换为CuPy数组")
    print("3. 要使用Numba加速，需要使用@jit装饰器编译函数")
    print("4. 并行化深度扫描可以显著提升整体速度")

if __name__ == '__main__':
    main()

