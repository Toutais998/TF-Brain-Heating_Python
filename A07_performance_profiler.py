"""
性能分析工具
用于识别性能瓶颈和监控计算时间
"""
import time
import functools
import sys
from collections import defaultdict

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_times = {}
        self.call_counts = defaultdict(int)
        
    def timeit(self, name):
        """装饰器：测量函数执行时间"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.call_counts[name] += 1
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start
                    self.timings[name].append(elapsed)
                    return result
                except Exception as e:
                    elapsed = time.time() - start
                    self.timings[name].append(elapsed)
                    raise e
            return wrapper
        return decorator
    
    def start(self, name):
        """开始计时"""
        self.start_times[name] = time.time()
    
    def stop(self, name):
        """停止计时并记录"""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.timings[name].append(elapsed)
            del self.start_times[name]
            return elapsed
        return 0
    
    def get_stats(self, name):
        """获取统计信息"""
        if name not in self.timings or len(self.timings[name]) == 0:
            return None
        
        times = self.timings[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'calls': self.call_counts[name]
        }
    
    def print_summary(self):
        """打印性能摘要"""
        print("\n" + "="*60)
        print("性能分析摘要")
        print("="*60)
        
        # 按总时间排序
        sorted_timings = sorted(
            [(name, sum(times)) for name, times in self.timings.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        total_time = sum(times for _, times in sorted_timings)
        
        print(f"\n总计算时间: {total_time:.2f} 秒\n")
        print(f"{'函数名':<30} {'调用次数':<10} {'总时间(s)':<12} {'平均时间(s)':<12} {'占比(%)':<10}")
        print("-"*80)
        
        for name, total in sorted_timings:
            stats = self.get_stats(name)
            if stats:
                percentage = (total / total_time * 100) if total_time > 0 else 0
                print(f"{name:<30} {stats['calls']:<10} {total:<12.2f} {stats['mean']:<12.4f} {percentage:<10.1f}")
        
        print("="*60)

# 全局性能分析器实例
profiler = PerformanceProfiler()

def profile_function(name):
    """函数性能分析装饰器"""
    return profiler.timeit(name)

