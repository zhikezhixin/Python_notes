"""
基于多路径压缩策略的加权并查集优化实现与性能分析
本程序实现了四种路径压缩策略的加权并查集，并提供完整的性能测试和可视化分析
"""

from enum import Enum
from typing import TypeVar, Generic, List, Dict, Optional, Set, Tuple
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

NodeType = TypeVar('NodeType')


class CompressionMode(Enum):
    """路径压缩模式枚举"""
    NONE = "无压缩"
    FULL = "完全压缩"
    HALVE = "折半压缩"
    SPLIT = "分裂压缩"


class WeightedUnionFind(Generic[NodeType]):
    """加权并查集实现 - 支持四种路径压缩策略"""

    def __init__(self, data: Optional[List[NodeType]] = None,
                 mode: CompressionMode = CompressionMode.FULL):
        """
        初始化并查集

        Args:
            data: 初始数据列表
            mode: 路径压缩模式
        """
        self.mode = mode
        self._index_map: Dict[NodeType, int] = {}
        self._parent: List[int] = []
        self._set_size: List[int] = []
        self._set_quantity: int = 0

        # 性能统计
        self.find_count = 0
        self.union_count = 0
        self.path_length_sum = 0
        self.max_path_length = 0

        if data:
            self._initialize_with_data(data)

    def _initialize_with_data(self, data: List[NodeType]):
        """使用数据初始化并查集"""
        n = len(data)
        self._parent = list(range(n))
        self._set_size = [1] * n
        self._set_quantity = n

        for i, node in enumerate(data):
            if node in self._index_map:
                raise ValueError(f"重复元素: {node}")
            self._index_map[node] = i

    def get_index(self, node: NodeType) -> int:
        """获取节点索引，不存在则创建"""
        if node not in self._index_map:
            index = len(self._parent)
            self._index_map[node] = index
            self._parent.append(index)
            self._set_size.append(1)
            self._set_quantity += 1
        return self._index_map[node]

    def try_get_index(self, node: NodeType) -> Optional[int]:
        """尝试获取节点索引，不存在返回None"""
        return self._index_map.get(node)

    def _find_none(self, x: int) -> int:
        """无压缩查找"""
        path_length = 0
        while self._parent[x] != x:
            x = self._parent[x]
            path_length += 1

        self.path_length_sum += path_length
        self.max_path_length = max(self.max_path_length, path_length)
        return x

    def _find_full(self, x: int) -> int:
        """完全路径压缩"""
        path_length = 0
        cur = x

        # 找根
        while self._parent[cur] != cur:
            cur = self._parent[cur]
            path_length += 1

        root = cur

        # 回溯压缩
        while self._parent[x] != x:
            parent = self._parent[x]
            self._parent[x] = root
            x = parent

        self.path_length_sum += path_length
        self.max_path_length = max(self.max_path_length, path_length)
        return root

    def _find_halve(self, x: int) -> int:
        """折半路径压缩"""
        path_length = 0
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
            path_length += 1

        self.path_length_sum += path_length
        self.max_path_length = max(self.max_path_length, path_length)
        return x

    def _find_split(self, x: int) -> int:
        """分裂路径压缩"""
        path_length = 0
        while self._parent[x] != x:
            parent = self._parent[x]
            grandparent = self._parent[parent]
            self._parent[x] = grandparent
            x = parent
            path_length += 1

        self.path_length_sum += path_length
        self.max_path_length = max(self.max_path_length, path_length)
        return x

    def find_root(self, x: int) -> int:
        """根据模式查找根节点"""
        assert x < len(self._parent), f"索引越界: {x}"
        self.find_count += 1

        if self.mode == CompressionMode.NONE:
            return self._find_none(x)
        elif self.mode == CompressionMode.FULL:
            return self._find_full(x)
        elif self.mode == CompressionMode.HALVE:
            return self._find_halve(x)
        elif self.mode == CompressionMode.SPLIT:
            return self._find_split(x)

    def union(self, n1: NodeType, n2: NodeType) -> bool:
        """
        合并两个节点所在的集合

        Returns:
            bool: 是否成功合并（原本不在同一集合）
        """
        if n1 == n2:
            return False

        self.union_count += 1
        index_n1 = self.get_index(n1)
        index_n2 = self.get_index(n2)
        root_n1 = self.find_root(index_n1)
        root_n2 = self.find_root(index_n2)

        if root_n1 == root_n2:
            return False

        # 按集合大小合并 - 小集合合并到大集合
        if self._set_size[root_n1] < self._set_size[root_n2]:
            self._set_size[root_n2] += self._set_size[root_n1]
            self._parent[root_n1] = root_n2
        else:
            self._set_size[root_n1] += self._set_size[root_n2]
            self._parent[root_n2] = root_n1

        self._set_quantity -= 1
        return True

    def same_set(self, n1: NodeType, n2: NodeType) -> bool:
        """判断两个节点是否在同一集合"""
        idx1 = self.try_get_index(n1)
        idx2 = self.try_get_index(n2)

        if idx1 is None or idx2 is None:
            return False

        return self.find_root(idx1) == self.find_root(idx2)

    def get_set_quantity(self) -> int:
        """获取集合数量"""
        return self._set_quantity

    def is_exist(self, node: NodeType) -> bool:
        """判断节点是否存在"""
        return node in self._index_map

    def get_statistics(self) -> Dict:
        """获取性能统计信息"""
        avg_path_length = (self.path_length_sum / self.find_count
                           if self.find_count > 0 else 0)
        return {
            'find_count': self.find_count,
            'union_count': self.union_count,
            'avg_path_length': avg_path_length,
            'max_path_length': self.max_path_length,
            'set_quantity': self._set_quantity
        }

    def get_tree_heights(self) -> List[int]:
        """获取所有树的高度分布"""
        heights = []
        visited = set()

        for i in range(len(self._parent)):
            if i in visited:
                continue
            if self._parent[i] == i:  # 找到根节点
                height = self._calculate_tree_height(i, visited)
                heights.append(height)

        return heights

    def _calculate_tree_height(self, root: int, visited: Set[int]) -> int:
        """计算以root为根的树的高度"""
        max_height = 0
        stack = [(root, 0)]
        visited.add(root)

        while stack:
            node, depth = stack.pop()
            max_height = max(max_height, depth)

            # 找所有子节点
            for i in range(len(self._parent)):
                if i not in visited and self._parent[i] == node:
                    stack.append((i, depth + 1))
                    visited.add(i)

        return max_height


@dataclass
class TestResult:
    """测试结果数据类"""
    mode: CompressionMode
    data_size: int
    time_elapsed: float
    find_count: int
    union_count: int
    avg_path_length: float
    max_path_length: int
    final_set_count: int
    tree_heights: List[int]


class PerformanceTester:
    """性能测试类"""

    def __init__(self):
        self.results: List[TestResult] = []

    def test_random_unions(self, size: int, union_ratio: float = 0.8) -> Dict[CompressionMode, TestResult]:
        """
        测试随机合并操作

        Args:
            size: 数据规模
            union_ratio: 合并操作占比
        """
        print(f"\n{'=' * 60}")
        print(f"测试场景: 随机合并 | 数据规模: {size:,}")
        print(f"{'=' * 60}")

        results = {}
        data = list(range(size))
        num_operations = int(size * 2)
        num_unions = int(num_operations * union_ratio)

        for mode in CompressionMode:
            print(f"\n测试模式: {mode.value}")

            # 创建并查集
            ufs = WeightedUnionFind(data.copy(), mode)

            # 生成操作序列
            union_ops = [(random.randint(0, size - 1), random.randint(0, size - 1))
                         for _ in range(num_unions)]
            query_ops = [(random.randint(0, size - 1), random.randint(0, size - 1))
                         for _ in range(num_operations - num_unions)]

            # 执行测试
            start_time = time.perf_counter()

            for n1, n2 in union_ops:
                ufs.union(n1, n2)

            for n1, n2 in query_ops:
                ufs.same_set(n1, n2)

            elapsed = time.perf_counter() - start_time

            # 收集结果
            stats = ufs.get_statistics()
            tree_heights = ufs.get_tree_heights()

            result = TestResult(
                mode=mode,
                data_size=size,
                time_elapsed=elapsed,
                find_count=stats['find_count'],
                union_count=stats['union_count'],
                avg_path_length=stats['avg_path_length'],
                max_path_length=stats['max_path_length'],
                final_set_count=stats['set_quantity'],
                tree_heights=tree_heights
            )

            results[mode] = result
            self.results.append(result)

            print(f"  耗时: {elapsed:.6f}秒")
            print(f"  平均路径长度: {stats['avg_path_length']:.2f}")
            print(f"  最大路径长度: {stats['max_path_length']}")
            print(f"  最终集合数: {stats['set_quantity']}")

        return results

    def test_worst_case(self, size: int) -> Dict[CompressionMode, TestResult]:
        """
        测试最坏情况（链式结构）

        Args:
            size: 数据规模
        """
        print(f"\n{'=' * 60}")
        print(f"测试场景: 最坏情况(链式合并) | 数据规模: {size:,}")
        print(f"{'=' * 60}")

        results = {}

        for mode in CompressionMode:
            print(f"\n测试模式: {mode.value}")

            data = list(range(size))
            ufs = WeightedUnionFind(data.copy(), mode)

            start_time = time.perf_counter()

            # 构造链式结构: 0-1, 1-2, 2-3, ...
            for i in range(size - 1):
                ufs.union(i, i + 1)

            # 执行查询操作
            for _ in range(size):
                ufs.same_set(0, random.randint(0, size - 1))

            elapsed = time.perf_counter() - start_time

            stats = ufs.get_statistics()
            tree_heights = ufs.get_tree_heights()

            result = TestResult(
                mode=mode,
                data_size=size,
                time_elapsed=elapsed,
                find_count=stats['find_count'],
                union_count=stats['union_count'],
                avg_path_length=stats['avg_path_length'],
                max_path_length=stats['max_path_length'],
                final_set_count=stats['set_quantity'],
                tree_heights=tree_heights
            )

            results[mode] = result
            self.results.append(result)

            print(f"  耗时: {elapsed:.6f}秒")
            print(f"  平均路径长度: {stats['avg_path_length']:.2f}")
            print(f"  最大路径长度: {stats['max_path_length']}")

        return results

    def test_scale_performance(self, sizes: List[int]):
        """测试不同规模下的性能表现"""
        print(f"\n{'=' * 60}")
        print("测试场景: 规模扩展性能分析")
        print(f"{'=' * 60}")

        for size in sizes:
            self.test_random_unions(size)

    def generate_report(self) -> pd.DataFrame:
        """生成性能报告表格"""
        data = []
        for result in self.results:
            avg_height = np.mean(result.tree_heights) if result.tree_heights else 0
            max_height = max(result.tree_heights) if result.tree_heights else 0

            data.append({
                '压缩模式': result.mode.value,
                '数据规模': result.data_size,
                '执行时间(秒)': f"{result.time_elapsed:.6f}",
                'Find操作次数': result.find_count,
                'Union操作次数': result.union_count,
                '平均路径长度': f"{result.avg_path_length:.2f}",
                '最大路径长度': result.max_path_length,
                '平均树高': f"{avg_height:.2f}",
                '最大树高': max_height,
                '最终集合数': result.final_set_count
            })

        return pd.DataFrame(data)


class Visualizer:
    """可视化工具类"""

    @staticmethod
    def plot_time_comparison(results: Dict[CompressionMode, TestResult],
                             title: str = "执行时间对比"):
        """绘制执行时间对比图"""
        modes = list(results.keys())
        times = [results[mode].time_elapsed for mode in modes]
        labels = [mode.value for mode in modes]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(modes)), times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        plt.xlabel('路径压缩模式', fontsize=12)
        plt.ylabel('执行时间 (秒)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(range(len(modes)), labels, rotation=15)
        plt.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for i, (bar, time_val) in enumerate(zip(bars, times)):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.01,
                     f'{time_val:.6f}s', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_path_length_comparison(results: Dict[CompressionMode, TestResult]):
        """绘制路径长度对比图"""
        modes = list(results.keys())
        avg_lengths = [results[mode].avg_path_length for mode in modes]
        max_lengths = [results[mode].max_path_length for mode in modes]
        labels = [mode.value for mode in modes]

        x = np.arange(len(modes))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width / 2, avg_lengths, width, label='平均路径长度', color='#4ECDC4')
        bars2 = ax.bar(x + width / 2, max_lengths, width, label='最大路径长度', color='#FF6B6B')

        ax.set_xlabel('路径压缩模式', fontsize=12)
        ax.set_ylabel('路径长度', fontsize=12)
        ax.set_title('路径长度对比分析', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_scale_performance(tester: PerformanceTester, scenario: str = "random"):
        """绘制规模性能曲线"""
        # 按规模和模式分组
        data_by_mode = defaultdict(lambda: {'sizes': [], 'times': []})

        for result in tester.results:
            data_by_mode[result.mode]['sizes'].append(result.data_size)
            data_by_mode[result.mode]['times'].append(result.time_elapsed)

        plt.figure(figsize=(12, 7))
        colors = {'无压缩': '#FF6B6B', '完全压缩': '#4ECDC4',
                  '折半压缩': '#45B7D1', '分裂压缩': '#FFA07A'}

        for mode, data in data_by_mode.items():
            if data['sizes']:
                # 排序数据点
                sorted_pairs = sorted(zip(data['sizes'], data['times']))
                sizes, times = zip(*sorted_pairs)

                plt.plot(sizes, times, 'o-', label=mode.value,
                         linewidth=2, markersize=8, color=colors.get(mode.value, '#666'))

        plt.xlabel('数据规模', fontsize=12)
        plt.ylabel('执行时间 (秒)', fontsize=12)
        plt.title('不同规模下的性能表现', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_tree_height_distribution(results: Dict[CompressionMode, TestResult]):
        """绘制树高度分布"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

        for idx, (mode, result) in enumerate(results.items()):
            if result.tree_heights:
                axes[idx].hist(result.tree_heights, bins=30, color=colors[idx],
                               alpha=0.7, edgecolor='black')
                axes[idx].set_xlabel('树高度', fontsize=11)
                axes[idx].set_ylabel('频数', fontsize=11)
                axes[idx].set_title(f'{mode.value} - 树高度分布', fontsize=12, fontweight='bold')
                axes[idx].grid(axis='y', alpha=0.3)

                # 添加统计信息
                avg_height = np.mean(result.tree_heights)
                max_height = max(result.tree_heights)
                axes[idx].axvline(avg_height, color='red', linestyle='--',
                                  linewidth=2, label=f'平均: {avg_height:.2f}')
                axes[idx].legend(fontsize=10)

        plt.tight_layout()
        plt.show()


def run_comprehensive_test():
    """运行综合测试"""
    print("=" * 60)
    print("基于多路径压缩策略的加权并查集性能分析系统")
    print("=" * 60)

    tester = PerformanceTester()

    # 1. 不同规模的随机合并测试
    print("\n【测试一】不同规模下的随机合并性能")
    sizes = [1000, 5000, 10000]
    for size in sizes:
        results = tester.test_random_unions(size)
        Visualizer.plot_time_comparison(results, f"随机合并测试 (N={size:,})")
        Visualizer.plot_path_length_comparison(results)

    # 2. 最坏情况测试
    print("\n【测试二】最坏情况(链式结构)性能对比")
    worst_results = tester.test_worst_case(5000)
    Visualizer.plot_time_comparison(worst_results, "最坏情况测试 (链式合并, N=5000)")
    Visualizer.plot_path_length_comparison(worst_results)
    Visualizer.plot_tree_height_distribution(worst_results)

    # 3. 规模扩展性测试
    print("\n【测试三】规模扩展性能分析")
    scale_sizes = [500, 1000, 2000, 5000, 10000]
    for size in scale_sizes:
        tester.test_random_unions(size)

    Visualizer.plot_scale_performance(tester)

    # 4. 生成报告
    print("\n【性能测试报告】")
    report = tester.generate_report()
    print(report.to_string(index=False))

    # 保存报告
    report.to_csv('union_find_performance_report.csv', index=False, encoding='utf-8-sig')
    print("\n报告已保存至: union_find_performance_report.csv")

    return tester, report


if __name__ == "__main__":
    # 运行完整测试
    tester, report = run_comprehensive_test()

    print("\n" + "=" * 60)
    print("测试完成！所有图表和数据已生成")
    print("=" * 60)