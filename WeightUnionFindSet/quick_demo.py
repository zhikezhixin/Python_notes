"""
快速演示脚本 - 5分钟看到核心结果
适合快速验证和课堂演示
"""

from WeightedUnionFindSet import *
import matplotlib.pyplot as plt


def quick_demo():
    """快速演示核心功能"""
    print("\n" + "=" * 70)
    print(" " * 15 + "加权并查集性能分析 - 快速演示")
    print("=" * 70)

    # 演示1：基本使用
    print("\n【演示1】基本使用示例")
    print("-" * 70)

    data = list(range(10))
    ufs = WeightedUnionFind(data, CompressionMode.FULL)

    print("初始状态：10个独立元素")
    print(f"集合数量: {ufs.get_set_quantity()}")

    print("\n执行合并操作：")
    operations = [(0, 1), (2, 3), (1, 3), (4, 5), (6, 7), (5, 7)]
    for n1, n2 in operations:
        result = ufs.union(n1, n2)
        print(f"  union({n1}, {n2}) -> {'成功' if result else '已在同一集合'}")

    print(f"\n最终集合数量: {ufs.get_set_quantity()}")
    print(f"统计信息: {ufs.get_statistics()}")

    print("\n查询示例：")
    queries = [(0, 3), (0, 4), (4, 7)]
    for n1, n2 in queries:
        result = ufs.same_set(n1, n2)
        print(f"  {n1} 和 {n2} {'在' if result else '不在'}同一集合")

    # 演示2：性能对比（小规模）
    print("\n\n【演示2】性能快速对比 (N=3,000)")
    print("-" * 70)

    tester = PerformanceTester()
    results = tester.test_random_unions(size=3000)

    print("\n性能排名：")
    sorted_results = sorted(results.items(), key=lambda x: x[1].time_elapsed)
    for rank, (mode, result) in enumerate(sorted_results, 1):
        speedup = sorted_results[-1][1].time_elapsed / result.time_elapsed
        print(f"  {rank}. {mode.value:12s} - {result.time_elapsed:.6f}秒 "
              f"(相对最慢提速{speedup:.1f}x)")

    # 演示3：可视化
    print("\n\n【演示3】生成可视化图表")
    print("-" * 70)

    # 创建综合对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('并查集性能快速对比分析', fontsize=16, fontweight='bold')

    modes = list(results.keys())
    labels = [m.value for m in modes]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    # 子图1：执行时间
    times = [results[m].time_elapsed for m in modes]
    axes[0, 0].bar(range(len(modes)), times, color=colors, alpha=0.8)
    axes[0, 0].set_title('执行时间对比', fontweight='bold')
    axes[0, 0].set_ylabel('时间 (秒)')
    axes[0, 0].set_xticks(range(len(modes)))
    axes[0, 0].set_xticklabels(labels, rotation=15)
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, t in enumerate(times):
        axes[0, 0].text(i, t, f'{t:.4f}', ha='center', va='bottom', fontsize=9)

    # 子图2：平均路径长度
    avg_paths = [results[m].avg_path_length for m in modes]
    axes[0, 1].bar(range(len(modes)), avg_paths, color=colors, alpha=0.8)
    axes[0, 1].set_title('平均路径长度', fontweight='bold')
    axes[0, 1].set_ylabel('路径长度')
    axes[0, 1].set_xticks(range(len(modes)))
    axes[0, 1].set_xticklabels(labels, rotation=15)
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, p in enumerate(avg_paths):
        axes[0, 1].text(i, p, f'{p:.2f}', ha='center', va='bottom', fontsize=9)

    # 子图3：最大路径长度
    max_paths = [results[m].max_path_length for m in modes]
    axes[1, 0].bar(range(len(modes)), max_paths, color=colors, alpha=0.8)
    axes[1, 0].set_title('最大路径长度', fontweight='bold')
    axes[1, 0].set_ylabel('路径长度')
    axes[1, 0].set_xticks(range(len(modes)))
    axes[1, 0].set_xticklabels(labels, rotation=15)
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, m in enumerate(max_paths):
        axes[1, 0].text(i, m, f'{m}', ha='center', va='bottom', fontsize=9)

    # 子图4：性能提升倍数（相对无压缩）
    baseline = results[CompressionMode.NONE].time_elapsed
    speedups = [baseline / results[m].time_elapsed for m in modes]
    bars = axes[1, 1].bar(range(len(modes)), speedups, color=colors, alpha=0.8)
    axes[1, 1].set_title('性能提升倍数 (相对无压缩)', fontweight='bold')
    axes[1, 1].set_ylabel('提升倍数')
    axes[1, 1].set_xticks(range(len(modes)))
    axes[1, 1].set_xticklabels(labels, rotation=15)
    axes[1, 1].axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, s in enumerate(speedups):
        axes[1, 1].text(i, s, f'{s:.1f}x', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('quick_demo_results.png', dpi=300, bbox_inches='tight')
    print("✅ 图表已保存: quick_demo_results.png")
    plt.show()

    # 演示4：生成简报
    print("\n\n【演示4】性能简报")
    print("-" * 70)

    print("\n核心发现：")
    best_mode = sorted_results[0][0]
    worst_mode = sorted_results[-1][0]
    best_time = sorted_results[0][1].time_elapsed
    worst_time = sorted_results[-1][1].time_elapsed
    speedup = worst_time / best_time

    print(f"  ✓ 最优策略：{best_mode.value} ({best_time:.6f}秒)")
    print(f"  ✓ 最差策略：{worst_mode.value} ({worst_time:.6f}秒)")
    print(f"  ✓ 性能提升：{speedup:.1f} 倍")
    print(f"  ✓ 平均路径：{results[best_mode].avg_path_length:.2f} "
          f"→ {results[worst_mode].avg_path_length:.2f}")

    print("\n推荐建议：")
    print(f"  • 高性能场景：使用 {CompressionMode.FULL.value}")
    print(f"  • 平衡场景：使用 {CompressionMode.HALVE.value} 或 {CompressionMode.SPLIT.value}")
    print(f"  • 学习用途：先理解 {CompressionMode.NONE.value}，再学习优化策略")

    # 生成简易报告
    print("\n\n【演示5】生成快速报告")
    print("-" * 70)

    report_data = []
    for mode in modes:
        result = results[mode]
        report_data.append({
            '压缩策略': mode.value,
            '执行时间': f"{result.time_elapsed:.6f}s",
            '平均路径': f"{result.avg_path_length:.2f}",
            '最大路径': result.max_path_length,
            '相对提升': f"{speedups[modes.index(mode)]:.2f}x"
        })

    df = pd.DataFrame(report_data)
    print("\n" + df.to_string(index=False))

    df.to_csv('quick_demo_report.csv', index=False, encoding='utf-8-sig')
    print("\n✅ 报告已保存: quick_demo_report.csv")

    print("\n" + "=" * 70)
    print(" " * 20 + "快速演示完成！")
    print("=" * 70)
    print("\n生成的文件：")
    print("  📊 quick_demo_results.png  - 综合对比图")
    print("  📄 quick_demo_report.csv   - 性能报告")
    print("\n💡 提示：运行 advanced_tests.py 获取完整的学报级分析")

    return results


def interactive_comparison():
    """交互式对比两种策略"""
    print("\n" + "=" * 70)
    print(" " * 20 + "交互式策略对比")
    print("=" * 70)

    print("\n可用的压缩策略：")
    modes_list = list(CompressionMode)
    for i, mode in enumerate(modes_list):
        print(f"  {i + 1}. {mode.value}")

    print("\n请选择两个策略进行对比（输入数字，用空格分隔，如：1 2）")
    print("或直接按回车使用默认对比（无压缩 vs 完全压缩）")

    try:
        choice = input("您的选择: ").strip()
        if choice:
            indices = [int(x) - 1 for x in choice.split()]
            mode1, mode2 = modes_list[indices[0]], modes_list[indices[1]]
        else:
            mode1, mode2 = CompressionMode.NONE, CompressionMode.FULL
    except:
        mode1, mode2 = CompressionMode.NONE, CompressionMode.FULL
        print("使用默认对比")

    print(f"\n正在对比：{mode1.value} vs {mode2.value}")
    print("-" * 70)

    size = 5000
    data = list(range(size))

    # 测试两种策略
    results_compare = {}
    for mode in [mode1, mode2]:
        print(f"\n测试 {mode.value}...")
        ufs = WeightedUnionFind(data.copy(), mode)

        start = time.perf_counter()

        # 执行操作
        for i in range(size // 2):
            ufs.union(i, i + size // 2)
        for _ in range(size):
            ufs.same_set(0, np.random.randint(0, size))

        elapsed = time.perf_counter() - start
        stats = ufs.get_statistics()

        results_compare[mode] = {
            'time': elapsed,
            'stats': stats
        }

        print(f"  完成！耗时: {elapsed:.6f}秒")

    # 对比结果
    print("\n" + "=" * 70)
    print("对比结果：")
    print("=" * 70)

    print(f"\n{'指标':<20} {mode1.value:<15} {mode2.value:<15} {'差异':<15}")
    print("-" * 70)

    time1 = results_compare[mode1]['time']
    time2 = results_compare[mode2]['time']
    stats1 = results_compare[mode1]['stats']
    stats2 = results_compare[mode2]['stats']

    print(f"{'执行时间':<20} {time1:.6f}s{'':<6} {time2:.6f}s{'':<6} "
          f"{abs(time1 - time2) / min(time1, time2) * 100:.1f}%")

    print(f"{'平均路径长度':<20} {stats1['avg_path_length']:.2f}{'':<13} "
          f"{stats2['avg_path_length']:.2f}{'':<13} "
          f"{abs(stats1['avg_path_length'] - stats2['avg_path_length']):.2f}")

    print(f"{'最大路径长度':<20} {stats1['max_path_length']:<15} "
          f"{stats2['max_path_length']:<15} "
          f"{abs(stats1['max_path_length'] - stats2['max_path_length'])}")

    print(f"{'Find操作次数':<20} {stats1['find_count']:<15} "
          f"{stats2['find_count']:<15} {stats1['find_count'] - stats2['find_count']}")

    faster = mode1 if time1 < time2 else mode2
    speedup = max(time1, time2) / min(time1, time2)

    print(f"\n🏆 结论：{faster.value} 快 {speedup:.2f} 倍")

    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 时间对比
    axes[0].bar([mode1.value, mode2.value], [time1, time2],
                color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    axes[0].set_title('执行时间对比', fontweight='bold', fontsize=13)
    axes[0].set_ylabel('时间 (秒)')
    axes[0].grid(axis='y', alpha=0.3)
    for i, (t, m) in enumerate([(time1, mode1), (time2, mode2)]):
        axes[0].text(i, t, f'{t:.6f}s', ha='center', va='bottom', fontsize=10)

    # 路径长度对比
    metrics = ['平均路径', '最大路径']
    x = np.arange(len(metrics))
    width = 0.35

    vals1 = [stats1['avg_path_length'], stats1['max_path_length']]
    vals2 = [stats2['avg_path_length'], stats2['max_path_length']]

    axes[1].bar(x - width / 2, vals1, width, label=mode1.value,
                color='#FF6B6B', alpha=0.8)
    axes[1].bar(x + width / 2, vals2, width, label=mode2.value,
                color='#4ECDC4', alpha=0.8)
    axes[1].set_title('路径长度对比', fontweight='bold', fontsize=13)
    axes[1].set_ylabel('长度')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('interactive_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ 对比图已保存: interactive_comparison.png")
    plt.show()


def stress_test():
    """压力测试 - 大规模数据"""
    print("\n" + "=" * 70)
    print(" " * 25 + "压力测试")
    print("=" * 70)

    print("\n⚠️  警告：压力测试将使用大规模数据，可能需要较长时间")
    print("推荐配置：16GB+ 内存")

    sizes = [10000, 50000, 100000]

    print(f"\n将测试以下规模：{', '.join([f'{s:,}' for s in sizes])}")
    print("\n是否继续？(y/n，默认n): ", end='')

    try:
        confirm = input().strip().lower()
    except:
        confirm = 'n'

    if confirm != 'y':
        print("已取消压力测试")
        return

    print("\n开始压力测试...")
    print("-" * 70)

    results_stress = {mode: [] for mode in CompressionMode}

    for size in sizes:
        print(f"\n测试规模: {size:,}")
        data = list(range(size))

        for mode in CompressionMode:
            print(f"  测试 {mode.value}...", end=' ')
            ufs = WeightedUnionFind(data.copy(), mode)

            start = time.perf_counter()

            # 大量操作
            num_unions = int(size * 0.8)
            num_queries = int(size * 0.5)

            for _ in range(num_unions):
                n1, n2 = np.random.randint(0, size, 2)
                ufs.union(n1, n2)

            for _ in range(num_queries):
                n1, n2 = np.random.randint(0, size, 2)
                ufs.same_set(n1, n2)

            elapsed = time.perf_counter() - start
            results_stress[mode].append(elapsed)

            print(f"{elapsed:.4f}秒")

    # 绘制压力测试结果
    plt.figure(figsize=(12, 7))

    colors = {'无压缩': '#FF6B6B', '完全压缩': '#4ECDC4',
              '折半压缩': '#45B7D1', '分裂压缩': '#FFA07A'}

    for mode in CompressionMode:
        plt.plot(sizes, results_stress[mode], 'o-',
                 label=mode.value, linewidth=2.5, markersize=10,
                 color=colors.get(mode.value, '#666'))

    plt.xlabel('数据规模', fontsize=13, fontweight='bold')
    plt.ylabel('执行时间 (秒)', fontsize=13, fontweight='bold')
    plt.title('压力测试 - 大规模性能表现', fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('stress_test_results.png', dpi=300, bbox_inches='tight')
    print("\n✅ 压力测试结果已保存: stress_test_results.png")
    plt.show()

    print("\n" + "=" * 70)
    print("压力测试完成！")
    print("=" * 70)


def main_menu():
    """主菜单"""
    while True:
        print("\n" + "=" * 70)
        print(" " * 15 + "并查集性能分析系统 - 主菜单")
        print("=" * 70)
        print("\n请选择功能：")
        print("  1. 快速演示 (5分钟，推荐首次使用)")
        print("  2. 交互式策略对比")
        print("  3. 压力测试 (大规模数据)")
        print("  4. 完整性能测试 (生成学报所需所有数据)")
        print("  5. 进阶分析 (详细的学术分析)")
        print("  0. 退出")
        print("-" * 70)

        try:
            choice = input("请输入选项编号: ").strip()
        except:
            choice = '0'

        if choice == '1':
            quick_demo()
        elif choice == '2':
            interactive_comparison()
        elif choice == '3':
            stress_test()
        elif choice == '4':
            print("\n启动完整性能测试...")
            from WeightedUnionFindSet import run_comprehensive_test
            run_comprehensive_test()
        elif choice == '5':
            print("\n启动进阶分析...")
            from advanced_tests import generate_complete_documentation
            generate_complete_documentation()
        elif choice == '0':
            print("\n感谢使用！祝您的学报撰写顺利！🎓")
            break
        else:
            print("\n❌ 无效选项，请重新选择")

        input("\n按回车键继续...")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║          基于多路径压缩策略的加权并查集性能分析系统              ║
    ║                                                                  ║
    ║              Weighted Union-Find Performance Analyzer            ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝

    功能特点：
      ✓ 四种路径压缩策略完整实现
      ✓ 全面的性能测试和对比
      ✓ 精美的可视化图表
      ✓ 学报级别的数据分析
      ✓ 交互式功能选择

    """)

    print("选择运行模式：")
    print("  1. 交互式菜单（推荐）")
    print("  2. 直接运行快速演示")
    print("  3. 直接运行完整测试")

    try:
        mode = input("\n请选择 (默认1): ").strip() or '1'
    except:
        mode = '1'

    if mode == '1':
        main_menu()
    elif mode == '2':
        quick_demo()
    elif mode == '3':
        from WeightedUnionFindSet import run_comprehensive_test

        run_comprehensive_test()
    else:
        print("无效选择，启动快速演示...")
        quick_demo()