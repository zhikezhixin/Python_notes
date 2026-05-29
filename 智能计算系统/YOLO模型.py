# -*- coding: utf-8 -*-
"""
YOLOv5 延时性能汇总表（防重叠版）
依赖：matplotlib、pandas
pip install matplotlib pandas
"""

import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ===== 1) 中文字体配置（优先微软雅黑/黑体）=====
def set_chinese_font():
    preferred = ["Microsoft YaHei", "SimHei", "Source Han Sans CN", "Noto Sans CJK SC"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for fam in preferred:
        if fam in available:
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams["axes.unicode_minus"] = False

set_chinese_font()

# ===== 2) 原始数据 =====
data = {
    "阶段": ["训练阶段", "验证阶段", "推理阶段（检测）", "整体平均（报告值）"],
    "Batch大小": ["16", "32", "1", "——"],
    "图像尺寸": ["640×640", "512×512", "512×512", "——"],
    "设备": ["RTX 4060 Laptop GPU"] * 4,
    "平均延时（ms/图像）": ["——", "≈ 12.5", "≈ 53.0", "12.5～53.0"],
    "详细组成": [
        "——",
        "pre-process：0.6  |  inference：9.9  |  NMS：2.0",
        "pre-process：0.9  |  inference：24.5  |  NMS：27.6",
        "——"
    ],
    "说明": [
        "训练阶段主要输出 Loss（box_loss、obj_loss、cls_loss），不计算 Speed。",
        "平均每张图 12.5ms，包含完整推理过程（预处理 + 前向计算 + 后处理）。",
        "单张图片检测总耗时约 53ms，性能优异，可达约 19 FPS。",
        "所有阶段平均延时远低于实验标准的 100 ms，达到“优秀档”要求。"
    ]
}
df = pd.DataFrame(data)

# ===== 3) 列宽对应的换行宽度（可按需要调整）=====
# 每列对应的最大字符数，超过自动换行
wrap_widths = {
    "阶段": 8,
    "Batch大小": 6,
    "图像尺寸": 8,
    "设备": 18,
    "平均延时（ms/图像）": 12,
    "详细组成": 36,     # 文本较长，给更大的换行宽度
    "说明": 42          # 文本较长，给更大的换行宽度
}

def wrap_col_text(series: pd.Series, width: int) -> pd.Series:
    return series.apply(lambda s: "\n".join(textwrap.fill(str(s), width=width).splitlines()))

for col, w in wrap_widths.items():
    df[col] = wrap_col_text(df[col], w)

# ===== 4) 绘制表格（放大画布 + 指定列宽 + 较小字体）=====
def draw_table(df: pd.DataFrame,
               title: str = "YOLOv5 模型延时性能汇总表",
               png_path: str = "YOLOv5_延时性能汇总表.png",
               pdf_path: str = "YOLOv5_延时性能汇总表.pdf"):
    # 画布更大，适配长文本
    FIG_W, FIG_H = 17, 7   # 单位：英寸
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.axis('off')

    # 列宽：每列宽度占轴宽的比例，总和 ~ 1.0
    # 7 列：适当多给“详细组成/说明”
    col_widths = [0.08, 0.09, 0.09, 0.16, 0.12, 0.22, 0.24]

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     colWidths=col_widths,
                     loc='center')

    # 字体与缩放（字体小点 + 拉高行高）
    table.auto_set_font_size(False)
    FONT_SIZE = 10
    table.set_fontsize(FONT_SIZE)
    table.scale(1.15, 1.8)  # (宽缩放, 高缩放)

    # 表头样式 + 斑马纹
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#DCE6F1")
            cell.set_text_props(weight='bold', color='black')
        elif r % 2 == 0:
            cell.set_facecolor("#F7F9FC")

        # 左对齐“说明/详细组成”列更易读
        if r >= 1 and df.columns[c] in ("详细组成", "说明"):
            cell._loc = 'left'  # 左对齐
            # 给左对齐的单元格加一点内边距
            cell.PAD = 0.02

    # 标题
    plt.title(title, fontsize=18, fontweight='bold', pad=18)

    # 导出高分辨率图片/PDF
    plt.tight_layout()
    plt.savefig(png_path, dpi=350, bbox_inches='tight')  # 提高到 350dpi
    plt.savefig(pdf_path, dpi=350, bbox_inches='tight')
    plt.close()

    print(f"[OK] 已生成：{os.path.abspath(png_path)}")
    print(f"[OK] 已生成：{os.path.abspath(pdf_path)}")

if __name__ == "__main__":
    draw_table(df)
