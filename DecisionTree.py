import pandas as pd
import numpy as np
from collections import Counter
import math
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from io import BytesIO
import base64
from PIL import Image, ImageTk


class WatermelonDecisionTree:
    def __init__(self):
        self.tree = None
        self.feature_names = None

    def load_data(self):
        """加载西瓜数据集"""
        data = {
            '编号': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            '色泽': ['青绿', '乌黑', '乌黑', '青绿', '浅白', '青绿', '乌黑', '乌黑', '乌黑', '青绿', '浅白', '浅白',
                     '青绿', '浅白', '乌黑', '浅白', '青绿'],
            '根蒂': ['蜷缩', '蜷缩', '蜷缩', '蜷缩', '蜷缩', '稍蜷', '稍蜷', '稍蜷', '稍蜷', '硬挺', '硬挺', '蜷缩',
                     '稍蜷', '稍蜷', '稍蜷', '蜷缩', '硬挺'],
            '敲声': ['浊响', '沉闷', '浊响', '沉闷', '浊响', '浊响', '浊响', '浊响', '沉闷', '清脆', '清脆', '浊响',
                     '浊响', '沉闷', '浊响', '浊响', '沉闷'],
            '纹理': ['清晰', '清晰', '清晰', '清晰', '清晰', '清晰', '稍糊', '清晰', '稍糊', '清晰', '模糊', '模糊',
                     '稍糊', '稍糊', '清晰', '模糊', '清晰'],
            '脐部': ['凹陷', '凹陷', '凹陷', '凹陷', '凹陷', '凹陷', '凹陷', '凹陷', '平坦', '平坦', '平坦', '平坦',
                     '凹陷', '凹陷', '稍凹', '稍凹', '稍凹'],
            '触感': ['硬滑', '硬滑', '硬滑', '硬滑', '硬滑', '软粘', '软粘', '硬滑', '硬滑', '软粘', '硬滑', '硬滑',
                     '硬滑', '硬滑', '软粘', '硬滑', '硬滑'],
            '好瓜': ['是', '是', '是', '是', '是', '是', '是', '是', '否', '否', '否', '否', '否', '否', '否', '否',
                     '否']
        }
        return pd.DataFrame(data)

    def calculate_entropy(self, labels):
        """计算信息熵"""
        if len(labels) == 0:
            return 0
        label_counts = Counter(labels)
        entropy = 0
        total = len(labels)
        for count in label_counts.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * math.log2(probability)
        return entropy

    def calculate_information_gain(self, data, feature, target):
        """计算信息增益"""
        total_entropy = self.calculate_entropy(data[target])
        feature_values = data[feature].unique()
        weighted_entropy = 0
        total_samples = len(data)
        for value in feature_values:
            subset = data[data[feature] == value]
            weight = len(subset) / total_samples
            weighted_entropy += weight * self.calculate_entropy(subset[target])
        return total_entropy - weighted_entropy

    def select_best_feature(self, data, features, target):
        """选择最佳分裂特征"""
        best_feature = None
        best_gain = -1
        for feature in features:
            gain = self.calculate_information_gain(data, feature, target)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        return best_feature, best_gain

    def build_tree(self, data, features, target, depth=0, max_depth=5):
        """构建决策树"""
        unique_labels = data[target].unique()
        if len(unique_labels) == 1:
            return {'type': 'leaf', 'label': unique_labels[0], 'samples': len(data)}
        if len(features) == 0 or depth >= max_depth:
            majority_label = data[target].mode()[0]
            return {'type': 'leaf', 'label': majority_label, 'samples': len(data)}
        best_feature, best_gain = self.select_best_feature(data, features, target)
        if best_gain <= 0:
            majority_label = data[target].mode()[0]
            return {'type': 'leaf', 'label': majority_label, 'samples': len(data)}
        node = {'type': 'internal', 'feature': best_feature, 'gain': best_gain, 'children': {}, 'samples': len(data)}
        remaining_features = [f for f in features if f != best_feature]
        feature_values = data[best_feature].unique()
        for value in feature_values:
            subset = data[data[best_feature] == value]
            if len(subset) > 0:
                node['children'][value] = self.build_tree(subset, remaining_features, target, depth + 1, max_depth)
        return node

    def fit(self):
        """训练决策树"""
        data = self.load_data()
        features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
        target = '好瓜'
        self.feature_names = features
        self.tree = self.build_tree(data, features, target)
        predictions = []
        for _, row in data.iterrows():
            pred = self.predict_single(row)
            predictions.append(pred)
        accuracy = sum(1 for i, pred in enumerate(predictions) if pred == data.iloc[i][target]) / len(data)
        return accuracy, data

    def predict_single(self, sample):
        """预测单个样本"""
        if self.tree is None:
            raise ValueError("模型尚未训练，请先调用fit()方法")
        node = self.tree
        while node['type'] == 'internal':
            feature = node['feature']
            feature_value = sample[feature]
            if feature_value in node['children']:
                node = node['children'][feature_value]
            else:
                return '是'
        return node['label']

    def get_tree_structure(self, node=None, prefix='', is_last=True):
        """获取决策树结构字符串"""
        if node is None:
            node = self.tree
        if node is None:
            return ["树尚未构建"]
        result = []
        prefix_str = prefix + ('└── ' if is_last else '├── ')
        if node['type'] == 'leaf':
            result.append(f"{prefix_str}结果: {node['label']} (样本数: {node['samples']})")
        else:
            result.append(f"{prefix_str}{node['feature']} (信息增益: {node['gain']:.3f}, 样本数: {node['samples']})")
            children = list(node['children'].items())
            for i, (value, child) in enumerate(children):
                is_last_child = i == len(children) - 1
                child_prefix = prefix + ('    ' if is_last else '│   ')
                result.append(child_prefix + ('└── ' if is_last_child else '├── ') + value)
                result.extend(
                    self.get_tree_structure(child, child_prefix + ('    ' if is_last_child else '│   '), is_last_child))
        return result


class WatermelonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("西瓜决策树分类器")
        self.model = WatermelonDecisionTree()
        self.data = None
        self.accuracy = None
        self.setup_gui()

    def setup_gui(self):
        """设置GUI界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 数据显示区域
        ttk.Label(main_frame, text="西瓜数据集", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
        self.data_text = tk.Text(main_frame, height=10, width=80)
        self.data_text.grid(row=1, column=0, columnspan=2, pady=5)

        # 训练按钮
        ttk.Button(main_frame, text="训练模型", command=self.train_model).grid(row=2, column=0, pady=5)
        self.accuracy_label = ttk.Label(main_frame, text="准确率: 未训练")
        self.accuracy_label.grid(row=2, column=1, pady=5)

        # 树结构显示
        ttk.Label(main_frame, text="决策树结构", font=("Arial", 12, "bold")).grid(row=3, column=0, columnspan=2, pady=5)
        self.tree_text = tk.Text(main_frame, height=10, width=80)
        self.tree_text.grid(row=4, column=0, columnspan=2, pady=5)

        # 决策树可视化
        ttk.Button(main_frame, text="显示决策树图", command=self.show_tree_graph).grid(row=5, column=0, columnspan=2,
                                                                                       pady=5)
        self.tree_canvas = tk.Canvas(main_frame, width=600, height=400)
        self.tree_canvas.grid(row=6, column=0, columnspan=2, pady=5)

        # 输入区域
        ttk.Label(main_frame, text="输入测试样本", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=2,
                                                                                    pady=5)
        self.inputs = {}
        features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
        options = {
            '色泽': ['青绿', '乌黑', '浅白'],
            '根蒂': ['蜷缩', '稍蜷', '硬挺'],
            '敲声': ['浊响', '沉闷', '清脆'],
            '纹理': ['清晰', '稍糊', '模糊'],
            '脐部': ['凹陷', '稍凹', '平坦'],
            '触感': ['硬滑', '软粘']
        }
        for i, feature in enumerate(features):
            ttk.Label(main_frame, text=feature).grid(row=8 + i // 2, column=0 if i % 2 == 0 else 1, sticky=tk.W, padx=5)
            self.inputs[feature] = ttk.Combobox(main_frame, values=options[feature], state="readonly")
            self.inputs[feature].grid(row=8 + i // 2, column=0 if i % 2 == 0 else 1, padx=5, pady=2)
            self.inputs[feature].set(options[feature][0])

        # 预测按钮
        ttk.Button(main_frame, text="预测", command=self.predict).grid(row=10, column=0, columnspan=2, pady=10)
        self.result_label = ttk.Label(main_frame, text="预测结果: ")
        self.result_label.grid(row=11, column=0, columnspan=2, pady=5)

    def train_model(self):
        """训练模型并显示数据和树结构"""
        try:
            self.accuracy, self.data = self.model.fit()
            self.accuracy_label.config(text=f"准确率: {self.accuracy:.1%}")
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(tk.END, self.data.to_string(index=False))
            self.tree_text.delete(1.0, tk.END)
            tree_structure = self.model.get_tree_structure()
            self.tree_text.insert(tk.END, "\n".join(tree_structure))
        except Exception as e:
            messagebox.showerror("错误", f"训练模型失败: {str(e)}")

    def show_tree_graph(self):
        """显示决策树图形"""
        if self.model.tree is None:
            messagebox.showwarning("警告", "请先训练模型！")
            return

        G = nx.DiGraph()
        pos = {}
        labels = {}

        def add_nodes_edges(node, parent=None, level=0, x=0, node_id=0):
            node_name = f"{node_id}"
            if node['type'] == 'leaf':
                label = f"结果: {node['label']}\n样本: {node['samples']}"
            else:
                label = f"{node['feature']}\n增益: {node['gain']:.3f}\n样本: {node['samples']}"
            G.add_node(node_name, label=label)
            pos[node_name] = (x, -level)
            labels[node_name] = label
            if parent is not None:
                G.add_edge(parent, node_name)

            if node['type'] == 'internal':
                children = list(node['children'].items())
                child_count = len(children)
                next_x = x - (child_count - 1) / 2
                for i, (value, child) in enumerate(children):
                    child_id = f"{node_id}.{i + 1}"
                    G.add_edge(node_name, child_id, label=value)
                    add_nodes_edges(child, node_name, level + 1, next_x + i, child_id)

        add_nodes_edges(self.model.tree)

        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(G, pos, ax=ax, with_labels=True, labels=labels, node_color='lightblue',
                node_size=2000, font_size=8, font_weight='bold', arrows=True)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        # 将图形嵌入到tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.tree_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack()
        plt.close(fig)

    def predict(self):
        """预测输入样本"""
        try:
            sample = {feature: combobox.get() for feature, combobox in self.inputs.items()}
            prediction = self.model.predict_single(sample)
            result = "好瓜" if prediction == "是" else "坏瓜"
            self.result_label.config(text=f"预测结果: {result}")
        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {str(e)}")


def main():
    root = tk.Tk()
    app = WatermelonGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()