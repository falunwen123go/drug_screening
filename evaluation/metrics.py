"""
评估指标和可视化模块
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from typing import Dict, Optional, Tuple
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    """模型评估器"""
    
    @staticmethod
    def evaluate_regression(y_true: np.ndarray, 
                           y_pred: np.ndarray) -> Dict[str, float]:
        """
        回归任务评估
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            评估指标字典
        """
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'Pearson_r': np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
        }
        
        return metrics
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        分类任务评估
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率（用于计算AUC）
            
        Returns:
            评估指标字典
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'F1': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        # 计算AUC（如果提供概率）
        if y_prob is not None:
            try:
                metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['AUC-ROC'] = 0.0
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "评估结果"):
        """打印评估指标"""
        print("\n" + "=" * 50)
        print(title)
        print("=" * 50)
        for metric_name, value in metrics.items():
            print(f"{metric_name:15s}: {value:.4f}")
        print("=" * 50 + "\n")


class ResultVisualizer:
    """结果可视化工具"""
    
    def __init__(self, save_dir: str = './evaluation/figures'):
        """
        Args:
            save_dir: 图表保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_history(self, 
                             history: Dict,
                             save_name: str = 'training_history.png'):
        """
        绘制训练历史曲线
        
        Args:
            history: 训练历史字典，包含train_loss, val_loss等
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss曲线
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Metric曲线
        if 'train_metric' in history:
            axes[1].plot(history['train_metric'], label='Train Metric', linewidth=2)
            axes[1].plot(history['val_metric'], label='Val Metric', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Metric', fontsize=12)
            axes[1].set_title('Training and Validation Metric', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存到: {save_path}")
        plt.close()
    
    def plot_regression_results(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               title: str = "回归预测结果",
                               save_name: str = 'regression_plot.png'):
        """
        绘制回归预测散点图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 散点图
        ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        
        # 理想线（y=x）
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
        
        # 计算指标
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        ax.set_xlabel('True Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'{title}\nR² = {r2:.3f}, RMSE = {rmse:.3f}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"回归结果图已保存到: {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             labels: Optional[list] = None,
                             title: str = "混淆矩阵",
                             save_name: str = 'confusion_matrix.png'):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 类别标签
            title: 图表标题
            save_name: 保存文件名
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
        plt.close()
    
    def plot_roc_curve(self,
                      y_true: np.ndarray,
                      y_prob: np.ndarray,
                      title: str = "ROC曲线",
                      save_name: str = 'roc_curve.png'):
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_prob: 预测概率
            title: 图表标题
            save_name: 保存文件名
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存到: {save_path}")
        plt.close()
    
    def plot_feature_distribution(self,
                                  features: np.ndarray,
                                  feature_names: Optional[list] = None,
                                  title: str = "特征分布",
                                  save_name: str = 'feature_distribution.png'):
        """
        绘制特征分布直方图
        
        Args:
            features: 特征矩阵
            feature_names: 特征名称列表
            title: 图表标题
            save_name: 保存文件名
        """
        n_features = min(features.shape[1], 12)  # 最多显示12个特征
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        axes = axes.flatten()
        
        for i in range(n_features):
            axes[i].hist(features[:, i], bins=30, alpha=0.7, edgecolor='black')
            if feature_names and i < len(feature_names):
                axes[i].set_title(feature_names[i], fontsize=10)
            else:
                axes[i].set_title(f'Feature {i}', fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_features, 12):
            axes[i].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征分布图已保存到: {save_path}")
        plt.close()


if __name__ == "__main__":
    # 测试评估器
    print("测试回归评估...")
    y_true_reg = np.random.randn(100)
    y_pred_reg = y_true_reg + np.random.randn(100) * 0.3
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_regression(y_true_reg, y_pred_reg)
    evaluator.print_metrics(metrics, "回归模型评估")
    
    # 测试可视化
    print("\n测试可视化...")
    visualizer = ResultVisualizer()
    
    # 训练历史
    history = {
        'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
        'val_loss': [1.1, 0.85, 0.65, 0.55, 0.5],
        'train_metric': [0.5, 0.6, 0.7, 0.75, 0.8],
        'val_metric': [0.48, 0.58, 0.68, 0.72, 0.75]
    }
    visualizer.plot_training_history(history)
    
    # 回归结果
    visualizer.plot_regression_results(y_true_reg, y_pred_reg)
    
    print("评估模块测试完成！")
