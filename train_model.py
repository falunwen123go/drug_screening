"""
完整的药物筛选模型训练脚本
支持多个MoleculeNet数据集：BBBP、ESOL、Tox21
数据下载到本地data目录
使用分层采样确保训练/验证/测试集分布一致，避免过拟合
"""
import sys
import os

# === Windows控制台编码修复（必须在最前面） ===
if sys.platform == 'win32':
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    # 尝试设置控制台代码页
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8
        kernel32.SetConsoleCP(65001)
    except:
        pass
    
    # 使用ASCII安全输出
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import numpy as np
import warnings

# 抑制所有警告
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 抑制RDKit警告
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# 抑制DeepChem的特征化警告输出
import logging
logging.getLogger('deepchem').setLevel(logging.ERROR)

# 重定向DeepChem的print输出（抑制Failed to featurize警告）
class SuppressDeepChemOutput:
    """临时抑制DeepChem的标准输出"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

sys.path.append('.')

from data.data_loader import DrugDataLoader
from features.feature_extraction import MolecularFeaturizer  # 使用新的模块名
from models.drug_models import DrugPredictorMLP, DrugPredictorMLPv2  # 添加增强版模型
from training.trainer_v2 import DrugModelTrainer  # 使用简洁版训练器
from training.trainer import create_data_loaders  # 数据加载器辅助函数
from evaluation.metrics import ModelEvaluator, ResultVisualizer


def train_bbbp():
    """训练BBBP血脑屏障穿透性预测模型 - 使用分层采样避免过拟合"""
    print("\n" + "="*70)
    print("  BBBP 血脑屏障穿透性预测模型训练")
    print("  Blood-Brain Barrier Penetration Prediction")
    print("  使用分层采样确保类别分布一致")
    print("="*70)
    
    # 1. 数据加载 - 使用分层采样
    print("\n[Step 1/6] 加载MoleculeNet BBBP数据集（分层采样）...")
    print("  数据将保存到: ./data/raw/bbbp/")
    print("  (正在加载中，请稍候...)")
    loader = DrugDataLoader(data_dir='./data/raw')
    
    # 使用上下文管理器抑制DeepChem的警告输出
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    # 使用分层采样确保类别分布一致
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        X_train, y_train, X_valid, y_valid, X_test, y_test, tasks = \
            loader.load_moleculenet_with_stratified_split(
                dataset_name='BBBP',
                featurizer='ECFP',
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                random_state=42
            )
    
    print("  数据加载完成!")
    
    # 2. 数据统计
    print("\n[Step 2/6] 数据分布统计...")
    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  训练集: {len(X_train)} 样本 (正例比例: {(y_train > 0.5).mean():.2%})")
    print(f"  验证集: {len(X_valid)} 样本 (正例比例: {(y_valid > 0.5).mean():.2%})")
    print(f"  测试集: {len(X_test)} 样本 (正例比例: {(y_test > 0.5).mean():.2%})")
    
    # 3. 创建数据加载器
    print("\n[Step 3/6] 创建PyTorch数据加载器...")
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_valid, y_valid, batch_size=32
    )
    
    # 4. 创建增强版模型 - 更强的正则化
    print("\n[Step 4/6] 创建增强版MLP神经网络模型...")
    input_dim = X_train.shape[1]
    model = DrugPredictorMLPv2(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],  # 更小的网络减少过拟合
        output_dim=1,
        dropout=0.6,           # 更高的dropout防止过拟合
        use_batch_norm=True,
        activation='leaky_relu',
        task_type='binary'
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  网络结构: {input_dim} -> 128 -> 64 -> 32 -> 1")
    print(f"  总参数量: {param_count:,}")
    print(f"  正则化: Dropout=0.6, BatchNorm, 渐进式Dropout")
    
    # 5. 训练 - 使用数据增强和Label Smoothing防止过拟合
    print("\n[Step 5/6] 开始GPU加速训练（启用数据增强）...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  训练设备: {device}")
    if device == 'cuda':
        print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
    
    trainer = DrugModelTrainer(
        model=model,
        device=device,
        learning_rate=0.0003,  # 更低学习率避免过快收敛
        weight_decay=5e-3,     # 更强L2正则化
        task_type='binary',
        use_scheduler=True,
        scheduler_type='plateau',
        # 数据增强参数
        use_data_augmentation=True,
        noise_std=0.05,        # 添加高斯噪声
        feature_dropout=0.15,  # 随机遮蔽15%特征
        mixup_alpha=0.2,       # Mixup增强
        label_smoothing=0.1    # 标签平滑
    )
    
    print(f"  数据增强: 高斯噪声(std=0.05), 特征遮蔽(15%), Mixup(alpha=0.2)")
    print(f"  正则化: L2={5e-3}, Label Smoothing=0.1")
    
    os.makedirs('./saved_models', exist_ok=True)
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,            # 减少epochs，配合早停
        early_stopping_patience=15,  # 更早停止防止过拟合
        save_best_model=True,
        model_save_path='./saved_models/bbbp_model.pth'
    )
    
    # 6. 评估
    print("\n[Step 6/6] 模型评估...")
    model.load_state_dict(torch.load('./saved_models/bbbp_model.pth', 
                                     map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred_prob = torch.sigmoid(model(X_test_tensor)).cpu().numpy().flatten()
    
    y_pred_labels = (y_pred_prob > 0.5).astype(int)
    
    # 转换标签为整数
    y_test_int = (y_test > 0.5).astype(int)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(
        y_test_int, y_pred_labels, y_prob=y_pred_prob
    )
    
    print("\n" + "-"*50)
    print("BBBP 测试集评估结果:")
    print("-"*50)
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1-Score:  {metrics['F1']:.4f}")
    print(f"  AUC-ROC:   {metrics['AUC-ROC']:.4f}")
    print("-"*50)
    
    # 可视化
    os.makedirs('./evaluation/figures', exist_ok=True)
    visualizer = ResultVisualizer(save_dir='./evaluation/figures')
    
    visualizer.plot_training_history(trainer.history, save_name='bbbp_training_history.png')
    visualizer.plot_roc_curve(y_test_int, y_pred_prob, 
                              title='BBBP ROC Curve', save_name='bbbp_roc_curve.png')
    visualizer.plot_confusion_matrix(y_test_int, y_pred_labels,
                                     labels=['Non-penetrating', 'Penetrating'],
                                     title='BBBP Confusion Matrix', 
                                     save_name='bbbp_confusion_matrix.png')
    
    return metrics


def train_esol():
    """训练ESOL水溶解度预测模型（回归任务）- 使用随机分割"""
    print("\n" + "="*70)
    print("  ESOL 水溶解度预测模型训练")
    print("  Aqueous Solubility Prediction (Regression)")
    print("  使用特征缩放 + 数据增强防止过拟合")
    print("="*70)
    
    # 1. 数据加载 - 使用分层采样函数（回归任务会自动使用随机分割）
    print("\n[Step 1/6] 加载MoleculeNet ESOL数据集...")
    print("  (正在加载中，请稍候...)")
    loader = DrugDataLoader(data_dir='./data/raw')
    
    import io
    from contextlib import redirect_stdout, redirect_stderr
    from sklearn.preprocessing import StandardScaler
    
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        X_train, y_train, X_valid, y_valid, X_test, y_test, tasks = \
            loader.load_moleculenet_with_stratified_split(
                dataset_name='ESOL',
                featurizer='ECFP',
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                random_state=42
            )
    
    print("  数据加载完成!")
    
    # 2. 特征缩放 - 对小数据集非常重要
    print("\n[Step 2/6] 特征预处理（StandardScaler缩放）...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    # 同时对标签进行标准化（回归任务有帮助）
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_scaled = (y_train - y_mean) / y_std
    y_valid_scaled = (y_valid - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std
    
    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  训练集: {len(X_train)} 样本, 原始均值: {y_train.mean():.3f}")
    print(f"  验证集: {len(X_valid)} 样本, 原始均值: {y_valid.mean():.3f}")
    print(f"  测试集: {len(X_test)} 样本, 原始均值: {y_test.mean():.3f}")
    print(f"  溶解度范围: [{y_train.min():.2f}, {y_train.max():.2f}] log mol/L")
    print(f"  标签标准化: mean={y_mean:.3f}, std={y_std:.3f}")
    
    # 3. 创建数据加载器（使用缩放后的数据）
    print("\n[Step 3/6] 创建数据加载器...")
    train_loader, val_loader = create_data_loaders(
        X_train_scaled, y_train_scaled, X_valid_scaled, y_valid_scaled, batch_size=32
    )
    
    # 4. 创建更简单的回归模型 - 防止过拟合
    print("\n[Step 4/6] 创建简化回归模型（防过拟合）...")
    input_dim = X_train.shape[1]
    model = DrugPredictorMLPv2(
        input_dim=input_dim,
        hidden_dims=[64, 32],     # 更小的网络防止过拟合
        output_dim=1,
        dropout=0.5,              # 更高的dropout
        use_batch_norm=True,
        activation='leaky_relu',
        task_type='regression'
    )
    
    print(f"  网络结构: {input_dim} -> 64 -> 32 -> 1")
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  正则化: Dropout=0.5, BatchNorm, 渐进式Dropout")
    
    # 5. 训练 - 启用数据增强
    print("\n[Step 5/6] 开始训练（启用数据增强）...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = DrugModelTrainer(
        model=model,
        device=device,
        learning_rate=0.0003,     # 更低学习率
        weight_decay=1e-2,        # 更强L2正则化
        task_type='regression',
        use_scheduler=True,
        scheduler_type='plateau',
        # 数据增强参数
        use_data_augmentation=True,
        noise_std=0.1,            # 高斯噪声
        feature_dropout=0.2,      # 特征遮蔽
        mixup_alpha=0.1,          # Mixup（回归任务用较小alpha）
        label_smoothing=0.0       # 回归任务不用label smoothing
    )
    
    print(f"  数据增强: 高斯噪声(std=0.1), 特征遮蔽(20%), Mixup(alpha=0.1)")
    print(f"  正则化: L2={1e-2}, Dropout=0.5")
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=80,                # 减少epochs
        early_stopping_patience=20,  # 更早停止
        save_best_model=True,
        model_save_path='./saved_models/esol_model.pth'
    )
    
    # 6. 评估（需要反标准化预测结果）
    print("\n[Step 6/6] 模型评估...")
    model.load_state_dict(torch.load('./saved_models/esol_model.pth', 
                                     map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy().flatten()
    
    # 反标准化预测值
    y_pred = y_pred_scaled * y_std + y_mean
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_regression(y_test, y_pred)
    
    print("\n" + "-"*50)
    print("ESOL 测试集评估结果:")
    print("-"*50)
    print(f"  RMSE:      {metrics['RMSE']:.4f}")
    print(f"  MAE:       {metrics['MAE']:.4f}")
    print(f"  R^2:       {metrics['R2']:.4f}")
    print(f"  Pearson r: {metrics['Pearson_r']:.4f}")
    print("-"*50)
    
    # 可视化
    visualizer = ResultVisualizer(save_dir='./evaluation/figures')
    visualizer.plot_training_history(trainer.history, save_name='esol_training_history.png')
    visualizer.plot_regression_results(y_test, y_pred, 
                                       title='ESOL Prediction vs Actual',
                                       save_name='esol_scatter.png')
    
    return metrics


def main():
    print("\n")
    print("="*70)
    print("       基于大数据分析的药物筛选系统 - 完整训练流程")
    print("       Drug Screening System Based on Big Data Analysis")
    print("="*70)
    print(f"\n运行时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = {}
    
    # 训练BBBP分类模型
    try:
        bbbp_metrics = train_bbbp()
        results['BBBP'] = bbbp_metrics
    except Exception as e:
        print(f"\nBBBP训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 训练ESOL回归模型
    try:
        esol_metrics = train_esol()
        results['ESOL'] = esol_metrics
    except Exception as e:
        print(f"\nESOL训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 总结
    print("\n")
    print("="*70)
    print("                    训练完成总结")
    print("="*70)
    
    if 'BBBP' in results:
        print("\n[BBBP] 血脑屏障穿透性预测 (分类任务):")
        print(f"  - AUC-ROC: {results['BBBP']['AUC-ROC']:.4f}")
        print(f"  - F1-Score: {results['BBBP']['F1']:.4f}")
        print(f"  - 模型: saved_models/bbbp_model.pth")
    
    if 'ESOL' in results:
        print("\n[ESOL] 水溶解度预测 (回归任务):")
        print(f"  - RMSE: {results['ESOL']['RMSE']:.4f}")
        print(f"  - R^2: {results['ESOL']['R2']:.4f}")
        print(f"  - 模型: saved_models/esol_model.pth")
    
    print("\n[图表输出]:")
    print("  - evaluation/figures/bbbp_*.png")
    print("  - evaluation/figures/esol_*.png")
    
    print("\n[Web界面]:")
    print("  运行命令: streamlit run web/app.py")
    print("  访问地址: http://localhost:8501")
    
    print("\n" + "="*70)
    print("                    所有训练任务完成!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
