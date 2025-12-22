"""
完整的药物筛选模型训练脚本
支持多个MoleculeNet数据集：BBBP、ESOL、Tox21
数据下载到本地data目录
"""
import torch
import numpy as np
import sys
import os
import warnings

# 抑制所有警告
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 抑制RDKit警告
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append('.')

from data.data_loader import DrugDataLoader
from features.feature_extraction import MolecularFeaturizer  # 使用新的模块名
from models.drug_models import DrugPredictorMLP
from training.trainer import DrugModelTrainer, create_data_loaders
from evaluation.metrics import ModelEvaluator, ResultVisualizer


def train_bbbp():
    """训练BBBP血脑屏障穿透性预测模型"""
    print("\n" + "="*70)
    print("  BBBP 血脑屏障穿透性预测模型训练")
    print("  Blood-Brain Barrier Penetration Prediction")
    print("="*70)
    
    # 1. 数据加载（下载到本地data目录）
    print("\n[Step 1/6] 加载MoleculeNet BBBP数据集...")
    print("  数据将保存到: ./data/raw/bbbp/")
    loader = DrugDataLoader(data_dir='./data/raw')
    
    train_data, valid_data, test_data, tasks = loader.load_moleculenet_dataset(
        dataset_name='BBBP',
        featurizer='ECFP',
        split='scaffold',
        save_local=True  # 保存到本地CSV
    )
    
    # 2. 提取特征
    print("\n[Step 2/6] 提取分子特征 (ECFP指纹)...")
    X_train = train_data.X
    y_train = train_data.y.flatten()
    X_valid = valid_data.X
    y_valid = valid_data.y.flatten()
    X_test = test_data.X
    y_test = test_data.y.flatten()
    
    # 处理NaN
    y_train = np.nan_to_num(y_train, nan=0.0)
    y_valid = np.nan_to_num(y_valid, nan=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0)
    
    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  训练集: {len(X_train)} 样本 (正例比例: {y_train.mean():.2%})")
    print(f"  验证集: {len(X_valid)} 样本 (正例比例: {y_valid.mean():.2%})")
    print(f"  测试集: {len(X_test)} 样本 (正例比例: {y_test.mean():.2%})")
    
    # 3. 创建数据加载器
    print("\n[Step 3/6] 创建PyTorch数据加载器...")
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_valid, y_valid, batch_size=32
    )
    
    # 4. 创建模型
    print("\n[Step 4/6] 创建MLP神经网络模型...")
    input_dim = X_train.shape[1]
    model = DrugPredictorMLP(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        output_dim=1,
        dropout=0.5,
        task_type='binary'
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  网络结构: {input_dim} -> 512 -> 256 -> 128 -> 1")
    print(f"  总参数量: {param_count:,}")
    
    # 5. 训练
    print("\n[Step 5/6] 开始GPU加速训练...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  训练设备: {device}")
    if device == 'cuda':
        print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
    
    trainer = DrugModelTrainer(
        model=model,
        device=device,
        learning_rate=0.0005,
        task_type='binary'
    )
    
    os.makedirs('./saved_models', exist_ok=True)
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=15,
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
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(
        y_test.astype(int), y_pred_labels, y_prob=y_pred_prob
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
    visualizer.plot_roc_curve(y_test.astype(int), y_pred_prob, 
                              title='BBBP ROC Curve', save_name='bbbp_roc_curve.png')
    visualizer.plot_confusion_matrix(y_test.astype(int), y_pred_labels,
                                     labels=['Non-penetrating', 'Penetrating'],
                                     title='BBBP Confusion Matrix', 
                                     save_name='bbbp_confusion_matrix.png')
    
    return metrics


def train_esol():
    """训练ESOL水溶解度预测模型（回归任务）"""
    print("\n" + "="*70)
    print("  ESOL 水溶解度预测模型训练")
    print("  Aqueous Solubility Prediction (Regression)")
    print("="*70)
    
    # 1. 数据加载
    print("\n[Step 1/6] 加载MoleculeNet ESOL数据集...")
    loader = DrugDataLoader(data_dir='./data/raw')
    
    train_data, valid_data, test_data, tasks = loader.load_moleculenet_dataset(
        dataset_name='ESOL',
        featurizer='ECFP',
        split='scaffold'
    )
    
    # 2. 提取特征
    print("\n[Step 2/6] 提取分子特征...")
    X_train = train_data.X
    y_train = train_data.y.flatten()
    X_valid = valid_data.X
    y_valid = valid_data.y.flatten()
    X_test = test_data.X
    y_test = test_data.y.flatten()
    
    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_valid)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    print(f"  溶解度范围: [{y_train.min():.2f}, {y_train.max():.2f}] log mol/L")
    
    # 3. 创建数据加载器
    print("\n[Step 3/6] 创建数据加载器...")
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_valid, y_valid, batch_size=32
    )
    
    # 4. 创建回归模型
    print("\n[Step 4/6] 创建回归模型...")
    input_dim = X_train.shape[1]
    model = DrugPredictorMLP(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        output_dim=1,
        dropout=0.3,
        task_type='regression'
    )
    
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. 训练
    print("\n[Step 5/6] 开始训练...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = DrugModelTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        task_type='regression'
    )
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=150,
        early_stopping_patience=20,
        save_best_model=True,
        model_save_path='./saved_models/esol_model.pth'
    )
    
    # 6. 评估
    print("\n[Step 6/6] 模型评估...")
    model.load_state_dict(torch.load('./saved_models/esol_model.pth', 
                                     map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_test_tensor).cpu().numpy().flatten()
    
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
