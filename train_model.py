"""
完整的模型训练脚本
用于BBBP数据集（血脑屏障穿透性预测）
"""
import torch
import numpy as np
import sys
import os

sys.path.append('.')

from data.data_loader import DrugDataLoader
from features.molecular_features import MolecularFeaturizer, FeatureScaler
from models.drug_models import DrugPredictorMLP
from training.trainer import DrugModelTrainer, create_data_loaders
from evaluation.metrics import ModelEvaluator, ResultVisualizer


def main():
    print("="*60)
    print("药物筛选模型训练 - BBBP数据集")
    print("="*60)
    
    # ==================== 1. 数据加载 ====================
    print("\n[1/6] 加载数据...")
    loader = DrugDataLoader(data_dir='./data/raw')
    
    train_data, valid_data, test_data, tasks = loader.load_moleculenet_dataset(
        dataset_name='BBBP',
        featurizer='ECFP',
        split='scaffold'
    )
    
    print(f"[OK] 数据加载完成")
    
    # ==================== 2. 特征提取 ====================
    print("\n[2/6] 提取分子特征...")
    
    # 从DeepChem数据集中提取特征和标签
    X_train = train_data.X
    y_train = train_data.y.flatten()
    
    X_valid = valid_data.X
    y_valid = valid_data.y.flatten()
    
    X_test = test_data.X
    y_test = test_data.y.flatten()
    
    print(f"[OK] 特征提取完成")
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_valid.shape}")
    print(f"  测试集: {X_test.shape}")
    
    # ==================== 3. 创建数据加载器 ====================
    print("\n[3/6] 创建数据加载器...")
    train_loader, val_loader = create_data_loaders(
        X_train, y_train,
        X_valid, y_valid,
        batch_size=64
    )
    print(f"[OK] 数据加载器创建完成 (batch_size=64)")
    
    # ==================== 4. 创建模型 ====================
    print("\n[4/6] 创建模型...")
    # 使用实际特征维度（ECFP默认1024）
    input_dim = X_train.shape[1]
    model = DrugPredictorMLP(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        output_dim=1,
        dropout=0.3,
        task_type='binary'
    )
    
    print(f"[OK] 模型创建完成")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # ==================== 5. 训练模型 ====================
    print("\n[5/6] 开始训练...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  使用设备: {device}")
    
    trainer = DrugModelTrainer(
        model=model,
        device=device,
        learning_rate=0.0001,
        task_type='binary'
    )
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=15,
        save_best_model=True,
        model_save_path='./saved_models/bbbp_model.pth'
    )
    
    print(f"\n[OK] 训练完成！")
    
    # ==================== 6. 评估模型 ====================
    print("\n[6/6] 评估模型...")
    
    # 加载最佳模型
    model.load_state_dict(torch.load('./saved_models/bbbp_model.pth', 
                                     map_location=device))
    
    # 在测试集上预测
    y_pred = trainer.predict(X_test)
    y_pred_labels = (y_pred > 0.5).astype(int).flatten()
    
    # 计算指标
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(
        y_test.astype(int),
        y_pred_labels,
        y_prob=y_pred.flatten()
    )
    
    evaluator.print_metrics(metrics, "BBBP测试集评估结果")
    
    # 可视化
    print("\n生成可视化图表...")
    visualizer = ResultVisualizer(save_dir='./evaluation/figures')
    
    # 训练历史
    visualizer.plot_training_history(
        trainer.history,
        save_name='bbbp_training_history.png'
    )
    
    # ROC曲线
    visualizer.plot_roc_curve(
        y_test.astype(int),
        y_pred.flatten(),
        title='BBBP模型ROC曲线',
        save_name='bbbp_roc_curve.png'
    )
    
    # 混淆矩阵
    visualizer.plot_confusion_matrix(
        y_test.astype(int),
        y_pred_labels,
        labels=['不穿透', '穿透'],
        title='BBBP预测混淆矩阵',
        save_name='bbbp_confusion_matrix.png'
    )
    
    print("\n" + "="*60)
    print("训练流程完成！")
    print("="*60)
    print(f"模型保存位置: ./saved_models/bbbp_model.pth")
    print(f"图表保存位置: ./evaluation/figures/")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
