"""
模型评估脚本 - 评估已训练的BBBP模型
"""
import torch
import numpy as np
import sys
import os

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append('.')

from data.data_loader import DrugDataLoader
from models.drug_models import DrugPredictorMLP
from evaluation.metrics import ModelEvaluator, ResultVisualizer


def main():
    print("="*60)
    print("BBBP模型评估")
    print("="*60)
    
    # 1. 加载数据
    print("\n[1/4] 加载测试数据...")
    loader = DrugDataLoader(data_dir='./data/raw')
    
    train_data, valid_data, test_data, tasks = loader.load_moleculenet_dataset(
        dataset_name='BBBP',
        featurizer='ECFP',
        split='scaffold'
    )
    
    X_test = test_data.X
    y_test = test_data.y.flatten()
    
    print(f"[OK] 测试集: {X_test.shape[0]} 样本")
    
    # 2. 加载模型
    print("\n[2/4] 加载模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  设备: {device}")
    
    input_dim = X_test.shape[1]
    model = DrugPredictorMLP(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        output_dim=1,
        dropout=0.3,
        task_type='binary'
    )
    
    model.load_state_dict(torch.load('./saved_models/bbbp_model.pth', 
                                     map_location=device,
                                     weights_only=True))
    model.to(device)
    model.eval()
    print(f"[OK] 模型加载完成")
    
    # 3. 预测
    print("\n[3/4] 进行预测...")
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = torch.sigmoid(model(X_test_tensor)).cpu().numpy().flatten()
    
    y_pred_labels = (y_pred > 0.5).astype(int)
    print(f"[OK] 预测完成")
    
    # 4. 评估
    print("\n[4/4] 计算评估指标...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(
        y_test.astype(int),
        y_pred_labels,
        y_prob=y_pred
    )
    
    print("\n" + "="*60)
    print("BBBP血脑屏障穿透性预测 - 测试集评估结果")
    print("="*60)
    
    # 打印详细指标
    print(f"\n准确率 (Accuracy):     {metrics['Accuracy']:.4f}")
    print(f"精确率 (Precision):    {metrics['Precision']:.4f}")
    print(f"召回率 (Recall):       {metrics['Recall']:.4f}")
    print(f"F1分数 (F1-Score):     {metrics['F1']:.4f}")
    print(f"AUC-ROC:               {metrics['AUC-ROC']:.4f}")
    
    # 5. 生成可视化
    print("\n" + "="*60)
    print("生成可视化图表...")
    print("="*60)
    
    os.makedirs('./evaluation/figures', exist_ok=True)
    visualizer = ResultVisualizer(save_dir='./evaluation/figures')
    
    # ROC曲线
    visualizer.plot_roc_curve(
        y_test.astype(int),
        y_pred,
        title='BBBP Blood-Brain Barrier Penetration ROC Curve',
        save_name='bbbp_roc_curve.png'
    )
    print("  [OK] ROC曲线: evaluation/figures/bbbp_roc_curve.png")
    
    # 混淆矩阵
    visualizer.plot_confusion_matrix(
        y_test.astype(int),
        y_pred_labels,
        labels=['Non-penetrating', 'Penetrating'],
        title='BBBP Prediction Confusion Matrix',
        save_name='bbbp_confusion_matrix.png'
    )
    print("  [OK] 混淆矩阵: evaluation/figures/bbbp_confusion_matrix.png")
    
    # 预测分布
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    
    # 正负样本的预测分布
    pos_preds = y_pred[y_test == 1]
    neg_preds = y_pred[y_test == 0]
    
    plt.hist(neg_preds, bins=30, alpha=0.7, label='Non-penetrating (True)', color='blue')
    plt.hist(pos_preds, bins=30, alpha=0.7, label='Penetrating (True)', color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('BBBP Prediction Distribution')
    plt.legend()
    plt.savefig('./evaluation/figures/bbbp_prediction_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] 预测分布: evaluation/figures/bbbp_prediction_distribution.png")
    
    print("\n" + "="*60)
    print("评估完成!")
    print("="*60)
    print(f"模型文件: ./saved_models/bbbp_model.pth")
    print(f"图表目录: ./evaluation/figures/")
    print("="*60)
    
    return metrics


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
