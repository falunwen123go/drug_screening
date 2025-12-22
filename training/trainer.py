"""
模型训练模块
提供训练、验证、早停等功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple, Dict
from tqdm import tqdm
import os


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善幅度
            mode: 'min'表示指标越小越好，'max'表示越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前评估指标
            
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class DrugModelTrainer:
    """药物模型训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001,
                 task_type: str = 'regression'):
        """
        Args:
            model: PyTorch模型
            device: 训练设备 ('cuda' or 'cpu')
            learning_rate: 学习率
            task_type: 任务类型 ('regression', 'binary', 'multiclass')
        """
        self.model = model.to(device)
        self.device = device
        self.task_type = task_type
        
        # 损失函数
        if task_type == 'regression':
            self.criterion = nn.MSELoss()
        elif task_type == 'binary':
            self.criterion = nn.BCEWithLogitsLoss()  # 更稳定，输入为logits
        elif task_type == 'multiclass':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': []
        }
        
    def train_epoch(self, train_loader: DataLoader, show_progress: bool = False) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            show_progress: 是否显示进度条
            
        Returns:
            (平均loss, 平均指标)
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # 简化进度条显示
        loader = tqdm(train_loader, desc='Training', leave=False, ncols=80, 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') if show_progress else train_loader
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            
            # 计算损失
            loss = self.criterion(predictions.squeeze(), batch_y.squeeze())
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        metric = self._calculate_metric(np.array(all_predictions), np.array(all_targets))
        
        return avg_loss, metric
    
    def validate(self, val_loader: DataLoader, show_progress: bool = False) -> Tuple[float, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            show_progress: 是否显示进度条
            
        Returns:
            (平均loss, 平均指标)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            # 简化进度条显示
            loader = tqdm(val_loader, desc='Validation', leave=False, ncols=80,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') if show_progress else val_loader
            
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 前向传播
                predictions = self.model(batch_x)
                
                # 计算损失
                loss = self.criterion(predictions.squeeze(), batch_y.squeeze())
                
                # 记录
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metric = self._calculate_metric(np.array(all_predictions), np.array(all_targets))
        
        return avg_loss, metric
    
    def _calculate_metric(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        计算评估指标
        
        Args:
            predictions: 预测值
            targets: 真实值
            
        Returns:
            评估指标值
        """
        if self.task_type == 'regression':
            # RMSE
            mse = np.mean((predictions - targets) ** 2)
            return np.sqrt(mse)
        elif self.task_type == 'binary':
            # AUC-ROC (简化版：使用准确率)
            pred_labels = (predictions > 0.5).astype(int)
            accuracy = np.mean(pred_labels == targets)
            return accuracy
        elif self.task_type == 'multiclass':
            # 准确率
            pred_labels = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_labels == targets)
            return accuracy
        else:
            return 0.0
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            early_stopping_patience: Optional[int] = 10,
            save_best_model: bool = True,
            model_save_path: str = './saved_models/best_model.pth'):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            save_best_model: 是否保存最佳模型
            model_save_path: 模型保存路径
        """
        print(f"开始训练... Device: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"训练配置: epochs={epochs}, early_stopping_patience={early_stopping_patience}")
        print("-" * 70)
        
        # 早停
        early_stopping = None
        if early_stopping_patience:
            early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')
        
        best_val_loss = float('inf')
        
        # 使用tqdm显示总进度
        epoch_pbar = tqdm(range(epochs), desc='Training', unit='epoch', ncols=100)
        
        for epoch in epoch_pbar:
            # 训练
            train_loss, train_metric = self.train_epoch(train_loader, show_progress=False)
            
            # 验证
            val_loss, val_metric = self.validate(val_loader, show_progress=False)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metric'].append(train_metric)
            self.history['val_metric'].append(val_metric)
            
            # 更新进度条描述
            metric_name = 'RMSE' if self.task_type == 'regression' else 'Acc'
            epoch_pbar.set_postfix({
                'TrLoss': f'{train_loss:.4f}',
                f'Tr{metric_name}': f'{train_metric:.4f}',
                'ValLoss': f'{val_loss:.4f}',
                f'Val{metric_name}': f'{val_metric:.4f}'
            })
            
            # 保存最佳模型 / 更新最佳损失
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_best_model:
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), model_save_path)
            
            # 早停检查
            if early_stopping and early_stopping(val_loss):
                epoch_pbar.close()
                print(f"\n早停触发！在第 {epoch+1} 轮停止训练 (best_val_loss={best_val_loss:.4f})")
                break
        
        print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入特征
            
        Returns:
            预测结果
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy()
    
    def load_model(self, model_path: str):
        """加载模型权重"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"模型已从 {model_path} 加载")


def create_data_loaders(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        batch_size: 批次大小
        
    Returns:
        (train_loader, val_loader)
    """
    # 转换为Tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 创建Dataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试训练器
    from models.drug_models import DrugPredictorMLP
    
    print("生成模拟数据...")
    X_train = np.random.randn(1000, 2048)
    y_train = np.random.randn(1000)
    X_val = np.random.randn(200, 2048)
    y_val = np.random.randn(200)
    
    print("创建数据加载器...")
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32)
    
    print("创建模型...")
    model = DrugPredictorMLP(input_dim=2048, hidden_dims=[512, 256], output_dim=1)
    
    print("创建训练器...")
    trainer = DrugModelTrainer(model, learning_rate=0.001, task_type='regression')
    
    print("开始训练...")
    trainer.fit(train_loader, val_loader, epochs=5, early_stopping_patience=3)
