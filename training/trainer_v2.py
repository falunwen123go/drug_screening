"""
模型训练模块 V2 - 简洁版
提供训练、验证、早停等功能
移除批次级进度条，只保留简洁的表格输出
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple, Dict, List
import os
import time


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, mode: str = 'min'):
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
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class DrugModelTrainer:
    """药物模型训练器 V2 - 简洁版（增强版：支持数据增强和正则化）"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 task_type: str = 'regression',
                 use_scheduler: bool = True,
                 scheduler_type: str = 'plateau',
                 # 数据增强参数
                 use_data_augmentation: bool = False,
                 noise_std: float = 0.1,
                 feature_dropout: float = 0.1,
                 mixup_alpha: float = 0.0,
                 # Label Smoothing参数
                 label_smoothing: float = 0.0):
        """
        初始化训练器
        
        Args:
            model: PyTorch模型
            device: 训练设备 ('cuda' or 'cpu')
            learning_rate: 学习率
            weight_decay: 权重衰减（L2正则化）
            task_type: 任务类型 ('regression', 'binary', 'multiclass')
            use_scheduler: 是否使用学习率调度器
            scheduler_type: 调度器类型 ('plateau' or 'cosine')
            use_data_augmentation: 是否使用数据增强
            noise_std: 高斯噪声标准差
            feature_dropout: 特征随机遮蔽比例
            mixup_alpha: Mixup增强的alpha参数（0表示不使用）
            label_smoothing: 标签平滑参数（0表示不使用）
        """
        self.model = model.to(device)
        self.device = device
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler
        
        # 数据增强参数
        self.use_data_augmentation = use_data_augmentation
        self.noise_std = noise_std
        self.feature_dropout = feature_dropout
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        
        # 损失函数（支持Label Smoothing）
        if task_type == 'regression':
            self.criterion = nn.MSELoss()
        elif task_type == 'binary':
            # 二分类使用自定义带平滑的BCE
            self.criterion = nn.BCEWithLogitsLoss()
        elif task_type == 'multiclass':
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # 优化器 - AdamW 带权重衰减
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = None
        if use_scheduler:
            if scheduler_type == 'plateau':
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer, 
                    mode='min', 
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7
                )
            elif scheduler_type == 'cosine':
                self.scheduler = CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=20,
                    T_mult=2,
                    eta_min=1e-7
                )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': [],
            'learning_rate': []
        }
        
    def _apply_data_augmentation(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用数据增强策略
        
        Args:
            batch_x: 输入特征
            batch_y: 标签
            
        Returns:
            增强后的 (batch_x, batch_y)
        """
        if not self.use_data_augmentation:
            return batch_x, batch_y
        
        # 1. 高斯噪声注入
        if self.noise_std > 0:
            noise = torch.randn_like(batch_x) * self.noise_std
            batch_x = batch_x + noise
        
        # 2. 特征随机遮蔽（Feature Dropout）
        if self.feature_dropout > 0:
            mask = torch.bernoulli(torch.ones_like(batch_x) * (1 - self.feature_dropout))
            batch_x = batch_x * mask / (1 - self.feature_dropout)  # 缩放以保持期望值
        
        # 3. Mixup增强（可选）
        if self.mixup_alpha > 0:
            batch_x, batch_y = self._mixup(batch_x, batch_y)
        
        return batch_x, batch_y
    
    def _mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mixup数据增强
        """
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y
    
    def _apply_label_smoothing(self, y: torch.Tensor) -> torch.Tensor:
        """
        对二分类标签应用Label Smoothing
        """
        if self.task_type == 'binary' and self.label_smoothing > 0:
            # 将0变成label_smoothing，将1变成1-label_smoothing
            y = y * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return y
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        训练一个epoch（无进度条，静默模式）
        支持数据增强和Label Smoothing
        
        Returns:
            (平均损失, 评估指标)
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 应用数据增强
            batch_x, batch_y = self._apply_data_augmentation(batch_x, batch_y)
            
            # 应用Label Smoothing
            batch_y_smooth = self._apply_label_smoothing(batch_y)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            
            # 计算损失（使用平滑后的标签）
            loss = self.criterion(predictions.squeeze(), batch_y_smooth.squeeze())
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录（使用原始标签计算指标）
            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())  # 原始标签
        
        avg_loss = total_loss / len(train_loader)
        metric = self._calculate_metric(np.array(all_predictions), np.array(all_targets))
        
        return avg_loss, metric
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        验证模型（无进度条，静默模式）
        
        Returns:
            (平均损失, 评估指标)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions.squeeze(), batch_y.squeeze())
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metric = self._calculate_metric(np.array(all_predictions), np.array(all_targets))
        
        return avg_loss, metric
    
    def _calculate_metric(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算评估指标"""
        if self.task_type == 'regression':
            # RMSE
            mse = np.mean((predictions - targets) ** 2)
            return np.sqrt(mse)
        elif self.task_type == 'binary':
            # 准确率 (简化版)
            pred_labels = (predictions > 0.5).astype(int)
            accuracy = np.mean(pred_labels.flatten() == targets.flatten())
            return accuracy
        elif self.task_type == 'multiclass':
            # 准确率
            pred_labels = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_labels == targets)
            return accuracy
        else:
            return 0.0
    
    def _print_progress_bar(self, current: int, total: int, width: int = 30):
        """打印简单的进度条"""
        percent = current / total
        filled = int(width * percent)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {current:3d}/{total}"
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            early_stopping_patience: Optional[int] = 10,
            save_best_model: bool = True,
            model_save_path: str = './saved_models/best_model.pth',
            verbose: bool = True):
        """
        训练模型 - 简洁版本
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            save_best_model: 是否保存最佳模型
            model_save_path: 模型保存路径
            verbose: 是否显示详细信息
        """
        # 打印训练配置
        if verbose:
            print("\n" + "=" * 70)
            print("                        训练配置")
            print("=" * 70)
            print(f"  设备: {self.device}", end="")
            if self.device == 'cuda':
                print(f" ({torch.cuda.get_device_name(0)})")
            else:
                print()
            print(f"  模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"  学习率: {self.learning_rate}")
            print(f"  总轮数: {epochs}")
            print(f"  早停耐心: {early_stopping_patience}")
            print(f"  学习率调度: {'启用' if self.use_scheduler else '禁用'}")
            print("=" * 70)
        
        # 早停
        early_stopping = None
        if early_stopping_patience:
            early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')
        
        best_val_loss = float('inf')
        best_epoch = 0
        metric_name = 'RMSE' if self.task_type == 'regression' else 'Acc'
        
        # 打印表头
        if verbose:
            print()
            print(f"┌{'─'*8}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*10}┐")
            print(f"│{'Epoch':^8}│{'TrainLoss':^12}│{f'Train{metric_name}':^12}│{'ValLoss':^12}│{f'Val{metric_name}':^12}│{'LR':^12}│{'Status':^10}│")
            print(f"├{'─'*8}┼{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*10}┤")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练一个epoch
            train_loss, train_metric = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_metric = self.validate(val_loader)
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metric'].append(train_metric)
            self.history['val_metric'].append(val_metric)
            self.history['learning_rate'].append(current_lr)
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 状态标记
            status = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                status = "★ Best"
                if save_best_model:
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), model_save_path)
            
            # 打印epoch结果
            if verbose:
                lr_str = f"{current_lr:.2e}" if current_lr < 0.001 else f"{current_lr:.6f}"
                print(f"│{epoch+1:^8}│{train_loss:^12.4f}│{train_metric:^12.4f}│{val_loss:^12.4f}│{val_metric:^12.4f}│{lr_str:^12}│{status:^10}│")
            
            # 早停检查
            if early_stopping and early_stopping(val_loss):
                if verbose:
                    print(f"└{'─'*8}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*10}┘")
                    print()
                    print(f"⚡ 早停触发！在第 {epoch+1} 轮停止训练")
                    print(f"   最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})")
                    print(f"   早停计数: {early_stopping_patience} 轮无改善")
                break
        else:
            if verbose:
                print(f"└{'─'*8}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*10}┘")
        
        # 训练完成信息
        total_time = time.time() - start_time
        if verbose:
            print()
            print(f"✓ 训练完成")
            print(f"   总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
            print(f"   最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})")
            print(f"   模型已保存到: {model_save_path}")
        
        return self.history
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            预测结果数组
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                pred = self.model(batch_x)
                predictions.extend(pred.cpu().numpy())
        
        return np.array(predictions)
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            包含各项指标的字典
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                predictions = self.model(batch_x)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        if self.task_type == 'regression':
            mse = np.mean((predictions - targets) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - targets))
            
            # R^2
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Pearson相关系数
            if len(predictions) > 1:
                corr = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
            else:
                corr = 0
            
            return {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Pearson': corr
            }
        
        elif self.task_type == 'binary':
            from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
            
            # Sigmoid处理
            pred_probs = 1 / (1 + np.exp(-predictions.flatten()))
            pred_labels = (pred_probs > 0.5).astype(int)
            targets_flat = targets.flatten().astype(int)
            
            return {
                'AUC-ROC': roc_auc_score(targets_flat, pred_probs),
                'F1': f1_score(targets_flat, pred_labels),
                'Precision': precision_score(targets_flat, pred_labels),
                'Recall': recall_score(targets_flat, pred_labels),
                'Accuracy': accuracy_score(targets_flat, pred_labels)
            }
        
        else:
            pred_labels = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_labels == targets)
            return {'Accuracy': accuracy}
    
    def load_model(self, model_path: str):
        """加载模型权重"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
    def get_history(self) -> Dict[str, List[float]]:
        """获取训练历史"""
        return self.history
