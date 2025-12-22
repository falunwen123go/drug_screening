"""
深度学习模型定义
包含：MLP、CNN、Multi-task DNN等多种模型架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DrugPredictorMLP(nn.Module):
    """
    多层感知机(MLP)模型
    适用于基于分子指纹的预测任务
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 output_dim: int = 1,
                 dropout: float = 0.2,
                 task_type: str = 'regression'):
        """
        Args:
            input_dim: 输入特征维度（如2048维Morgan指纹）
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度（回归为1，多分类为类别数）
            dropout: Dropout比例
            task_type: 任务类型 ('regression', 'binary', 'multiclass')
        """
        super(DrugPredictorMLP, self).__init__()
        
        self.task_type = task_type
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_dim)
            
        Returns:
            预测输出 (logits for binary/multiclass, raw for regression)
        """
        output = self.network(x)
        
        # 注意：二分类任务输出logits，配合BCEWithLogitsLoss使用
        # 推理时需要手动应用sigmoid
        if self.task_type == 'multiclass':
            output = F.softmax(output, dim=1)
        
        return output


class SMILESCNN(nn.Module):
    """
    基于SMILES的1D-CNN模型
    用于处理SMILES字符序列
    """
    
    def __init__(self,
                 vocab_size: int = 35,
                 embedding_dim: int = 128,
                 num_filters: int = 128,
                 kernel_sizes: List[int] = [3, 5, 7],
                 output_dim: int = 1,
                 dropout: float = 0.3,
                 task_type: str = 'regression'):
        """
        Args:
            vocab_size: 词汇表大小（SMILES字符集大小）
            embedding_dim: 字符嵌入维度
            num_filters: 每个卷积核的数量
            kernel_sizes: 卷积核大小列表（多尺度）
            output_dim: 输出维度
            dropout: Dropout比例
            task_type: 任务类型
        """
        super(SMILESCNN, self).__init__()
        
        self.task_type = task_type
        
        # 字符嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 多尺度卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        # 全连接层
        total_filters = num_filters * len(kernel_sizes)
        self.fc1 = nn.Linear(total_filters, 256)
        self.fc2 = nn.Linear(256, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_length) - SMILES编码
            
        Returns:
            预测输出
        """
        # 嵌入: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # 转置为CNN输入格式: (batch, embed_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        # 多尺度卷积并池化
        conv_outputs = []
        for conv in self.convs:
            conv_out = self.relu(conv(embedded))  # (batch, num_filters, new_seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch, num_filters)
        
        # 拼接所有卷积输出
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch, total_filters)
        
        # 全连接层
        x = self.dropout(concatenated)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        
        # 激活函数
        if self.task_type == 'binary':
            output = torch.sigmoid(output)
        elif self.task_type == 'multiclass':
            output = F.softmax(output, dim=1)
        
        return output


class MultiTaskDNN(nn.Module):
    """
    多任务深度神经网络
    同时预测多个性质（如溶解度、毒性、活性等）
    """
    
    def __init__(self,
                 input_dim: int,
                 shared_hidden_dims: List[int] = [512, 256],
                 task_configs: List[dict] = None,
                 dropout: float = 0.2):
        """
        Args:
            input_dim: 输入特征维度
            shared_hidden_dims: 共享层维度
            task_configs: 任务配置列表，每个任务包含:
                         {'name': str, 'output_dim': int, 'task_type': str}
            dropout: Dropout比例
        """
        super(MultiTaskDNN, self).__init__()
        
        if task_configs is None:
            task_configs = [
                {'name': 'task1', 'output_dim': 1, 'task_type': 'regression'}
            ]
        
        self.task_configs = task_configs
        
        # 共享特征提取层
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in shared_hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.BatchNorm1d(hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # 任务特定的输出头
        self.task_heads = nn.ModuleDict()
        
        for task in task_configs:
            task_name = task['name']
            output_dim = task['output_dim']
            
            # 每个任务一个简单的输出层
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_dim)
            )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_dim)
            
        Returns:
            字典，包含每个任务的预测结果
        """
        # 共享特征提取
        shared_features = self.shared_network(x)
        
        # 每个任务的预测
        outputs = {}
        for task in self.task_configs:
            task_name = task['name']
            task_type = task['task_type']
            
            task_output = self.task_heads[task_name](shared_features)
            
            # 应用激活函数
            if task_type == 'binary':
                task_output = torch.sigmoid(task_output)
            elif task_type == 'multiclass':
                task_output = F.softmax(task_output, dim=1)
            
            outputs[task_name] = task_output
        
        return outputs


class ADMETPredictor(nn.Module):
    """
    ADMET性质预测模型
    预测吸收、分布、代谢、排泄、毒性相关性质
    """
    
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dims: List[int] = [1024, 512, 256],
                 dropout: float = 0.3):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度
            dropout: Dropout比例
        """
        super(ADMETPredictor, self).__init__()
        
        # 定义ADMET预测任务
        task_configs = [
            {'name': 'solubility', 'output_dim': 1, 'task_type': 'regression'},  # 溶解度
            {'name': 'bbb_penetration', 'output_dim': 1, 'task_type': 'binary'},  # 血脑屏障
            {'name': 'herg_toxicity', 'output_dim': 1, 'task_type': 'binary'},  # hERG毒性
            {'name': 'hepatotoxicity', 'output_dim': 1, 'task_type': 'binary'},  # 肝毒性
        ]
        
        # 使用多任务模型
        self.model = MultiTaskDNN(
            input_dim=input_dim,
            shared_hidden_dims=hidden_dims,
            task_configs=task_configs,
            dropout=dropout
        )
    
    def forward(self, x):
        return self.model(x)


def get_model(model_type: str, **kwargs) -> nn.Module:
    """
    模型工厂函数
    
    Args:
        model_type: 模型类型 ('mlp', 'cnn', 'multitask', 'admet')
        **kwargs: 模型参数
        
    Returns:
        PyTorch模型
    """
    if model_type == 'mlp':
        return DrugPredictorMLP(**kwargs)
    elif model_type == 'cnn':
        return SMILESCNN(**kwargs)
    elif model_type == 'multitask':
        return MultiTaskDNN(**kwargs)
    elif model_type == 'admet':
        return ADMETPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 测试模型
    print("=" * 60)
    print("测试MLP模型")
    print("=" * 60)
    
    mlp = DrugPredictorMLP(
        input_dim=2048,
        hidden_dims=[512, 256, 128],
        output_dim=1,
        task_type='regression'
    )
    
    # 随机输入
    x = torch.randn(32, 2048)  # batch_size=32, input_dim=2048
    output = mlp(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in mlp.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("测试CNN模型")
    print("=" * 60)
    
    cnn = SMILESCNN(
        vocab_size=35,
        embedding_dim=128,
        num_filters=128,
        kernel_sizes=[3, 5, 7],
        output_dim=1,
        task_type='binary'
    )
    
    # SMILES编码输入
    x_smiles = torch.randint(0, 35, (32, 100))  # batch=32, seq_len=100
    output = cnn(x_smiles)
    print(f"输入形状: {x_smiles.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in cnn.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("测试多任务模型")
    print("=" * 60)
    
    task_configs = [
        {'name': 'solubility', 'output_dim': 1, 'task_type': 'regression'},
        {'name': 'toxicity', 'output_dim': 1, 'task_type': 'binary'},
        {'name': 'activity', 'output_dim': 3, 'task_type': 'multiclass'}
    ]
    
    multitask = MultiTaskDNN(
        input_dim=2048,
        shared_hidden_dims=[512, 256],
        task_configs=task_configs
    )
    
    x = torch.randn(32, 2048)
    outputs = multitask(x)
    
    print(f"输入形状: {x.shape}")
    for task_name, task_output in outputs.items():
        print(f"  {task_name}: {task_output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in multitask.parameters()):,}")
