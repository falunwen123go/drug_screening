"""
数据加载模块
功能：从MoleculeNet、ChEMBL等数据源加载药物数据集
数据将下载到本地data目录
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from typing import Tuple, List, Optional
import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
    import deepchem as dc
    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False
    print("Warning: DeepChem not installed. Some features may be unavailable.")


class DrugDataLoader:
    """药物数据加载器"""
    
    def __init__(self, data_dir: str = "./data/raw"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 原始数据存储目录
        """
        self.data_dir = os.path.abspath(data_dir)
        self.processed_dir = os.path.abspath("./data/processed")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def load_moleculenet_dataset(self, 
                                  dataset_name: str = 'BBBP',
                                  featurizer: str = 'ECFP',
                                  split: str = 'scaffold',
                                  save_local: bool = True) -> Tuple:
        """
        从MoleculeNet加载数据集，并保存到本地data目录
        
        Args:
            dataset_name: 数据集名称 (BBBP, Tox21, ESOL, BACE等)
            featurizer: 特征化方法 ('ECFP', 'GraphConv', 'Weave', 'Raw')
            split: 数据集分割方法 ('random', 'scaffold', 'stratified')
            save_local: 是否保存到本地data目录
            
        Returns:
            (train_dataset, valid_dataset, test_dataset, tasks)
        """
        if not DEEPCHEM_AVAILABLE:
            raise ImportError("DeepChem is required for MoleculeNet datasets. "
                            "Install it with: pip install deepchem")
        
        print(f"Loading {dataset_name} dataset from MoleculeNet...")
        print(f"Data will be saved to: {self.data_dir}")
        
        # 设置数据保存目录
        data_save_dir = os.path.join(self.data_dir, dataset_name.lower())
        os.makedirs(data_save_dir, exist_ok=True)
        
        # 加载数据集
        if dataset_name.upper() == 'BBBP':
            tasks, datasets, transformers = dc.molnet.load_bbbp(
                featurizer=featurizer, 
                splitter=split,
                reload=False,
                data_dir=data_save_dir  # 指定保存目录
            )
        elif dataset_name.upper() == 'TOX21':
            tasks, datasets, transformers = dc.molnet.load_tox21(
                featurizer=featurizer, 
                splitter=split,
                reload=False,
                data_dir=data_save_dir
            )
        elif dataset_name.upper() == 'ESOL':
            tasks, datasets, transformers = dc.molnet.load_delaney(
                featurizer=featurizer, 
                splitter=split,
                reload=False,
                data_dir=data_save_dir
            )
        elif dataset_name.upper() == 'BACE':
            tasks, datasets, transformers = dc.molnet.load_bace_classification(
                featurizer=featurizer, 
                splitter=split,
                reload=False,
                data_dir=data_save_dir
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        train_dataset, valid_dataset, test_dataset = datasets
        
        print(f"Dataset loaded successfully!")
        print(f"  Tasks: {tasks}")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(valid_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        
        # 保存为CSV到本地
        if save_local:
            self._save_dataset_to_csv(train_dataset, valid_dataset, test_dataset, 
                                      tasks, dataset_name)
        
        return train_dataset, valid_dataset, test_dataset, tasks
    
    def _save_dataset_to_csv(self, train_data, valid_data, test_data, 
                              tasks, dataset_name: str):
        """将数据集保存为CSV文件到本地data/processed目录"""
        save_dir = os.path.join(self.processed_dir, dataset_name.lower())
        os.makedirs(save_dir, exist_ok=True)
        
        for name, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            df = pd.DataFrame({
                'smiles': data.ids,
                **{f'label_{i}': data.y[:, i] for i in range(data.y.shape[1])}
            })
            csv_path = os.path.join(save_dir, f'{name}.csv')
            df.to_csv(csv_path, index=False)
            print(f"  Saved {name} set to: {csv_path}")
    
    def load_local_dataset(self, dataset_name: str = 'BBBP') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        从本地data/processed目录加载已保存的数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            (train_df, valid_df, test_df)
        """
        load_dir = os.path.join(self.processed_dir, dataset_name.lower())
        
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Local dataset not found: {load_dir}. "
                                   "Please run load_moleculenet_dataset first.")
        
        train_df = pd.read_csv(os.path.join(load_dir, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(load_dir, 'valid.csv'))
        test_df = pd.read_csv(os.path.join(load_dir, 'test.csv'))
        
        print(f"Loaded local dataset from: {load_dir}")
        print(f"  Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
        
        return train_df, valid_df, test_df
    
    def load_csv_dataset(self, 
                        csv_path: str,
                        smiles_col: str = 'smiles',
                        label_col: Optional[str] = None) -> pd.DataFrame:
        """
        从CSV文件加载自定义数据集
        
        Args:
            csv_path: CSV文件路径
            smiles_col: SMILES列名
            label_col: 标签列名（可选）
            
        Returns:
            DataFrame包含清洗后的数据
        """
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # 验证SMILES列存在
        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found in CSV")
        
        print(f"Original dataset size: {len(df)}")
        
        # 数据清洗
        df = self.clean_dataset(df, smiles_col)
        
        print(f"Cleaned dataset size: {len(df)}")
        
        return df
    
    def clean_dataset(self, 
                     df: pd.DataFrame, 
                     smiles_col: str = 'smiles') -> pd.DataFrame:
        """
        清洗数据集：去除重复、验证SMILES有效性
        
        Args:
            df: 输入DataFrame
            smiles_col: SMILES列名
            
        Returns:
            清洗后的DataFrame
        """
        original_size = len(df)
        
        # 去除缺失值
        df = df.dropna(subset=[smiles_col])
        print(f"  Removed {original_size - len(df)} rows with missing SMILES")
        
        # 去除重复SMILES
        original_size = len(df)
        df = df.drop_duplicates(subset=[smiles_col])
        print(f"  Removed {original_size - len(df)} duplicate SMILES")
        
        # 验证SMILES有效性并标准化
        valid_indices = []
        canonical_smiles = []
        
        for idx, smiles in enumerate(df[smiles_col]):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # 转换为规范SMILES
                canonical = Chem.MolToSmiles(mol)
                canonical_smiles.append(canonical)
                valid_indices.append(idx)
        
        df = df.iloc[valid_indices].copy()
        df[smiles_col] = canonical_smiles
        
        print(f"  Removed {original_size - len(df)} invalid SMILES")
        
        return df.reset_index(drop=True)
    
    def get_sample_molecules(self, n_samples: int = 10) -> List[str]:
        """
        生成示例分子SMILES（用于测试）
        
        Args:
            n_samples: 样本数量
            
        Returns:
            SMILES列表
        """
        sample_smiles = [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # 阿司匹林
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # 布洛芬
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # 咖啡因
            'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F',  # Celecoxib
            'CC(C)(C)NCC(COC1=CC=C(C=C1)COCCOC(C)C)O',  # Atenolol
            'C1=CC=C(C=C1)C2=CC=CC=C2',  # 联苯
            'c1ccc2c(c1)ccc3c2ccc4c3cccc4',  # 并四苯
            'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O',  # S-布洛芬
            'COc1ccc(cc1)C(=O)N',  # 对甲氧基苯甲酰胺
            'C1=CC=C(C=C1)O'  # 苯酚
        ]
        
        return sample_smiles[:n_samples]


if __name__ == "__main__":
    # 测试数据加载器
    loader = DrugDataLoader()
    
    # 测试MoleculeNet加载（如果安装了DeepChem）
    if DEEPCHEM_AVAILABLE:
        try:
            train, valid, test, tasks = loader.load_moleculenet_dataset('BBBP')
            print("\n✓ MoleculeNet BBBP dataset loaded successfully")
        except Exception as e:
            print(f"\n✗ Error loading MoleculeNet: {e}")
    
    # 测试示例分子
    samples = loader.get_sample_molecules(5)
    print(f"\n示例分子SMILES:")
    for i, smiles in enumerate(samples, 1):
        print(f"  {i}. {smiles}")
