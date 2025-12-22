"""
分子特征提取模块
功能：将SMILES转换为分子指纹、描述符等数值表示
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator  # 新API，避免弃用警告
from typing import List, Union, Optional
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class MolecularFeaturizer:
    """分子特征化工具类"""
    
    def __init__(self, fingerprint_size: int = 2048, radius: int = 2, fp_type: str = 'ecfp', fp_size: int = None):
        """
        初始化特征化器
        
        Args:
            fingerprint_size: 指纹向量长度
            radius: Morgan指纹半径（等同于ECFP4中的2）
            fp_type: 指纹类型 ('ecfp', 'morgan', 'maccs')
            fp_size: 指纹大小（兼容参数）
        """
        self.fingerprint_size = fp_size if fp_size else fingerprint_size
        self.radius = radius
        self.fp_type = fp_type
        
        # 使用新的MorganGenerator API（避免弃用警告）
        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.fingerprint_size
        )
        
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        将SMILES字符串转换为RDKit分子对象
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            RDKit分子对象，如果无效则返回None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except:
            return None
    
    def get_morgan_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """
        生成Morgan指纹（ECFP - Extended Connectivity Fingerprints）
        使用新的rdFingerprintGenerator API避免弃用警告
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            指纹向量（numpy array），失败返回None
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        try:
            # 使用新的MorganGenerator API（避免DEPRECATION警告）
            fp = self.morgan_generator.GetFingerprintAsNumPy(mol)
            return fp.astype(np.int8)
        except Exception:
            # 兼容旧版本RDKit
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, 
                radius=self.radius,
                nBits=self.fingerprint_size
            )
            arr = np.zeros((self.fingerprint_size,), dtype=np.int8)
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
    
    def extract_features(self, smiles: str) -> Optional[np.ndarray]:
        """
        提取分子特征（兼容方法）
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            特征向量
        """
        return self.get_morgan_fingerprint(smiles)
    
    def get_maccs_keys(self, smiles: str) -> Optional[np.ndarray]:
        """
        生成MACCS Keys指纹（166位固定长度）
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            MACCS指纹向量
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        from rdkit.Chem import MACCSkeys
        fp = MACCSkeys.GenMACCSKeys(mol)
        
        arr = np.zeros((167,), dtype=np.int8)  # MACCS keys是167位
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        
        return arr
    
    def get_molecular_descriptors(self, smiles: str) -> Optional[dict]:
        """
        计算分子描述符（物理化学性质）
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            描述符字典
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        descriptors = {
            # 基本性质
            'MolecularWeight': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),  # 脂水分配系数
            'TPSA': Descriptors.TPSA(mol),  # 极性表面积
            'NumHDonors': Descriptors.NumHDonors(mol),  # 氢键供体数
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),  # 氢键受体数
            
            # 原子和键信息
            'NumAtoms': mol.GetNumAtoms(),
            'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            
            # 其他重要描述符
            'MolMR': Descriptors.MolMR(mol),  # 摩尔折射率
            'BertzCT': Descriptors.BertzCT(mol),  # 分子复杂度
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),  # 脂肪环数量
        }
        
        return descriptors
    
    def get_descriptor_vector(self, smiles: str) -> Optional[np.ndarray]:
        """
        将分子描述符转换为向量形式
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            描述符向量
        """
        desc_dict = self.get_molecular_descriptors(smiles)
        if desc_dict is None:
            return None
        
        return np.array(list(desc_dict.values()), dtype=np.float32)
    
    def batch_featurize(self, 
                       smiles_list: List[str], 
                       method: str = 'morgan') -> np.ndarray:
        """
        批量特征化分子
        
        Args:
            smiles_list: SMILES字符串列表
            method: 特征化方法 ('morgan', 'maccs', 'descriptors')
            
        Returns:
            特征矩阵 (n_samples, n_features)
        """
        features = []
        
        for smiles in smiles_list:
            if method == 'morgan':
                feat = self.get_morgan_fingerprint(smiles)
            elif method == 'maccs':
                feat = self.get_maccs_keys(smiles)
            elif method == 'descriptors':
                feat = self.get_descriptor_vector(smiles)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if feat is not None:
                features.append(feat)
            else:
                # 如果特征化失败，使用零向量
                if method == 'morgan':
                    features.append(np.zeros(self.fingerprint_size))
                elif method == 'maccs':
                    features.append(np.zeros(167))
                elif method == 'descriptors':
                    features.append(np.zeros(12))  # 12个描述符
        
        return np.array(features)
    
    def smiles_to_one_hot(self, smiles: str, max_length: int = 100) -> np.ndarray:
        """
        将SMILES字符串转换为one-hot编码（用于CNN/RNN）
        
        Args:
            smiles: SMILES字符串
            max_length: 最大长度（截断或填充）
            
        Returns:
            one-hot编码矩阵 (max_length, vocab_size)
        """
        # SMILES字符集
        charset = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 
                  'c', 'n', 'o', 's',
                  '(', ')', '[', ']', '=', '#', '@', '+', '-',
                  '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  'H', ' ']
        
        char_to_idx = {char: idx for idx, char in enumerate(charset)}
        vocab_size = len(charset)
        
        # 初始化one-hot矩阵
        one_hot = np.zeros((max_length, vocab_size))
        
        # 填充one-hot编码
        for i, char in enumerate(smiles[:max_length]):
            if char in char_to_idx:
                one_hot[i, char_to_idx[char]] = 1
        
        return one_hot


class FeatureScaler:
    """特征标准化工具"""
    
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X: np.ndarray):
        """
        计算训练集的均值和标准差
        
        Args:
            X: 特征矩阵
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # 避免除以零
        self.std[self.std == 0] = 1.0
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        标准化特征
        
        Args:
            X: 特征矩阵
            
        Returns:
            标准化后的特征矩阵
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        拟合并转换
        """
        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":
    # 测试特征提取器
    featurizer = MolecularFeaturizer()
    
    # 测试分子
    aspirin = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    
    print("测试分子: 阿司匹林")
    print(f"SMILES: {aspirin}\n")
    
    # Morgan指纹
    morgan_fp = featurizer.get_morgan_fingerprint(aspirin)
    print(f"Morgan指纹维度: {morgan_fp.shape}")
    print(f"非零位数: {np.sum(morgan_fp)}\n")
    
    # 分子描述符
    descriptors = featurizer.get_molecular_descriptors(aspirin)
    print("分子描述符:")
    for name, value in descriptors.items():
        print(f"  {name}: {value:.2f}")
    
    # 批量特征化
    smiles_list = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
    ]
    
    print(f"\n批量特征化 {len(smiles_list)} 个分子...")
    features = featurizer.batch_featurize(smiles_list, method='morgan')
    print(f"特征矩阵形状: {features.shape}")
