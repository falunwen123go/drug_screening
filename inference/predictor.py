"""
药物预测和筛选模块
提供批量预测、Top-K筛选等功能
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Draw
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.molecular_features import MolecularFeaturizer


class DrugPredictor:
    """药物预测器"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 featurizer: MolecularFeaturizer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 feature_method: str = 'morgan'):
        """
        Args:
            model: 训练好的PyTorch模型
            featurizer: 特征提取器
            device: 计算设备
            feature_method: 特征化方法 ('morgan', 'maccs', 'descriptors')
        """
        self.model = model.to(device)
        self.model.eval()
        self.featurizer = featurizer
        self.device = device
        self.feature_method = feature_method
        
    def predict_single(self, smiles: str) -> float:
        """
        预测单个分子
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            预测分数
        """
        # 特征化
        if self.feature_method == 'morgan':
            features = self.featurizer.get_morgan_fingerprint(smiles)
        elif self.feature_method == 'maccs':
            features = self.featurizer.get_maccs_keys(smiles)
        elif self.feature_method == 'descriptors':
            features = self.featurizer.get_descriptor_vector(smiles)
        else:
            raise ValueError(f"Unknown feature method: {self.feature_method}")
        
        if features is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # 预测
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            prediction = self.model(features_tensor)
            
        return prediction.cpu().item()
    
    def predict_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        批量预测
        
        Args:
            smiles_list: SMILES列表
            
        Returns:
            预测分数数组
        """
        # 批量特征化
        features = self.featurizer.batch_featurize(
            smiles_list, 
            method=self.feature_method
        )
        
        # 批量预测
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            predictions = self.model(features_tensor)
            
        return predictions.cpu().numpy().flatten()
    
    def predict_with_properties(self, smiles: str) -> Dict:
        """
        预测并返回分子性质
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            包含预测和性质的字典
        """
        # 预测
        try:
            prediction = self.predict_single(smiles)
        except:
            prediction = None
        
        # 计算分子性质
        properties = self.featurizer.get_molecular_descriptors(smiles)
        
        result = {
            'smiles': smiles,
            'prediction': prediction,
            'properties': properties
        }
        
        return result


class DrugScreener:
    """药物虚拟筛选器"""
    
    def __init__(self, predictor: DrugPredictor):
        """
        Args:
            predictor: 药物预测器
        """
        self.predictor = predictor
    
    def screen_library(self,
                      smiles_list: List[str],
                      top_k: int = 100,
                      ascending: bool = False) -> pd.DataFrame:
        """
        筛选化合物库
        
        Args:
            smiles_list: SMILES列表
            top_k: 返回Top-K个化合物
            ascending: True表示分数越小越好，False表示越大越好
            
        Returns:
            筛选结果DataFrame
        """
        print(f"开始筛选 {len(smiles_list)} 个化合物...")
        
        # 批量预测
        predictions = self.predictor.predict_batch(smiles_list)
        
        # 创建DataFrame
        results = pd.DataFrame({
            'smiles': smiles_list,
            'score': predictions
        })
        
        # 排序
        results = results.sort_values('score', ascending=ascending)
        
        # 提取Top-K
        top_candidates = results.head(top_k).copy()
        
        # 添加分子性质
        properties_list = []
        for smiles in top_candidates['smiles']:
            props = self.predictor.featurizer.get_molecular_descriptors(smiles)
            properties_list.append(props)
        
        # 合并性质到DataFrame
        props_df = pd.DataFrame(properties_list)
        top_candidates = pd.concat([top_candidates.reset_index(drop=True), props_df], axis=1)
        
        print(f"筛选完成！返回Top {top_k} 候选化合物")
        
        return top_candidates
    
    def filter_by_lipinski(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据Lipinski五规则过滤（药物相似性规则）
        
        Lipinski五规则：
        - 分子量 <= 500 Da
        - LogP <= 5
        - 氢键供体 <= 5
        - 氢键受体 <= 10
        
        Args:
            df: 包含分子性质的DataFrame
            
        Returns:
            过滤后的DataFrame
        """
        print("应用Lipinski五规则过滤...")
        original_count = len(df)
        
        filtered = df[
            (df['MolecularWeight'] <= 500) &
            (df['LogP'] <= 5) &
            (df['NumHDonors'] <= 5) &
            (df['NumHAcceptors'] <= 10)
        ].copy()
        
        print(f"  原始: {original_count} 个化合物")
        print(f"  过滤后: {len(filtered)} 个化合物")
        print(f"  通过率: {len(filtered)/original_count*100:.1f}%")
        
        return filtered
    
    def save_molecules_image(self,
                            smiles_list: List[str],
                            save_path: str,
                            mols_per_row: int = 5,
                            img_size: Tuple[int, int] = (300, 300)):
        """
        保存分子结构图
        
        Args:
            smiles_list: SMILES列表
            save_path: 保存路径
            mols_per_row: 每行显示的分子数
            img_size: 每个分子的图片大小
        """
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        mols = [m for m in mols if m is not None]  # 过滤无效分子
        
        if len(mols) == 0:
            print("没有有效的分子用于绘制")
            return
        
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            subImgSize=img_size,
            legends=[f"Mol {i+1}" for i in range(len(mols))]
        )
        
        img.save(save_path)
        print(f"分子结构图已保存到: {save_path}")
    
    def generate_report(self,
                       results: pd.DataFrame,
                       save_path: str = './screening_report.csv'):
        """
        生成筛选报告
        
        Args:
            results: 筛选结果DataFrame
            save_path: 保存路径
        """
        # 添加排名
        results['rank'] = range(1, len(results) + 1)
        
        # 重新排列列顺序
        cols = ['rank', 'smiles', 'score'] + [c for c in results.columns if c not in ['rank', 'smiles', 'score']]
        results = results[cols]
        
        # 保存
        results.to_csv(save_path, index=False)
        print(f"筛选报告已保存到: {save_path}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("筛选报告摘要")
        print("="*60)
        print(f"总候选数: {len(results)}")
        print(f"平均得分: {results['score'].mean():.4f}")
        print(f"最高得分: {results['score'].max():.4f}")
        print(f"最低得分: {results['score'].min():.4f}")
        print("="*60)
        
        # 显示Top 5
        print("\nTop 5 候选化合物:")
        print(results.head()[['rank', 'smiles', 'score', 'MolecularWeight', 'LogP']].to_string(index=False))


if __name__ == "__main__":
    # 测试预测器（需要先训练模型）
    print("药物预测器模块已加载")
    print("使用示例:")
    print("""
    # 1. 加载模型
    from models.drug_models import DrugPredictorMLP
    model = DrugPredictorMLP(input_dim=2048)
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 2. 创建预测器
    featurizer = MolecularFeaturizer()
    predictor = DrugPredictor(model, featurizer)
    
    # 3. 预测单个分子
    score = predictor.predict_single('CC(=O)OC1=CC=CC=C1C(=O)O')
    
    # 4. 批量筛选
    screener = DrugScreener(predictor)
    results = screener.screen_library(smiles_list, top_k=100)
    """)
