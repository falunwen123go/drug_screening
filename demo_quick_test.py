"""
快速演示脚本 - 数据加载和特征提取
"""
import sys
sys.path.append('.')

from data.data_loader import DrugDataLoader
from features.molecular_features import MolecularFeaturizer

print("="*60)
print("示例1: 测试数据加载")
print("="*60)

# 创建加载器
loader = DrugDataLoader(data_dir='./data/raw')

# 获取示例分子
sample_smiles = loader.get_sample_molecules(5)
print("\n示例分子SMILES:")
for i, smiles in enumerate(sample_smiles, 1):
    print(f"  {i}. {smiles}")

print("\n" + "="*60)
print("示例2: 测试特征提取")
print("="*60)

# 创建特征提取器
featurizer = MolecularFeaturizer(fingerprint_size=2048)

print("\n提取Morgan指纹和分子性质...")
for i, smiles in enumerate(sample_smiles[:3], 1):
    print(f"\n分子 {i}:")
    print(f"  SMILES: {smiles}")
    
    # Morgan指纹
    fp = featurizer.get_morgan_fingerprint(smiles)
    print(f"  指纹维度: {fp.shape}")
    print(f"  非零位数: {fp.sum()}")
    
    # 分子描述符
    desc = featurizer.get_molecular_descriptors(smiles)
    print(f"  分子量: {desc['MolecularWeight']:.2f} Da")
    print(f"  LogP: {desc['LogP']:.2f}")
    print(f"  TPSA: {desc['TPSA']:.2f} Ų")
    print(f"  氢键供体: {desc['NumHDonors']}")
    print(f"  氢键受体: {desc['NumHAcceptors']}")

print("\n" + "="*60)
print("✓ 测试完成！所有模块工作正常。")
print("="*60)
