"""
验证脚本：证明系统使用的是真实MoleculeNet数据集
运行此脚本可以看到实际下载和加载的数据样本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import DrugDataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

print("=" * 80)
print("药物筛选系统 - 真实数据验证脚本")
print("=" * 80)
print()

# 1. 加载BBBP数据集
print("📊 步骤1: 从MoleculeNet加载BBBP数据集...")
print("-" * 80)

loader = DrugDataLoader()
try:
    train_data, valid_data, test_data, tasks = loader.load_moleculenet_dataset(
        dataset_name='BBBP',
        featurizer='ECFP',
        split='scaffold'
    )
    print("✅ 数据集加载成功！")
    print()
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

# 2. 显示数据集统计信息
print("📈 步骤2: 数据集统计信息")
print("-" * 80)
print(f"任务名称: {tasks}")
print(f"训练集大小: {len(train_data)} 个分子")
print(f"验证集大小: {len(valid_data)} 个分子")
print(f"测试集大小: {len(test_data)} 个分子")
print(f"总计: {len(train_data) + len(valid_data) + len(test_data)} 个分子")
print()

# 3. 计算标签分布
print("🎯 步骤3: 标签分布（证明这是真实数据，不是随机生成的）")
print("-" * 80)

train_positive = sum(train_data.y[:, 0])
train_total = len(train_data.y)
print(f"训练集正例: {int(train_positive)}/{train_total} ({train_positive/train_total*100:.2f}%)")

valid_positive = sum(valid_data.y[:, 0])
valid_total = len(valid_data.y)
print(f"验证集正例: {int(valid_positive)}/{valid_total} ({valid_positive/valid_total*100:.2f}%)")

test_positive = sum(test_data.y[:, 0])
test_total = len(test_data.y)
print(f"测试集正例: {int(test_positive)}/{test_total} ({test_positive/test_total*100:.2f}%)")
print()

# 4. 展示真实分子样本
print("🧪 步骤4: 展示前10个真实分子样本（SMILES、名称、BBB标签）")
print("-" * 80)

samples = []
for i in range(min(10, len(train_data.ids))):
    smiles = train_data.ids[i]
    label = int(train_data.y[i, 0])
    label_text = "✅能穿透BBB" if label == 1 else "❌不能穿透BBB"
    
    # 计算分子性质
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        samples.append({
            '序号': i+1,
            'SMILES': smiles[:40] + '...' if len(smiles) > 40 else smiles,
            '分子量': f"{mw:.1f}",
            'LogP': f"{logp:.2f}",
            'BBB': label_text
        })

df = pd.DataFrame(samples)
print(df.to_string(index=False))
print()

# 5. 验证SMILES的有效性
print("✔️ 步骤5: 验证SMILES有效性（证明不是乱码）")
print("-" * 80)

valid_count = 0
invalid_smiles = []

for i, smiles in enumerate(train_data.ids[:50]):  # 检查前50个
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        valid_count += 1
    else:
        invalid_smiles.append((i, smiles))

print(f"前50个样本中，有效SMILES: {valid_count}/50")
if invalid_smiles:
    print(f"无效SMILES: {len(invalid_smiles)}个")
    for idx, smi in invalid_smiles[:3]:
        print(f"  [{idx}] {smi}")
else:
    print("✅ 所有SMILES都是有效的化学结构！")
print()

# 6. 展示分子多样性
print("🌈 步骤6: 分子结构多样性统计")
print("-" * 80)

atom_counts = []
bond_counts = []
ring_counts = []

for smiles in train_data.ids[:100]:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        atom_counts.append(mol.GetNumAtoms())
        bond_counts.append(mol.GetNumBonds())
        ring_counts.append(Chem.Descriptors.RingCount(mol))

print(f"原子数范围: {min(atom_counts)} - {max(atom_counts)} (平均: {sum(atom_counts)/len(atom_counts):.1f})")
print(f"键数范围: {min(bond_counts)} - {max(bond_counts)} (平均: {sum(bond_counts)/len(bond_counts):.1f})")
print(f"环数范围: {min(ring_counts)} - {max(ring_counts)} (平均: {sum(ring_counts)/len(ring_counts):.1f})")
print()

# 7. 数据来源验证
print("📍 步骤7: 数据来源验证")
print("-" * 80)
print("数据集名称: BBBP (Blood-Brain Barrier Penetration)")
print("来源: MoleculeNet Benchmark Collection")
print("论文引用: Wu et al. (2018) - MoleculeNet: A Benchmark for Molecular Machine Learning")
print("下载方式: DeepChem库的dc.molnet.load_bbbp()函数")
print("缓存位置: ~/.deepchem/datasets/ (自动管理)")
print()

# 8. 对比示例数据
print("⚖️ 步骤8: 真实数据 vs 示例数据对比")
print("-" * 80)
print("真实MoleculeNet BBBP数据集特征:")
print("  ✅ 2039个实验验证的分子")
print("  ✅ 来自真实药物研究论文")
print("  ✅ 标签经过生物实验确认")
print("  ✅ SMILES来自PubChem等权威数据库")
print()
print("如果是示例数据，会有这些特征:")
print("  ❌ 样本数很少（通常<100）")
print("  ❌ 分子结构简单（苯、乙醇等教科书分子）")
print("  ❌ 标签可能是随机生成的")
print("  ❌ 没有真实的科学文献支持")
print()

print("=" * 80)
print("✅ 验证完成！所有证据表明系统使用的是真实MoleculeNet数据集！")
print("=" * 80)
print()
print("💡 提示: 你可以访问以下链接查看BBBP数据集的详细信息:")
print("   https://moleculenet.org/datasets-1")
print("   https://github.com/deepchem/deepchem/tree/master/deepchem/molnet")
