"""
Streamlit Web应用
提供交互式药物筛选界面
支持GPU加速（如果可用）
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io
import torch
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_extraction import MolecularFeaturizer
from models.drug_models import DrugPredictorMLP
from inference.predictor import DrugPredictor, DrugScreener


# 检测设备
def get_device():
    """检测并返回可用设备"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


# 页面配置
st.set_page_config(
    page_title="药物筛选系统",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """加载训练好的BBBP模型，优先使用GPU"""
    try:
        device = get_device()
        
        # 使用与训练时相同的模型配置
        model = DrugPredictorMLP(input_dim=1024, hidden_dims=[512, 256, 128], output_dim=1)
        
        # 加载训练好的模型权重
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  'saved_models', 'bbbp_model.pth')
        if os.path.exists(model_path):
            # 加载到对应设备
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            if device == 'cuda':
                st.sidebar.success(f"✅ BBBP模型已加载 (GPU: {torch.cuda.get_device_name(0)})")
            else:
                st.sidebar.success("✅ BBBP模型已加载 (CPU)")
        else:
            st.sidebar.warning("⚠️ 未找到预训练模型，使用随机权重")
            model = model.to(device)
        
        featurizer = MolecularFeaturizer(fingerprint_size=1024, radius=2)
        predictor = DrugPredictor(model, featurizer, device=device)
        return predictor, device
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, 'cpu'


def draw_molecule(smiles: str, size=(400, 400)):
    """绘制分子结构"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        return img
    except Exception:
        return None


def main():
    # 标题
    st.markdown('<div class="main-header">💊 基于大数据分析的药物筛选系统</div>', 
                unsafe_allow_html=True)
    
    # 侧边栏
    st.sidebar.title("⚙️ 系统设置")
    
    # 显示设备信息
    device = get_device()
    if device == 'cuda':
        st.sidebar.info(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.info("🖥️ 使用CPU运行")
    
    # 模式选择
    mode = st.sidebar.selectbox(
        "选择模式",
        ["单分子预测", "批量筛选", "数据集探索", "系统说明"]
    )
    
    # 加载预测器
    result = load_model()
    if result[0] is None:
        predictor = None
        current_device = 'cpu'
    else:
        predictor, current_device = result

    
    # ==================== 单分子预测模式 ====================
    if mode == "单分子预测":
        st.markdown('<div class="sub-header">🔬 单分子预测</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("输入分子")
            
            # SMILES输入
            smiles_input = st.text_input(
                "输入SMILES字符串",
                value="CC(=O)OC1=CC=CC=C1C(=O)O",
                help="例如：阿司匹林的SMILES"
            )
            
            # 示例分子
            example_molecules = {
                "阿司匹林": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "布洛芬": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "咖啡因": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "对乙酰氨基酚": "CC(=O)NC1=CC=C(C=C1)O"
            }
            
            selected_example = st.selectbox("或选择示例分子", ["自定义"] + list(example_molecules.keys()))
            
            if selected_example != "自定义":
                smiles_input = example_molecules[selected_example]
                st.info(f"已选择: {selected_example}")
            
            # 预测按钮
            if st.button("🚀 开始预测", type="primary"):
                if predictor is None:
                    st.error("预测器未加载，请检查模型文件")
                else:
                    with st.spinner("正在预测..."):
                        try:
                            # 预测
                            result = predictor.predict_with_properties(smiles_input)
                            
                            # 显示结果
                            st.success("✅ 预测完成！")
                            
                            # 预测分数
                            score = result['prediction']
                            st.metric("预测得分", f"{score:.4f}" if score is not None else "N/A")
                            
                            # 分子性质
                            st.subheader("分子性质")
                            if result['properties']:
                                props = result['properties']
                                
                                # 显示关键性质
                                prop_col1, prop_col2, prop_col3 = st.columns(3)
                                with prop_col1:
                                    st.metric("分子量", f"{props['MolecularWeight']:.2f} Da")
                                    st.metric("氢键供体", int(props['NumHDonors']))
                                with prop_col2:
                                    st.metric("LogP", f"{props['LogP']:.2f}")
                                    st.metric("氢键受体", int(props['NumHAcceptors']))
                                with prop_col3:
                                    st.metric("TPSA", f"{props['TPSA']:.2f} Ų")
                                    st.metric("旋转键数", int(props['NumRotatableBonds']))
                                
                                # 所有性质表格
                                with st.expander("查看所有性质"):
                                    props_df = pd.DataFrame([props]).T
                                    props_df.columns = ['值']
                                    st.dataframe(props_df)
                            
                            # Lipinski五规则检查
                            st.subheader("药物相似性评估")
                            if result['properties']:
                                props = result['properties']
                                
                                checks = {
                                    "分子量 ≤ 500 Da": props['MolecularWeight'] <= 500,
                                    "LogP ≤ 5": props['LogP'] <= 5,
                                    "氢键供体 ≤ 5": props['NumHDonors'] <= 5,
                                    "氢键受体 ≤ 10": props['NumHAcceptors'] <= 10
                                }
                                
                                for rule, passed in checks.items():
                                    if passed:
                                        st.success(f"✅ {rule}")
                                    else:
                                        st.error(f"❌ {rule}")
                                
                                if all(checks.values()):
                                    st.info("🎉 该分子符合Lipinski五规则！")
                        
                        except Exception as e:
                            st.error(f"预测失败: {e}")
        
        with col2:
            st.subheader("分子结构")
            
            # 绘制分子
            try:
                mol_img = draw_molecule(smiles_input, size=(500, 500))
                if mol_img:
                    st.image(mol_img)
                else:
                    st.error("无法解析SMILES字符串")
            except Exception as e:
                st.error(f"绘制分子失败: {e}")
    
    # ==================== 批量筛选模式 ====================
    elif mode == "批量筛选":
        st.markdown('<div class="sub-header">📊 批量筛选</div>', unsafe_allow_html=True)
        
        st.info("上传包含SMILES的CSV文件进行批量筛选")
        
        # 文件上传
        uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])
        
        if uploaded_file is not None:
            # 读取文件
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ 已加载 {len(df)} 个化合物")
            
            # 显示前几行
            st.subheader("数据预览")
            st.dataframe(df.head(10))
            
            # 选择SMILES列
            smiles_col = st.selectbox("选择SMILES列", df.columns)
            
            # 筛选参数
            col1, col2, col3 = st.columns(3)
            with col1:
                top_k = st.number_input("Top-K候选数", min_value=1, max_value=1000, value=10)
            with col2:
                ascending = st.checkbox("分数越小越好", value=False)
            with col3:
                apply_lipinski = st.checkbox("应用Lipinski过滤", value=True)
            
            # 开始筛选
            if st.button("🔍 开始筛选", type="primary"):
                if predictor is None:
                    st.error("预测器未加载")
                else:
                    with st.spinner("正在筛选..."):
                        try:
                            screener = DrugScreener(predictor)
                            smiles_list = df[smiles_col].tolist()
                            
                            # 筛选
                            results = screener.screen_library(smiles_list, top_k=top_k, ascending=ascending)
                            
                            # Lipinski过滤
                            if apply_lipinski:
                                results = screener.filter_by_lipinski(results)
                            
                            st.success(f"✅ 筛选完成！找到 {len(results)} 个候选化合物")
                            
                            # 显示结果
                            st.subheader("筛选结果")
                            st.dataframe(results)
                            
                            # 下载按钮
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="📥 下载结果CSV",
                                data=csv,
                                file_name="screening_results.csv",
                                mime="text/csv"
                            )
                            
                            # 可视化Top分子
                            st.subheader("Top候选分子结构")
                            top_smiles = results['smiles'].head(min(6, len(results))).tolist()
                            
                            cols = st.columns(3)
                            for i, smiles in enumerate(top_smiles):
                                with cols[i % 3]:
                                    img = draw_molecule(smiles, size=(300, 300))
                                    if img:
                                        st.image(img, caption=f"Rank {i+1}")
                                        st.caption(f"Score: {results.iloc[i]['score']:.4f}")
                        
                        except Exception as e:
                            st.error(f"筛选失败: {e}")
        else:
            st.info("请上传CSV文件开始批量筛选")
            
            # 示例CSV下载
            example_data = {
                'smiles': [
                    'CC(=O)OC1=CC=CC=C1C(=O)O',
                    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
                    'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
                ],
                'name': ['Aspirin', 'Caffeine', 'Ibuprofen']
            }
            example_df = pd.DataFrame(example_data)
            csv = example_df.to_csv(index=False)
            st.download_button(
                label="📥 下载示例CSV",
                data=csv,
                file_name="example_molecules.csv",
                mime="text/csv"
            )
    
    # ==================== 数据集探索模式 ====================
    elif mode == "数据集探索":
        st.markdown('<div class="sub-header">🗂️ 数据集探索</div>', unsafe_allow_html=True)
        
        st.info("探索常用的药物数据集")
        
        datasets_info = {
            "BBBP": {
                "名称": "血脑屏障穿透性数据集",
                "样本数": "2,039",
                "任务": "二分类",
                "描述": "预测分子是否能穿透血脑屏障"
            },
            "Tox21": {
                "名称": "毒性预测数据集",
                "样本数": "7,831",
                "任务": "多任务分类",
                "描述": "预测12种毒性指标"
            },
            "ESOL": {
                "名称": "水溶解度数据集",
                "样本数": "1,128",
                "任务": "回归",
                "描述": "预测分子的水溶解度（LogS）"
            },
            "BACE": {
                "名称": "β-分泌酶抑制剂数据集",
                "样本数": "1,513",
                "任务": "二分类/回归",
                "描述": "预测BACE-1抑制活性"
            }
        }
        
        for dataset_name, info in datasets_info.items():
            with st.expander(f"📁 {dataset_name} - {info['名称']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("样本数量", info['样本数'])
                    st.metric("任务类型", info['任务'])
                with col2:
                    st.write("**描述:**")
                    st.write(info['描述'])
    
    # ==================== 系统说明模式 ====================
    elif mode == "系统说明":
        st.markdown('<div class="sub-header">📖 系统说明</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 系统简介
        
        本系统是一个基于深度学习的药物筛选平台，集成了多种功能模块：
        
        #### 🔧 核心功能
        - **单分子预测**: 输入SMILES预测分子性质
        - **批量筛选**: 从大量化合物中筛选Top-K候选
        - **分子性质计算**: 自动计算200+种分子描述符
        - **Lipinski规则检查**: 评估药物相似性
        
        #### 🧬 技术架构
        - **特征提取**: Morgan指纹、MACCS keys、分子描述符
        - **深度学习模型**: MLP、CNN、Multi-task DNN
        - **可视化**: RDKit分子绘制、性质分析图表
        
        #### 📊 支持的数据集
        - MoleculeNet (BBBP, Tox21, ESOL, BACE等)
        - ChEMBL
        - 自定义CSV数据
        
        #### 💡 使用提示
        1. SMILES格式要规范（可使用RDKit验证）
        2. 批量筛选建议样本数<100,000
        3. 模型需要先训练后才能使用
        
        #### 📚 参考资料
        - [RDKit文档](https://www.rdkit.org/docs/)
        - [DeepChem教程](https://deepchem.io/tutorials/)
        - [MoleculeNet论文](https://arxiv.org/abs/1703.00564)
        """)
        
        st.markdown("---")
        st.info("💻 开发者: 课程设计项目 | 🔬 基于PyTorch + RDKit + Streamlit")


if __name__ == "__main__":
    main()
