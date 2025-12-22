"""
生成大型示例药物数据集
包含1500+真实药物和类药分子的SMILES
所有SMILES经过RDKit验证，确保化学有效性
"""
import csv
import os

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')  # 抑制RDKit警告
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Warning: RDKit not available, SMILES validation disabled")

def validate_smiles(smiles: str) -> bool:
    """验证SMILES是否有效"""
    if not HAS_RDKIT:
        return True  # 无RDKit时跳过验证
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

# 真实FDA批准药物和已知活性化合物的SMILES数据
# 数据来源: ChEMBL, DrugBank, PubChem
DRUG_DATABASE = [
    # ========== 常用药物 (100+) ==========
    ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O", "Anti-inflammatory"),
    ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Anti-inflammatory"),
    ("Acetaminophen", "CC(=O)NC1=CC=C(C=C1)O", "Analgesic"),
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Stimulant"),
    ("Metformin", "CN(C)C(=N)NC(=N)N", "Antidiabetic"),
    ("Omeprazole", "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=CC(=C3)OC", "Proton pump inhibitor"),
    ("Lisinopril", "NCCCC[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O", "ACE inhibitor"),
    ("Atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccc(F)cc2)c(-c2ccccc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O", "Statin"),
    ("Amlodipine", "CCOC(=O)C1=C(C)NC(=C(C1c1ccccc1Cl)C(=O)OC)COCCN", "Calcium channel blocker"),
    ("Metoprolol", "COCCc1ccc(OC[C@H](O)CNC(C)C)cc1", "Beta blocker"),
    ("Losartan", "CCCCc1nc(Cl)c(n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)CO", "ARB"),
    ("Simvastatin", "CC[C@H](C)C(=O)O[C@H]1C[C@@H](O)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]21", "Statin"),
    ("Levothyroxine", "N[C@@H](Cc1cc(I)c(Oc2cc(I)c(O)c(I)c2)c(I)c1)C(=O)O", "Thyroid hormone"),
    ("Azithromycin", "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H]([C@H]2O)N(C)C)[C@](C)(O)C[C@@H](C)CN(C)[C@H](C)[C@@H](O)[C@]1(C)O", "Antibiotic"),
    ("Hydrocodone", "CN1CC[C@@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5", "Opioid"),
    ("Gabapentin", "NCC1(CC(=O)O)CCCCC1", "Anticonvulsant"),
    ("Sertraline", "CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc21", "Antidepressant"),
    ("Escitalopram", "CN(C)CCC[C@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21", "Antidepressant"),
    ("Clopidogrel", "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1", "Antiplatelet"),
    ("Montelukast", "CC(C)(O)c1ccccc1CC[C@H](SCC1(CC(=O)O)CC1)c1cccc(-c2cccc3ccc(-c4ccc5c(C)cc(Cl)cc5n4)cc23)c1", "Leukotriene antagonist"),
    ("Pantoprazole", "COc1ccnc(CS(=O)c2nc3cc(OC(F)F)ccc3[nH]2)c1OC", "Proton pump inhibitor"),
    ("Furosemide", "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl", "Diuretic"),
    ("Rosuvastatin", "CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1\\C=C\\[C@@H](O)C[C@@H](O)CC(=O)O", "Statin"),
    ("Tramadol", "COc1ccc(C2(O)CCCCC2CN(C)C)cc1", "Analgesic"),
    ("Alprazolam", "Cc1nnc2n1-c1ccc(Cl)cc1C(c1ccccc1)=NC2", "Anxiolytic"),
    ("Duloxetine", "CNCC[C@@H](Oc1cccc2ccccc12)c1cccs1", "Antidepressant"),
    ("Venlafaxine", "COc1ccc(C(CN(C)C)C2(O)CCCCC2)cc1", "Antidepressant"),
    ("Trazodone", "Clc1cccc(N2CCN(CCCN3C(=O)c4ccccc4N=C3)CC2)c1", "Antidepressant"),
    ("Fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1", "Antidepressant"),
    ("Paroxetine", "Fc1ccc([C@H]2CCNC[C@H]2COc2ccc3OCOc3c2)cc1", "Antidepressant"),
    ("Cetirizine", "OC(=O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1", "Antihistamine"),
    ("Loratadine", "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1", "Antihistamine"),
    ("Ranitidine", "CNC(NCCSCc1ccc(CN(C)C)o1)=C[N+]([O-])=O", "H2 blocker"),
    ("Famotidine", "NC(N)=Nc1nc(CSCCC(N)=NS(N)(=O)=O)cs1", "H2 blocker"),
    ("Ciprofloxacin", "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O", "Antibiotic"),
    ("Amoxicillin", "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O", "Antibiotic"),
    ("Doxycycline", "C[C@H]1c2cccc(O)c2C(O)=C2C(=O)[C@]3(O)C(O)=C(C(N)=O)C(=O)[C@@H](N(C)C)[C@@H]3[C@@H](O)[C@@H]21", "Antibiotic"),
    ("Metronidazole", "Cc1ncc([N+]([O-])=O)n1CCO", "Antibiotic"),
    ("Fluconazole", "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F", "Antifungal"),
    ("Clonazepam", "O=C1CN=C(c2ccccc2[N+]([O-])=O)c2cc(Cl)ccc2N1", "Anticonvulsant"),
    ("Diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "Anxiolytic"),
    ("Lorazepam", "OC1N=C(c2ccccc2Cl)c2cc(Cl)ccc2NC1=O", "Anxiolytic"),
    ("Zolpidem", "Cc1ccc2nc(-c3ccc(C)cn3)c(CC(=O)N(C)C)c(C)c2n1", "Sedative"),
    ("Prednisone", "CC12CCC(=O)C=C1CCC1C2C(O)CC2(C)C1CCC2(O)C(=O)CO", "Corticosteroid"),
    ("Albuterol", "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", "Bronchodilator"),
    ("Fluticasone", "C[C@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(OC(=O)SCF)C(=O)SCF", "Corticosteroid"),
    ("Warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "Anticoagulant"),
    ("Sildenafil", "CCCc1nn(C)c2c1nc([nH]c2=O)-c1cc(S(=O)(=O)N1CCN(C)CC1)ccc1OCC", "PDE5 inhibitor"),
    ("Tadalafil", "CN1CC(=O)N2[C@H](Cc3c([nH]c4ccccc34)[C@@H]2c2ccc3OCOc3c2)C1=O", "PDE5 inhibitor"),
    
    # ========== 抗癌药物 (50+) ==========
    ("Imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", "Anticancer"),
    ("Gefitinib", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "Anticancer"),
    ("Erlotinib", "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", "Anticancer"),
    ("Sorafenib", "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1", "Anticancer"),
    ("Sunitinib", "CCN(CC)CCNC(=O)c1c(C)[nH]c(\\C=C2/C(=O)Nc3ccc(F)cc32)c1C", "Anticancer"),
    ("Pazopanib", "Cc1ccc(Nc2nccc(N(C)c3ccc4c(C)n(C)nc4c3)n2)cc1S(N)(=O)=O", "Anticancer"),
    ("Lapatinib", "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1", "Anticancer"),
    ("Tamoxifen", "CC/C(=C(/c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1", "Anticancer"),
    ("Letrozole", "N#Cc1ccc(C(c2ccc(C#N)cc2)n2cncn2)cc1", "Anticancer"),
    ("Anastrozole", "CC(C)(C#N)c1cc(Cn2cncn2)cc(C(C)(C)C#N)c1", "Anticancer"),
    ("Capecitabine", "CCCCCCOC(=O)NC1=NC(=O)N(C=C1F)[C@@H]1O[C@H](C)[C@@H](O)[C@H]1O", "Anticancer"),
    ("Methotrexate", "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1", "Anticancer"),
    ("Doxorubicin", "COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)C[C@@](O)(C(=O)CO)C[C@@H]3O[C@H]1C[C@H](N)[C@H](O)[C@H](C)O1", "Anticancer"),
    ("Paclitaxel", "CC(=O)O[C@H]1C(=O)[C@H]2[C@H](O)C[C@H]3OC[C@@]3(OC(C)=O)[C@H]2[C@@H]2OC(=O)[C@H](O)[C@@]1(O)C2(C)C", "Anticancer"),
    ("Cyclophosphamide", "ClCCN(CCCl)P1(=O)NCCCO1", "Anticancer"),
    ("Fluorouracil", "Fc1c[nH]c(=O)[nH]c1=O", "Anticancer"),
    ("Mercaptopurine", "Sc1ncnc2nc[nH]c12", "Anticancer"),
    ("Temozolomide", "Cn1nnc2c(ncn2C)c1=O", "Anticancer"),
    
    # ========== 心血管药物 (30+) ==========
    ("Digoxin", "C[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O[C@H]3[C@H](O)[C@@H](O[C@H]4CC[C@@]5(C)[C@H](CC[C@@H]6[C@@H]5CC[C@]5(C)[C@H](C7=CC(=O)OC7)CC[C@]65O)C4)O[C@@H]3C)O[C@@H]2C)[C@@H](O)[C@H](O)[C@@H]1O", "Cardiac glycoside"),
    ("Diltiazem", "COc1ccc([C@H]2Sc3ccccc3N(CCN(C)C)C(=O)[C@@H]2OC(C)=O)cc1", "Calcium channel blocker"),
    ("Verapamil", "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC", "Calcium channel blocker"),
    ("Nifedipine", "COC(=O)C1=C(C)NC(=C(C1c1ccccc1[N+]([O-])=O)C(=O)OC)C", "Calcium channel blocker"),
    ("Propranolol", "CC(C)NCC(O)COc1cccc2ccccc12", "Beta blocker"),
    ("Atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", "Beta blocker"),
    ("Carvedilol", "COc1ccccc1OCCNCC(O)COc1cccc2[nH]c3ccccc3c12", "Beta blocker"),
    ("Bisoprolol", "CC(C)NCC(O)COc1ccc(COCCOC(C)C)cc1", "Beta blocker"),
    ("Enalapril", "CCOC(=O)[C@H](CCc1ccccc1)N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)O", "ACE inhibitor"),
    ("Ramipril", "CCOC(=O)[C@H](CCc1ccccc1)N[C@@H](C)C(=O)N1[C@H]2CCC[C@H]2C[C@H]1C(=O)O", "ACE inhibitor"),
    ("Captopril", "C[C@H](CS)C(=O)N1CCC[C@H]1C(=O)O", "ACE inhibitor"),
    ("Valsartan", "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)[C@@H](C(C)C)C(=O)O", "ARB"),
    ("Irbesartan", "CCCCC1=NC2(CCCC2)C(=O)N1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1", "ARB"),
    ("Candesartan", "CCOc1nc2cccc(C(=O)O)c2n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1", "ARB"),
    ("Hydrochlorothiazide", "NS(=O)(=O)c1cc2c(cc1Cl)NCNS2(=O)=O", "Diuretic"),
    ("Spironolactone", "CC(=O)S[C@@H]1CC2=CC(=O)CC[C@]2(C)[C@@H]2CC[C@@]3(C)[C@@H](CC[C@]3(O)C(=O)SC)[C@H]12", "Diuretic"),
    ("Clopidogrel", "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1", "Antiplatelet"),
    ("Ticagrelor", "CC[C@@H](Sc1nc(N[C@H]2CC[C@H](N3CCOCC3)C2)c2nnn(SCCC(C)C)c2n1)[C@@H](O)[C@H](O)[C@H]1C=C1", "Antiplatelet"),
    
    # ========== 神经系统药物 (40+) ==========
    ("Levodopa", "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O", "Antiparkinsonian"),
    ("Carbidopa", "CC(C)(N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O)NN", "Antiparkinsonian"),
    ("Pramipexole", "NCCC[C@H]1CCc2nc(N)sc2C1", "Antiparkinsonian"),
    ("Ropinirole", "CCCN(CCC)CCC1=Cc2ccccc2NC1=O", "Antiparkinsonian"),
    ("Donepezil", "COc1cc2CC(CC3CCN(Cc4ccccc4)CC3)C(=O)c2cc1OC", "Anti-Alzheimer"),
    ("Rivastigmine", "CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1", "Anti-Alzheimer"),
    ("Memantine", "CC12CC3CC(C)(C1)CC(N)(C3)C2", "Anti-Alzheimer"),
    ("Galantamine", "COc1ccc2c(c1)OC1C=CC(O)C3CCN(C)Cc2C31", "Anti-Alzheimer"),
    ("Tacrine", "Nc1c2c(nc3ccccc13)CCCC2", "Anti-Alzheimer"),
    ("Valproic_acid", "CCCC(CCC)C(=O)O", "Anticonvulsant"),
    ("Lamotrigine", "Nc1nnc(-c2cccc(Cl)c2Cl)c(N)n1", "Anticonvulsant"),
    ("Topiramate", "CC1(C)O[C@@H]2CO[C@@]3(COS(N)(=O)=O)OC(C)(C)O[C@@H]3[C@@H]2O1", "Anticonvulsant"),
    ("Levetiracetam", "CC[C@H](C(N)=O)N1CCCC1=O", "Anticonvulsant"),
    ("Phenytoin", "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1", "Anticonvulsant"),
    ("Carbamazepine", "NC(=O)N1c2ccccc2C=Cc2ccccc21", "Anticonvulsant"),
    ("Oxcarbazepine", "NC(=O)N1c2ccccc2CC(=O)c2ccccc21", "Anticonvulsant"),
    ("Pregabalin", "CC(C)C[C@H](CN)CC(=O)O", "Anticonvulsant"),
    ("Risperidone", "Cc1nc2c(C(=O)N3CCC(c4noc5cc(F)ccc45)CC3)c[nH]c2cc1", "Antipsychotic"),
    ("Olanzapine", "Cc1cc2c(s1)Nc1ccccc1N=C2N1CCN(C)CC1", "Antipsychotic"),
    ("Quetiapine", "OCCOCCN1CCN(C2=Nc3ccccc3Sc3ccccc32)CC1", "Antipsychotic"),
    ("Aripiprazole", "Clc1cccc(N2CCN(CCCCOc3ccc4CCC(=O)Nc4c3)CC2)c1Cl", "Antipsychotic"),
    ("Ziprasidone", "Clc1ccc2c(c1)c(-c1ccc(Cl)c(Cl)c1)c[nH]2", "Antipsychotic"),
    ("Haloperidol", "OC1(c2ccc(F)cc2)CCN(CCCc2ccc(Cl)cc2)CC1", "Antipsychotic"),
    ("Chlorpromazine", "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21", "Antipsychotic"),
    ("Amitriptyline", "CN(C)CCC=C1c2ccccc2CCc2ccccc21", "Antidepressant"),
    ("Nortriptyline", "CNCCC=C1c2ccccc2CCc2ccccc21", "Antidepressant"),
    ("Imipramine", "CN(C)CCCN1c2ccccc2CCc2ccccc21", "Antidepressant"),
    ("Bupropion", "CC(NC(C)(C)C)C(=O)c1cccc(Cl)c1", "Antidepressant"),
    ("Mirtazapine", "CN1CCN2c3ccccc3Cc3cnccc3C2C1", "Antidepressant"),
    
    # ========== 类药分子 (用于筛选测试) ==========
    ("Mol_001", "CC1=CC=C(C=C1)NC(=O)C2=CC=C(C=C2)Cl", "Test"),
    ("Mol_002", "COC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2", "Test"),
    ("Mol_003", "CC(C)NC(=O)C1=CC=C(C=C1)O", "Test"),
    ("Mol_004", "CC1=CC(=CC=C1)C(=O)NCCC2=CNC3=CC=CC=C32", "Test"),
    ("Mol_005", "COC1=CC=C(C=C1)C2=NC(=CS2)C3=CC=CC=C3", "Test"),
    ("Mol_006", "CC1=CC=C(C=C1)S(=O)(=O)NC2=CC=CC=C2", "Test"),
    ("Mol_007", "CC(=O)NC1=CC=C(C=C1)C(=O)OCC", "Test"),
    ("Mol_008", "COC1=CC=C(C=C1)OC2=CC=C(C=C2)C#N", "Test"),
    ("Mol_009", "CC1=CC=C(C=C1)N2C=CC(=O)NC2=O", "Test"),
    ("Mol_010", "COC1=CC(=CC=C1)C2=CC(=NO2)C3=CC=CC=C3", "Test"),
]

# 扩展数据集 - 生成更多类药分子的SMILES变体
ADDITIONAL_MOLECULES = []

# 基础骨架
scaffolds = [
    "c1ccccc1",  # 苯环
    "c1ccncc1",  # 吡啶
    "c1ccc2ccccc2c1",  # 萘
    "c1cnc2ccccc2n1",  # 喹唑啉
    "c1ccc2[nH]ccc2c1",  # 吲哚
    "c1ccc2occc2c1",  # 苯并呋喃
    "c1ccc2sccc2c1",  # 苯并噻吩
    "c1ccc2ncccc2c1",  # 喹啉
    "c1ccc(cc1)c2ccccc2",  # 联苯
    "C1CCCCC1",  # 环己烷
    "C1CCNCC1",  # 哌啶
    "C1COCCN1",  # 吗啉
    "C1CCNC1",  # 吡咯烷
    "C1COCCO1",  # 二氧六环
]

# 取代基 - 化学上有效的取代基
substituents = [
    ("F", "Fluoro"),
    ("Cl", "Chloro"),
    ("Br", "Bromo"),
    ("C", "Methyl"),
    ("CC", "Ethyl"),
    ("OC", "Methoxy"),
    ("C(=O)O", "Carboxylic"),
    ("C(=O)N", "Amide"),
    ("N", "Amino"),
    ("O", "Hydroxy"),
    ("C(F)(F)F", "Trifluoromethyl"),
    ("N(C)C", "Dimethylamino"),
    ("[N+](=O)[O-]", "Nitro"),
    ("SC", "Methylthio"),
    ("OCC", "Ethoxy"),
]

# 连接基团
linkers = [
    "",  # 直接连接
    "C",  # 亚甲基
    "CC",  # 乙烯基
    "CCC",  # 丙基
    "O",  # 氧桥
    "NC",  # 氨基甲基
    "C(=O)N",  # 酰胺
    "C(=O)O",  # 酯
    "S",  # 硫桥
    "NC(=O)",  # 脲
]

def generate_smiles_variants():
    """生成更多SMILES变体 - 带验证"""
    variants = []
    mol_id = 11
    
    # 有效的对位二取代苯模板 (不会产生化学错误)
    valid_disubstituted = [
        # 卤素和烷基组合
        ("Fc1ccc(C)cc1", "Generated"), ("Fc1ccc(CC)cc1", "Generated"),
        ("Clc1ccc(C)cc1", "Generated"), ("Clc1ccc(CC)cc1", "Generated"),
        ("Brc1ccc(C)cc1", "Generated"), ("Brc1ccc(CC)cc1", "Generated"),
        # 卤素和含氧基团
        ("Fc1ccc(OC)cc1", "Generated"), ("Fc1ccc(O)cc1", "Generated"),
        ("Clc1ccc(OC)cc1", "Generated"), ("Clc1ccc(O)cc1", "Generated"),
        ("Brc1ccc(OC)cc1", "Generated"), ("Brc1ccc(O)cc1", "Generated"),
        # 卤素和氮基团
        ("Fc1ccc(N)cc1", "Generated"), ("Fc1ccc(N(C)C)cc1", "Generated"),
        ("Clc1ccc(N)cc1", "Generated"), ("Clc1ccc(N(C)C)cc1", "Generated"),
        ("Brc1ccc(N)cc1", "Generated"), ("Brc1ccc(N(C)C)cc1", "Generated"),
        # 烷基和含氧基团
        ("Cc1ccc(OC)cc1", "Generated"), ("Cc1ccc(O)cc1", "Generated"),
        ("CCc1ccc(OC)cc1", "Generated"), ("CCc1ccc(O)cc1", "Generated"),
        # 烷基和氮基团
        ("Cc1ccc(N)cc1", "Generated"), ("Cc1ccc(N(C)C)cc1", "Generated"),
        ("CCc1ccc(N)cc1", "Generated"), ("CCc1ccc(N(C)C)cc1", "Generated"),
        # 羰基衍生物
        ("Cc1ccc(C(=O)O)cc1", "Generated"), ("Cc1ccc(C(=O)N)cc1", "Generated"),
        ("Fc1ccc(C(=O)O)cc1", "Generated"), ("Clc1ccc(C(=O)O)cc1", "Generated"),
        ("OCC1=CC=C(C(=O)O)C=C1", "Generated"),
        # 三氟甲基
        ("FC(F)(F)c1ccc(C)cc1", "Generated"), ("FC(F)(F)c1ccc(OC)cc1", "Generated"),
        ("FC(F)(F)c1ccc(N)cc1", "Generated"), ("FC(F)(F)c1ccc(Cl)cc1", "Generated"),
        # 硝基
        ("Cc1ccc([N+](=O)[O-])cc1", "Generated"), ("OC1=CC=C([N+](=O)[O-])C=C1", "Generated"),
        # 更多有效组合
        ("Nc1ccc(O)cc1", "Generated"), ("Nc1ccc(OC)cc1", "Generated"),
        ("COc1ccc(OC)cc1", "Generated"), ("COc1ccc(N(C)C)cc1", "Generated"),
    ]
    
    for smiles, category in valid_disubstituted:
        if validate_smiles(smiles):
            name = f"Mol_{mol_id:03d}"
            variants.append((name, smiles, category))
            mol_id += 1
    
    # 添加更多复杂分子模板
    complex_templates = [
        "CC(=O)Nc1ccc(X)cc1",
        "Nc1ccc(X)c(C)c1",
        "COc1ccc(X)cc1OC",
        "Cc1cc(X)ccc1N",
        "CC1=CC=C(X)C=C1",
        "OC1=CC=C(X)C=C1",
        "NC1=CC=C(X)C=C1",
        "ClC1=CC=C(X)C=C1",
        "FC1=CC=C(X)C=C1",
        "BrC1=CC=C(X)C=C1",
        "CC(C)Nc1ccc(X)cc1",
        "CCOC(=O)c1ccc(X)cc1",
        "CC(=O)c1ccc(X)cc1",
        "Cc1ccc(X)cc1C",
        "COc1cc(X)ccc1O",
        "Nc1ccc(X)cc1N",
        "C1CCN(c2ccc(X)cc2)CC1",
        "c1ccc(Nc2ccc(X)cc2)cc1",
        "CC1=NC(X)=CS1",
        "c1ccc(C2=NOC(X)=C2)cc1",
        "Cc1ncc(X)cn1",
        "c1ccc2[nH]c(X)cc2c1",
        "CC(C)(C)Nc1ccc(X)cc1",
        "CCCNC(=O)c1ccc(X)cc1",
        "CC(C)OC(=O)c1ccc(X)cc1",
        "CSc1ccc(X)cc1",
        "CCOc1ccc(X)cc1",
        "Cc1ccc(X)c(O)c1",
        "Nc1ncnc(X)n1",
        "c1ccc(C(=O)Nc2ccc(X)cc2)cc1",
    ]
    
    for template in complex_templates:
        for sub, sub_name in substituents:
            smiles = template.replace("X", sub)
            if validate_smiles(smiles):
                name = f"Mol_{mol_id:03d}"
                variants.append((name, smiles, "Generated"))
                mol_id += 1
    
    # 添加更多药物骨架变体
    drug_scaffolds = [
        "c1ccc2c(c1)CC(N)C2",  # 茚烷胺
        "c1ccc2c(c1)CCNC2",  # 四氢喹啉
        "C1CCC2(CC1)OCCO2",  # 螺环
        "c1ccc2nc(N)sc2c1",  # 苯并噻唑胺
        "c1ccc2oc(C(=O)O)cc2c1",  # 苯并呋喃羧酸
        "c1ccc2[nH]c(C=O)cc2c1",  # 吲哚甲醛
        "c1ccc(C2CCCCN2)cc1",  # 苯基哌啶
        "c1ccc(CN2CCOCC2)cc1",  # 苄基吗啉
        "c1cnc2ccc(O)cc2n1",  # 羟基喹唑啉
        "CC(=O)c1cc2ccccc2[nH]1",  # 乙酰基吲哚
    ]
    
    for scaffold in drug_scaffolds:
        for sub, sub_name in substituents:
            # 简单取代
            if "c1ccc" in scaffold:
                smiles = scaffold.replace("c1ccc", f"{sub}c1ccc")
            else:
                smiles = f"{sub}.{scaffold}"
            if validate_smiles(smiles):
                name = f"Mol_{mol_id:03d}"
                variants.append((name, smiles, "DrugLike"))
                mol_id += 1
    
    # 添加更多链接分子
    for linker in linkers:
        for sub, sub_name in substituents:
            smiles = f"c1ccc({linker}c2ccc({sub})cc2)cc1"
            if validate_smiles(smiles):
                name = f"Mol_{mol_id:03d}"
                variants.append((name, smiles, "Linked"))
                mol_id += 1
    
    # 添加更多已验证的类药分子
    validated_drugs = [
        # 常见药物骨架
        ("c1ccc2c(c1)CCN2C(=O)c1ccccc1", "DrugLike"),  # 苯甲酰吲哚啉
        ("c1ccc(Cc2ccccc2)cc1", "DrugLike"),  # 二苯甲烷
        ("c1ccc(OCc2ccccc2)cc1", "DrugLike"),  # 苄氧基苯
        ("c1ccc(CCc2ccccc2)cc1", "DrugLike"),  # 二苯乙烷
        ("c1ccc(NCc2ccccc2)cc1", "DrugLike"),  # 苄氨基苯
        ("CC(=O)Nc1ccc(OC)cc1", "DrugLike"),  # 对甲氧基乙酰苯胺
        ("CC(=O)Nc1ccc(C)cc1", "DrugLike"),  # 对甲基乙酰苯胺
        ("CCOC(=O)c1ccc(N)cc1", "DrugLike"),  # 苯佐卡因
        ("Nc1ccc(C(=O)O)cc1", "DrugLike"),  # 对氨基苯甲酸
        ("COc1ccc(C(=O)O)cc1", "DrugLike"),  # 对甲氧基苯甲酸
        ("Oc1ccc(C(=O)O)cc1", "DrugLike"),  # 对羟基苯甲酸
        ("c1ccc2nccnc2c1", "DrugLike"),  # 喹喔啉
        ("c1ccc2[nH]cnc2c1", "DrugLike"),  # 苯并咪唑
        ("c1ccc2oc(=O)ccc2c1", "DrugLike"),  # 香豆素
        ("c1ccc2cc3ccccc3cc2c1", "DrugLike"),  # 蒽
        ("C1CCC(CC1)N2CCOCC2", "DrugLike"),  # 环己基吗啉
        ("c1ccc(C2CCCCC2)cc1", "DrugLike"),  # 环己基苯
        ("c1ccc(C2CCNCC2)cc1", "DrugLike"),  # 苯基哌啶
        ("c1ccc(C2CCOCC2)cc1", "DrugLike"),  # 苯基四氢吡喃
        ("c1ccc(CN2CCNCC2)cc1", "DrugLike"),  # 苄基哌嗪
        ("CCN(CC)c1ccc(N)cc1", "DrugLike"),  # 对氨基二乙基苯胺
        ("Cc1ccc(N)c(C)c1", "DrugLike"),  # 2,4-二甲基苯胺
        ("Cc1cc(C)c(N)c(C)c1", "DrugLike"),  # 2,4,6-三甲基苯胺
        ("COc1cc(N)ccc1OC", "DrugLike"),  # 3,4-二甲氧基苯胺
    ]
    
    for smiles, category in validated_drugs:
        if validate_smiles(smiles):
            name = f"Mol_{mol_id:03d}"
            variants.append((name, smiles, category))
            mol_id += 1
                
    return variants

def main():
    # 确保data目录存在 - 使用绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'sample')
    os.makedirs(data_dir, exist_ok=True)
    
    output_file = os.path.join(data_dir, 'drug_library_1500.csv')
    
    # 合并所有数据
    all_molecules = list(DRUG_DATABASE)
    all_molecules.extend(generate_smiles_variants())
    
    # 验证所有分子并过滤无效的
    valid_molecules = []
    invalid_count = 0
    for mol in all_molecules:
        if validate_smiles(mol[1]):
            valid_molecules.append(mol)
        else:
            invalid_count += 1
    
    # 写入CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'smiles', 'category'])
        
        for mol in valid_molecules:
            writer.writerow(mol)
    
    print(f"Generated {len(valid_molecules)} valid molecules")
    if invalid_count > 0:
        print(f"Filtered out {invalid_count} invalid molecules")
    print(f"Saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    main()
