# 模型配置文件
# 定义不同模型的元数据和显示信息

MODEL_CONFIG = {
    "bbbp_model.pth": {
        "name": "BBBP",
        "full_name": "Blood-Brain Barrier Permeability",
        "cn_name": "血脑屏障穿透性",
        "description": "预测药物分子穿透血脑屏障的能力",
        "task_type": "binary",  # binary classification
        "score_meaning": "high_better",  # 分数越高越好
        "high_label": "高概率穿透血脑屏障",
        "low_label": "低概率穿透血脑屏障",
        "threshold": 0.5,
        "unit": "",
        "icon": "🧠"
    },
    "esol_model.pth": {
        "name": "ESOL",
        "full_name": "Estimated SOLubility",
        "cn_name": "水溶性预测",
        "description": "预测药物分子在水中的溶解度 (log mol/L)",
        "task_type": "regression",  # regression task
        "score_meaning": "value",  # 分数是实际预测值
        "high_label": "高水溶性",
        "low_label": "低水溶性",
        "threshold": -3.0,  # log mol/L, -3以上算较好溶解性
        "unit": "log mol/L",
        "icon": "💧"
    }
}

def get_model_config(model_name: str) -> dict:
    """获取模型配置，如果模型不在配置中则返回默认配置"""
    if model_name in MODEL_CONFIG:
        return MODEL_CONFIG[model_name]
    
    # 默认配置
    return {
        "name": model_name.replace("_model.pth", "").upper(),
        "full_name": model_name,
        "cn_name": "未知模型",
        "description": "未配置的预测模型",
        "task_type": "unknown",
        "score_meaning": "unknown",
        "high_label": "高分数",
        "low_label": "低分数",
        "threshold": 0.5,
        "unit": "",
        "icon": "🔬"
    }
