"""
模型管理服务
提供模型的选择、推荐等功能
"""
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class ModelService:
    """模型服务类"""
    
    def __init__(self):
        pass
    
    def list_recommended_models(self) -> Dict[str, List[Dict[str, str]]]:
        """
        列出推荐的预训练模型
        
        Returns:
            按类型分组的推荐模型列表
        """
        return {
            "embedding_models": [
                {
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                    "description": "轻量级，适合快速训练和推理",
                    "size": "22M parameters"
                },
                {
                    "name": "sentence-transformers/all-MiniLM-L12-v2", 
                    "description": "中等大小，平衡性能和速度",
                    "size": "33M parameters"
                },
                {
                    "name": "sentence-transformers/all-mpnet-base-v2",
                    "description": "高性能，适合对质量要求高的场景",
                    "size": "109M parameters"
                },
                {
                    "name": "distilbert-base-uncased",
                    "description": "DistilBERT基础模型，训练速度快",
                    "size": "66M parameters"
                },
                {
                    "name": "bert-base-uncased",
                    "description": "经典BERT基础模型",
                    "size": "110M parameters"
                }
            ],
            "reranker_models": [
                {
                    "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "description": "轻量级reranker，适合快速训练",
                    "size": "22M parameters"
                },
                {
                    "name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                    "description": "中等大小reranker，平衡性能和速度",
                    "size": "33M parameters"
                },
                {
                    "name": "distilroberta-base",
                    "description": "DistilRoBERTa基础模型，适合reranker训练",
                    "size": "82M parameters"
                },
                {
                    "name": "roberta-base",
                    "description": "RoBERTa基础模型，性能较好",
                    "size": "125M parameters"
                }
            ],
            "chinese_models": [
                {
                    "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "description": "多语言模型，支持中文",
                    "size": "118M parameters"
                },
                {
                    "name": "bert-base-chinese",
                    "description": "中文BERT基础模型",
                    "size": "102M parameters"
                }
            ]
        }
    
    def search_models(self, query: str, training_type: Optional[str] = None) -> Dict[str, Any]:
        """
        搜索模型
        
        Args:
            query: 搜索关键词
            training_type: 训练类型过滤 (embedding/reranker)
            
        Returns:
            搜索结果
        """
        try:
            all_models = self.list_recommended_models()
            results = []
            
            query_lower = query.lower()
            
            # 搜索所有模型类别
            for category, models in all_models.items():
                # 如果指定了训练类型，进行过滤
                if training_type:
                    if training_type == "embedding" and "reranker" in category:
                        continue
                    elif training_type == "reranker" and "embedding" in category:
                        continue
                
                for model in models:
                    # 在模型名称和描述中搜索
                    if (query_lower in model["name"].lower() or 
                        query_lower in model["description"].lower()):
                        results.append({
                            **model,
                            "category": category
                        })
            
            return {
                "success": True,
                "message": f"找到 {len(results)} 个匹配的模型",
                "data": {
                    "query": query,
                    "training_type_filter": training_type,
                    "results": results
                }
            }
            
        except Exception as e:
            logger.error(f"搜索模型失败: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"搜索模型失败: {str(e)}",
                "data": None
            }
    
    def _get_recommended_training_types(self, model_name_or_path: str) -> List[str]:
        """根据模型特征推荐训练类型"""
        model_name_lower = model_name_or_path.lower()
        
        # 根据模型名称判断推荐的训练类型
        if "cross-encoder" in model_name_lower or "reranker" in model_name_lower:
            return ["reranker"]
        elif "sentence-transformer" in model_name_lower:
            return ["embedding", "reranker"]
        else:
            # 通用模型，两种类型都支持
            return ["embedding", "reranker"]

# 全局模型服务实例
model_service = ModelService()