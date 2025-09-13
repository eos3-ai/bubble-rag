from bubble_rag.databases.vector_databse.functions.embedding import OpenaiEmbeddingFunction
from bubble_rag.databases.vector_databse.functions.rerank import OpenaiRerankFunction
from bubble_rag.databases.relation_database import get_session
from bubble_rag.entity.relational.models import ModelConfig


def get_embedding_ef(model_id: str) -> OpenaiEmbeddingFunction:
    with next(get_session()) as session:
        model_conf: ModelConfig = session.get(ModelConfig, model_id)
        return OpenaiEmbeddingFunction(
            base_url=model_conf.model_base_url,
            api_key=model_conf.model_api_key,
            model_name=model_conf.model_name,
            dimensions=1024,
        )


def get_rerank_ef(model_id: str) -> OpenaiRerankFunction:
    with next(get_session()) as session:
        model_conf: ModelConfig = session.get(ModelConfig, model_id)
        return OpenaiRerankFunction(
            base_url=model_conf.model_base_url,
            api_key=model_conf.model_api_key,
            model_name=model_conf.model_name
        )
