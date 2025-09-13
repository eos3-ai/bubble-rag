from loguru import logger

from bubble_rag.databases.vector_databse import milvus_client
from pymilvus import DataType, Function, FunctionType


def create_coll(coll_name, embedding_dim):
    logger.info("============================ create_coll ============================")
    logger.info(f"create_coll {coll_name} {embedding_dim}")
    logger.info("============================ create_coll ============================")
    schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name='id', datatype=DataType.VARCHAR, is_primary=True, max_length=32)
    schema.add_field(field_name='doc_content', datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
    schema.add_field(field_name='doc_title', datatype=DataType.VARCHAR, max_length=8192)
    schema.add_field(field_name='doc_ctn_dense', datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    schema.add_field(field_name='doc_ctn_sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name='doc_file_id', datatype=DataType.VARCHAR, max_length=8192)
    schema.add_field(field_name='embedding_model_id', datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name='create_time', datatype=DataType.INT64, default_value=0)
    schema.add_field(field_name='update_time', datatype=DataType.INT64, default_value=0)

    bm25_function = Function(
        # Function name
        name="text_bm25_emb",
        # Name of the VARCHAR field containing raw text data
        input_field_names=["doc_content"],
        # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
        output_field_names=["doc_ctn_sparse"],
        function_type=FunctionType.BM25
    )
    schema.add_function(bm25_function)

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(index_name=f"{coll_name}_id_idx_dft", field_name='id')
    index_params.add_index(index_name=f"{coll_name}_doc_ctn_dense_idx_dft", field_name='doc_ctn_dense', index_type='IVF_FLAT', metric_type='IP')
    index_params.add_index(index_name=f"{coll_name}_doc_ctn_sparse_idx_dft", field_name='doc_ctn_sparse', index_type='SPARSE_INVERTED_INDEX', metric_type='BM25',
                           params={"inverted_index_algo": "DAAT_MAXSCORE"})

    collection_result = milvus_client.create_collection(collection_name=coll_name, schema=schema, index_params=index_params)
    logger.info(f"db coll {coll_name} create success")
    return collection_result


def delete_knowledge_base(coll_name):
    milvus_client.drop_collection(collection_name=coll_name)
    return True

