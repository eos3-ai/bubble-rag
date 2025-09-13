import os
from collections import defaultdict
from typing import List, Optional

import numpy as np
from loguru import logger

from pymilvus.model.base import BaseEmbeddingFunction
from pymilvus.model.utils import import_openai


class OpenaiEmbeddingFunction(BaseEmbeddingFunction):
    base_url: str
    api_key: str
    model_name: str

    def __init__(
            self,
            model_name: str = "embedding-model",
            api_key: Optional[str] = "qwertyuiopsadfgfhj",
            base_url: Optional[str] = None,
            dimensions: Optional[int] = None,
            **kwargs,
    ):
        import_openai()
        from openai import OpenAI

        base_url = base_url.removesuffix("/")
        if not base_url.endswith("v1"):
            base_url += "/v1"

        self._openai_model_meta_info = defaultdict(dict)

        self._model_config = dict({"api_key": api_key if api_key else "testkey", "base_url": base_url}, **kwargs)
        additional_encode_config = {}
        self._encode_config = {"model": model_name, **additional_encode_config}
        self.model_name = model_name
        self.client = OpenAI(**self._model_config)

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._encode(queries)

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._encode(documents)

    @property
    def dim(self):
        return self._openai_model_meta_info[self.model_name]["dim"]

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._encode(texts)

    def _encode_query(self, query: str) -> np.array:
        return self._encode(query)[0]

    def _encode_document(self, document: str) -> np.array:
        return self._encode(document)[0]

    def _call_openai_api(self, texts: List[str]):
        results = self.client.embeddings.create(input=texts, **self._encode_config).data
        return [np.array(data.embedding) for data in results]

    def _encode(self, texts: List[str]):
        return self._call_openai_api(texts)
