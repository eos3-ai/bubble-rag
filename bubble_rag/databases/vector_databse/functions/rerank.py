import os
from typing import List, Optional

import requests

from pymilvus.model.base import BaseRerankFunction, RerankResult


class OpenaiRerankFunction(BaseRerankFunction):
    base_url: str
    api_key: str
    model_name: str

    def __init__(self, base_url, model_name: str = "rerank-model", api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Accept-Encoding": "identity"}
        )
        self.model_name = model_name

    def __call__(self, query: str, documents: List[str], top_k: int = 5) -> List[RerankResult]:
        resp = self._session.post(  # type: ignore[assignment]
            self.base_url + "/v1/rerank",
            json={
                "query": query,
                "documents": documents,
                "model": self.model_name,
                "top_n": top_k,
            },
        ).json()
        if "results" not in resp:
            raise RuntimeError(resp["detail"])

        results = []
        for res in resp["results"]:
            results.append(
                RerankResult(
                    text=documents[res["index"]], score=res["relevance_score"], index=res["index"]
                )
            )
        return results
