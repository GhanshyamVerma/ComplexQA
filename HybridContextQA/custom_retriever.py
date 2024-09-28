# import QueryBundle
from llama_index.core import QueryBundle

# import NodeWithScore
from llama_index.core.schema import NodeWithScore

# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KGTableRetriever,
)

from typing import List

# Option 2: return a string (we use a raw LLM call for illustration)
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

from llama_index.core.settings import (
    Settings,
    llm_from_settings_or_context,
)

from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever

import os


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KGTableRetriever,
        mode: str = "OR",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        
        #HyDE 
        self.hyde = HyDEQueryTransform(include_original=True)
        
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    # def _retrieve(self, query_str) :
        """Retrieve nodes given query."""
        # query_bundle = QueryBundle(query_str)
        ## ---HYDE ---
        # query_bundle = self.hyde(query_bundle.query_str)

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        print("vector_nodes ",len(vector_nodes))
        ## --- LLM Reranker ---
        # reranker = LLMRerank(choice_batch_size=10, top_n=2)
        # vector_nodes = reranker.postprocess_nodes(vector_nodes, query_bundle)
        ## --- Cohere Reranker ---
        print("Using Cohere Reranker")
        api_key = os.environ["CO_API_KEY"]
        reranker = CohereRerank(model='rerank-english-v3.0',api_key=api_key, top_n=2)
        vector_nodes = reranker.postprocess_nodes(vector_nodes, query_bundle)
        print("Reranker ",len(vector_nodes))
        
        kg_nodes = self._kg_retriever.retrieve(query_bundle)
        print("KG_nodes ",len(kg_nodes))

        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes