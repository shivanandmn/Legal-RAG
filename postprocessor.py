from typing import Optional, List
from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from datetime import datetime
from service_config import ServiceConfig
from llama_index.core.postprocessor import SimilarityPostprocessor


class CaseSortPostprocessor(BaseNodePostprocessor):
    """
    a. Date of Decision
    b. Supreme Court> High Court> ITAT
    """

    @classmethod
    def class_name(cls) -> str:
        return "CaseSortPostprocessor"

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        nodes = sort_based_on_date_court(nodes)
        return nodes


def rename(court):
    court_name = court.lower()
    if "supreme" in court_name:
        court = "supreme court"
    elif "high" in court_name:
        court = "high court"
    elif "itat" in court_name:
        court = "itat"
    else:
        raise Exception(f"Invlid Court name found {court_name}")
    return court


def sort_based_on_date_court(nodes):
    court_priority = {
        "supreme court": 3,  # Highest priority
        "high court": 2,
        "itat": 1,  # Lowest priority
    }

    def sort_key_inverted(record):
        court, date = record
        court = rename(court)
        return (court_priority[court], date.toordinal())

    nodes = [
        (
            node,
            (
                node.metadata["court_name"],
                datetime.strptime(node.metadata["date_of_decision"], "%d/%m/%Y"),
            ),
        )
        for node in nodes
    ]
    nodes = sorted(nodes, key=lambda x: sort_key_inverted(x[1]), reverse=True)
    return [x[0] for x in nodes]


class LLMIncludeALLFieldsPostprocessor(BaseNodePostprocessor):
    exclude_keys_to_allow_all: list[str] = []
    """
    a. Date of Decision
    b. Supreme Court> High Court> ITAT
    """

    @classmethod
    def class_name(cls) -> str:
        return "CaseSortPostprocessor"

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        for node_with_score in nodes:
            node_with_score.node.excluded_llm_metadata_keys = (
                self.exclude_keys_to_allow_all
            )
        return nodes


class MetadataExistsPostprocessor(BaseNodePostprocessor):
    node_process_config: ServiceConfig = None
    """
    """

    @classmethod
    def class_name(cls) -> str:
        return "CaseSortPostprocessor"

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        for node_with_score in nodes:
            is_metadata_exists = node_with_score.node.metadata.get(
                "is_metadata_exists", False
            )
            if is_metadata_exists == False:
                # node_with_score.node.metadata = #Download from cache or mongodb
                pass
        return nodes


class SectionCasesOnCourt:
    """
    Section 1	Supreme Court / High Court   SC: retrieve up to 20 cases,  HC: retrieve up to 20 cases
    Section 2	ITAT / AAR	               ITAT: retrieve up to 20 cases, AAR: retrieve up to 20 cases
    Section 3	Other Cases	                   : retrieve up to 20 cases
    {'ITAT', 'Supreme Court', 'Other Courts', 'High Court', 'Authority for Advance Ruling'}

    """

    def __init__(self, config) -> None:
        self.court_threshold_count = config["court_threshold_count"]

    def postprocess_nodes(self, nodes, query_str):
        courts = {x: [] for x in self.court_threshold_count.keys()}
        for node_with_score in nodes:
            court_name = node_with_score.node.metadata["court_name"]
            if len(courts[court_name]) <= self.court_threshold_count[court_name]:
                courts[court_name].append(node_with_score)
        return courts


if __name__ == "__main__":
    query_str = (
        "Once penalty levied is set aside, prosecution proceedings have to be quashed?"
    )
    from llama_index.core.retrievers import VectorIndexRetriever
    from service_config import service_config
    from llama_index.core.query_engine import RetrieverQueryEngine

    service_config = ServiceConfig()
    SIMILARITY_TOP_K = service_config.query_search["similarity_top_k"]
    LONG_ANSW_TOP_K = service_config.query_search["long_answ_top_k"]
    postprocessor = LLMIncludeALLFieldsPostprocessor(node_process_config=service_config)
    simple_retriever = VectorIndexRetriever(
        index=service_config.system_indexer, similarity_top_k=SIMILARITY_TOP_K
    )
    simple_query_engine = RetrieverQueryEngine(
        retriever=simple_retriever,
        node_postprocessors=[postprocessor],
    )
    nodes = simple_query_engine.retrieve(query_str)
    print(nodes)
