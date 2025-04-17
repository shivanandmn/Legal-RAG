from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core.settings import Settings

_ = load_dotenv(find_dotenv())
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from service_config import service_config
from llama_index.core import get_response_synthesizer
from prompts import QA_PROMPT_TMPL, query_clf_prompt
from postprocessor import (
    LLMIncludeALLFieldsPostprocessor,
    SimilarityPostprocessor,
    SectionCasesOnCourt,
)
import os
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.schema import QueryBundle
from prompts import query_sectiowise_from_failed_json_prompt
from llms import get_gemini_token_count
from vs_indexer import load_pinecone_index
from data_validators.response import llm_to_response, CaseDetails

SIMILARITY_TOP_K = service_config.query_search["similarity_top_k"]
LONG_ANSW_TOP_K = service_config.query_search["long_answ_top_k"]
llm_field_postprocessor = LLMIncludeALLFieldsPostprocessor(
    exclude_keys_to_allow_all=service_config.query_search_sectionwise[
        "exclude_keys_to_allow_all"
    ]
)
similarity_postprocessor = SimilarityPostprocessor(
    similarity_cutoff=service_config.query_search["similarity_cutoff"]
)
cohere_rerank = CohereRerank(
    api_key=os.environ["COHERE_API_KEY"], top_n=LONG_ANSW_TOP_K
)

post_processors = [ llm_field_postprocessor ]
# if service_config.query_search.get("similarity_cutoff", None):
#     post_processors.append(similarity_postprocessor)


# FilterOperator
def get_retriever(favour_name="assessee_favour"):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="assessee_favour",
                operator=FilterOperator.EQ,
                value="True" if favour_name == "assessee_favour" else "False",
            ),
            MetadataFilter(
                key="revenue_favour",
                operator=FilterOperator.EQ,
                value="True" if favour_name == "revenue_favour" else "False",
            ),
        ],
        operator="and",
    )

    retriever = VectorIndexRetriever(
        index=service_config.system_indexer,
        similarity_top_k=SIMILARITY_TOP_K,
        filters=filters,
    )
    return retriever


# num_vectors = pinecone_index.describe_index_stats()
# num_vectors = num_vectors.namespaces["case_laws"].vector_count
# num_vectors

EMPTY_RESPONSE_REPLY_STR = """Your query did not receive a response from our server.

This might occur if there is no relevant information for your query or if the query is too vague. Please try again by framing your query as a direct question. Be sure to provide clear and specific details about the legal issue or case for a more accurate response."""


class SimpleRetrieverGeneration:
    def __init__(self, response_mode="compact_accumulate") -> None:
        SIMILARITY_TOP_K = service_config.query_search["similarity_top_k"]
        self.post_processors = post_processors
        self.task_type = service_config.query_search["task_type"]
        self.query_classification_model_name = service_config.query_search[
            "query_classification_model_name"
        ]
        self.indexer = service_config.system_indexer
        self.simple_retriever = VectorIndexRetriever(
            index=service_config.system_indexer, similarity_top_k=SIMILARITY_TOP_K
        )
        self.response_mode = response_mode
        self.query_engine = {
            model: self.create_query_engine(model)
            for model in service_config.MODELS.keys()
        }

    def get_gemini_token_count(self, text):
        return get_gemini_token_count(text)

    def model_with_task_type(self, model):
        return f"{model}:{self.task_type}"

    def create_query_engine(self, llm):
        if llm in [
            self.model_with_task_type("Flint"),
            self.model_with_task_type("Candid"),
        ]:
            prompt_helper = PromptHelper(1_000_000, num_output=8_000)
        else:
            prompt_helper = None
        Settings.llm = service_config.MODELS[llm]
        response_synthesizer = get_response_synthesizer(
            response_mode=self.response_mode,
            use_async=False,
            streaming=False,
            text_qa_template=QA_PROMPT_TMPL,
            prompt_helper=prompt_helper,
        )
        simple_query_engine = RetrieverQueryEngine(
            retriever=self.simple_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=self.post_processors,
        )
        return simple_query_engine

    def complete_query(self, query_str, model):
        response = service_config.MODELS[self.model_with_task_type(model)].complete(
            query_clf_prompt.format(query_str=query_str)
        )
        return response.text

    def get_query_response(self, query_str, model):
        return self._get_query_response(query_str, model)

    def _get_query_response(self, query_str, model):
        ## classify query
        extra_info = {"query_str": query_str}
        clf_model_name = self.query_classification_model_name
        # response = self.complete_query(
        #     query_str,
        #     clf_model_name,
        # )
        # print("Exception is added to get classification as True")
        response = "Yes"
        extra_info["query_clf_model_name"] = clf_model_name
        extra_info["query_class"] = response
        if response.lower() != "yes":

            return (
                f"Query seems to be irrelevent to the legal issue or case. Kindly elaborate or rephrase your query.",  # Last updated on 17.06.2024
                False,
                extra_info,
            )

        response = self.query_engine[self.model_with_task_type(model)].query(query_str)

        extra_info["sources"] = [
            {"file_name": x.node.metadata["file_name"], "score": x.score}
            for x in response.source_nodes
        ]
        source_node_map = {}
        for x in response.source_nodes:
            metadata = x.node.metadata
            try:
                metadata["court"] = service_config.courtname_to_court_map[
                    metadata["court_name"]
                ]
            except:
                metadata["court"] = metadata["court_name"]
            source_node_map[x.node.metadata["file_name"]] = {
                "metadata": metadata,
                "node_id": x.node.node_id,
            }
        extra_info["source_nodes_map"] = source_node_map

        response = response.response
        extra_info["raw_response"] = response
        extra_info["response_generation_model"] = model
        if response == "Empty Response":
            return (
                EMPTY_RESPONSE_REPLY_STR,
                False,
                extra_info,
            )
        else:
            return response, True, extra_info

    def get_retrieve_nodes(self, query_str, filters=[]):
        nodes = self.simple_retriever.retrieve(QueryBundle(query_str))
        for filter in filters:
            nodes = filter.postprocess_nodes(nodes, query_str=query_str)
        return nodes

    def retrieve_exact_match_nodes(self, key, value, topk=5):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key=key, operator=FilterOperator.EQ, value=value)
            ]
        )
        retriever = VectorIndexRetriever(
            index=self.indexer,
            similarity_top_k=topk,
            vector_store_query_mode="sparse",
            filters=filters,
        )
        nodes = retriever.retrieve("")
        return nodes

    def add_metadata(self, data, case_metadata):
        cases = []
        for cs in data:
            try:
                cs["metadata"] = case_metadata[cs["file_name"].strip()]["metadata"]
                cs["metadata"] = CaseDetails(**cs["metadata"]).model_dump()
                cs["name"] = cs["metadata"]["name_of_case"]
                cases.append(cs)
            except:
                pass
        return cases


class ITACTRetrieverGeneration(SimpleRetrieverGeneration):
    def __init__(self, response_mode="compact_accumulate"):
        SIMILARITY_TOP_K = service_config.itact_search["similarity_top_k"]
        self.task_type = service_config.itact_search["task_type"]
        self.indexer = service_config.system_indexer_itact
        self.simple_retriever = VectorIndexRetriever(
            index=self.indexer, similarity_top_k=SIMILARITY_TOP_K
        )
        self.query_classification_model_name = service_config.itact_search[
            "query_classification_model_name"
        ]
        
        self.response_mode = response_mode
        self.query_engine = {
            model: self.create_query_engine(model)
            for model in service_config.MODELS.keys()
        }
        self.post_processors = []


# get_structured_data
class QuerySearchSectionwiseRetrieverGeneration(SimpleRetrieverGeneration):
    def __init__(self, response_mode="compact_accumulate"):
        SIMILARITY_TOP_K = service_config.query_search_sectionwise["similarity_top_k"]
        self.task_type = service_config.query_search_sectionwise["task_type"]
        self.indexer = service_config.system_indexer
        self.simple_retriever = VectorIndexRetriever(
            index= self.indexer, similarity_top_k=SIMILARITY_TOP_K
        )
        self.query_classification_model_name = service_config.query_search_sectionwise[
            "query_classification_model_name"
        ]
        self.response_mode = response_mode
        self.post_processors = post_processors
        self.query_engine = {
            model: self.create_query_engine(model)
            for model in service_config.MODELS.keys()
        }

    def get_query_response(self, query_str, model):
        response, is_valid, extra_info = self._get_query_response(query_str, model)
        if not is_valid:
            return response, is_valid, extra_info
        
        is_valid, result = llm_to_response(response)
        if is_valid:
            result["cases"] = self.add_metadata(
                result["cases"], extra_info["source_nodes_map"]
            )
            if "file_name" in result:
                del result["file_name"]
            return result, is_valid, extra_info
        else:
            response = self.generate_json_schema_using_llm(response, model)
            is_valid, result = llm_to_response(response)
            if is_valid == False:
                result = {"cases":[]}
        extra_info["regenerated_json_using_llm"] = response
        result["cases"] = self.add_metadata(
            result["cases"], extra_info["source_nodes_map"]
        )
        if "file_name" in result:
            del result["file_name"]
        return result, is_valid, extra_info

    def generate_json_schema_using_llm(self, query_str, model):

        response = service_config.MODELS[self.model_with_task_type(model)].complete(
            query_sectiowise_from_failed_json_prompt.format(query_str=query_str)
        )
        return response.text


class DefinedTermsRetriever(SimpleRetrieverGeneration):
    def __init__(self, response_mode="compact_accumulate"):
        super().__init__(response_mode)
        self.post_processors = []
        self.task_type = service_config.defined_terms_search["task_type"]
        self.query_classification_model_name = service_config.defined_terms_search[
            "query_classification_model_name"
        ]
        self.simple_retriever = VectorIndexRetriever(
            index=service_config.system_indexer_defined_terms,
            similarity_top_k=service_config.defined_terms_search["similarity_top_k"],
        )
        self.post_processors = []
        section_wise = service_config.defined_terms_search.get("section_wise", None)
        self.metadata_index_namespace = service_config.query_search_sectionwise[
            "namespace"
        ]
        self.metadata_index = load_pinecone_index(
            self.metadata_index_namespace,
            service_config.query_search_sectionwise["pinecone_host"],
        )  ##temporirily added
        if section_wise:
            self.post_processors.append(SectionCasesOnCourt(section_wise))

    def get_retrieve_nodes(self, query_str):
        nodes = self.simple_retriever.retrieve(QueryBundle(query_str))
        for filter in self.post_processors:
            nodes = filter.postprocess_nodes(nodes, query_str=query_str)
        return nodes

    def get_defined_terms(self, query_str):
        nodes = self.get_retrieve_nodes(query_str)
        response_nodes = []
        node_ids = [node.node.metadata["related_node_id"] for node in nodes]
        node_id_to_meta = self.metadata_index.fetch(
            ids=node_ids, namespace=self.metadata_index_namespace
        )["vectors"]
        for node in nodes:
            metadata = node.node.metadata
            try:
                data = node_id_to_meta[metadata["related_node_id"]]["metadata"]
                data.update(metadata)
                data["court"] = service_config.courtname_to_court_map[
                    data["court_name"]
                ]
                try:
                    remove_fields = [
                        "_node_content",
                        "_node_type",
                        "doc_id",
                        "document_id",
                        "file_name",
                        "ref_doc_id",
                        "related_node_id",
                    ]
                    for field in remove_fields:
                        del data[field]

                except:
                    pass

                response_nodes.append(data)
            except:
                pass
        return response_nodes


# retriever_generator = SimpleRetrieverGeneration()
# itact_retriever_generator = ITACTRetrieverGeneration()
# retriever_generator = SimpleRetrieverGeneration()
# query_sectionwise_retriever_generator = QuerySearchSectionwiseRetrieverGeneration()
sectionwise_retriever_generator = QuerySearchSectionwiseRetrieverGeneration()
defined_terms_retriever_generator = DefinedTermsRetriever()


if __name__ == "__main__":
    queries_list = [
        "Whether the erection of a boiler by the assessee constitutes a manufacturing activity for the purpose of claiming investment allowance?",
        "supreme court decisions on whether depreciation can be claimed on goodwill?",
        "Reopening of assessment based on audit query"
    ]
    import time
    for query_str in queries_list:
        now = time.time()
        # nodes, response = retrieve_query_response(simple_query_engine, query_str)
        response = sectionwise_retriever_generator.get_query_response(query_str, "Flint")
        response_text = response[0]
        print(time.time()-now)
    # print(response)
# COMPUTATION OF TOTAL INCOMECHAPTER IV.pdf
# PROCEDURE FOR ASSESSMENTCHAPTER XIV.pdf
# DEDUCTIONS TO BE MADE IN COMPUTING TOTAL.pdf
# DETERMINATION OF TAX IN CERTAIN SPECIAL .pdf
# MISCELLANEOUSCHAPTER XXIII.pdf
