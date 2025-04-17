from llama_index.core.settings import Settings

from llama_index.embeddings.openai import OpenAIEmbedding

from llms import get_openai_model, get_gemini_model

from vs_indexer import load_pinecone_index, get_rag_index

import json
import prompts


def get_model_by_name(name, system_prompt):
    if name.startswith("gpt"):
        return get_openai_model(name, system_prompt)
    elif name.startswith("gemini"):
        return get_gemini_model(name, system_prompt)
    else:
        raise NotImplementedError(f"Model {name} not implemented")


class ServiceConfig:

    query_search = json.load(open("./config/query_search.json"))
    query_search_sectionwise = json.load(open("./config/query_search_sectionwise.json"))
    itact_search = json.load(open("./config/itact_search.json"))
    prompt_config = json.load(open("./config/prompt_config.json"))
    defined_terms_search = json.load(open("./config/defined_terms_search.json"))
    courtname_to_court_map = json.load(open("./config/court_name_to_court_map.json"))

    Settings.llm = get_openai_model()  # system_prompt="The Context"
    Settings.embed_model = OpenAIEmbedding(
        model=query_search["embedding_model_name"],
        dimensions=query_search["embedding_dim"],
    )
    system_indexer = get_rag_index(
        load_pinecone_index("", host=query_search["pinecone_host"]),
        namespace=query_search["namespace"],
    )
    system_indexer_itact = get_rag_index(
        load_pinecone_index("", host=itact_search["pinecone_host"]),
        namespace=itact_search["namespace"],
    )
    system_indexer_defined_terms = get_rag_index(
        load_pinecone_index("", host=defined_terms_search["pinecone_host"]),
        namespace=defined_terms_search["namespace"],
    )
    MODELS = {
        (name + ":" + task_type): get_model_by_name(
            model_name, system_prompt=prompts.qa_system_prompt
        )
        for (name, model_name), task_type in zip(
            query_search["models"].items(),
            [query_search["task_type"]] * len(query_search["models"]),
        )
    }
    MODELS.update(
        {
            f"{name}:{task_type}": get_model_by_name(
                model_name, system_prompt=prompts.itact_system_prompt
            )
            for (name, model_name), task_type in zip(
                itact_search["models"].items(),
                [itact_search["task_type"]] * len(itact_search["models"]),
            )
        }
    )
    MODELS.update(
        {
            f"{name}:{task_type}": get_model_by_name(
                model_name, system_prompt=prompts.defined_terms_system_prompt
            )
            for (name, model_name), task_type in zip(
                defined_terms_search["models"].items(),
                [defined_terms_search["task_type"]]
                * len(defined_terms_search["models"]),
            )
        }
    )

    MODELS.update(
        {
            f"{name}:{task_type}": get_model_by_name(
                model_name, system_prompt=prompts.qa_section_system_prompt
            )
            for (name, model_name), task_type in zip(
                defined_terms_search["models"].items(),
                [defined_terms_search["task_type"]]
                * len(defined_terms_search["models"]),
            )
        }
    )
    # query_search_sectionwise
    MODELS.update(
        {
            f"{name}:{task_type}": get_model_by_name(
                model_name, system_prompt=prompts.qa_sectionwise_system_prompt
            )
            for (name, model_name), task_type in zip(
                query_search_sectionwise["models"].items(),
                [query_search_sectionwise["task_type"]]
                * len(query_search_sectionwise["models"]),
            )
        }
    )


service_config = ServiceConfig()
print()
# load_pinecone_index("", host="alpha-1-iezijhd.svc.aped-4627-b74a.pinecone.io").delete(
#     delete_all=True, namespace="defined_terms"
# )
