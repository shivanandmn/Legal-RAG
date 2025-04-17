import os
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.schema import TextNode
from llama_index.core import Settings, VectorStoreIndex
from dotenv import load_dotenv, find_dotenv
import datetime
import logging
import sys
from functools import lru_cache

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


_ = load_dotenv(find_dotenv())

##TOOL: pinecone_index.describe_index_stats()
##TOOL: pinecone_index.delete(delete_all=True, namespace='')


@lru_cache
def load_pinecone_index(index_name, host=""):
    """Base Index from pinecone
    This can be used to create multi-tenant using llama-index vector-store object"""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if host == "":
        if index_name not in pc.list_indexes().names():
            # create the index
            print(f"Creating Index {index_name}")
            pc.create_index(
                name=index_name,
                dimension=256,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        print(f"About {index_name} Index :\n{pc.describe_index(index_name)}")
    pinecone_index = pc.Index(
        index_name, host=host
    )  ##TODO: In prod change from name to host url
    return pinecone_index


def get_rag_index(pinecone_index, namespace=""):
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, namespace=namespace
    )
    # print("INDEX NAME :", vector_store.index_name)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store  # , insert_batch_size=20
    )
    return index


def delete(pc, index_name):
    try:
        pc.delete_index(index_name)
        print(f"{index_name} is deleted!")
    except Exception as e:
        print("Error :", e)


# pinecone_index.delete(delete_all=True, namespace="case_laws")
