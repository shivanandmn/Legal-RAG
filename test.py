from tqdm import tqdm
import time
import pandas as pd
from retrieval import sectionwise_retriever_generator
import random

with open("./data/examples.txt", "r") as file:
    queries_list = file.readlines()
queries_list = [x.strip() for x in queries_list] 
queries_list = ['When Assessing Officer has duly considered the explanation of a matter during the assessment proceedings, whether invocation of revisionary order under section 263 by PCIT is incorrect?'] + random.choices(queries_list, k=10)
data = []
try:
    for query_str in tqdm(queries_list):
        now = time.time()
        # nodes, response = retrieve_query_response(simple_query_engine, query_str)
        results = sectionwise_retriever_generator.get_query_response(query_str, "Flint")
        data.append(results)
        print(time.time() - now)
except Exception as e:
    print(str(e))
pd.DataFrame(data=data, columns=["result", "is_valid", "extra_info"]).to_csv(
    f"./data/sample{len(queries_list)}.csv", index=False
)
