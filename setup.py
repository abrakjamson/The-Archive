"""
Copyright Abram Jackson 2024
"""

"""
This file uses the following packages that otherwise aren't needed
threadpoolctl, scipy, safetensors, regex, Pillow, joblib, scikit-learn, tokenizers, transformers, sentence-transformers
torch (with CUDA) 12.1
"""

import asyncio
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from pyarrow import dataset as ds, parquet
import datasets  
from datasets import Dataset
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from elasticsearch import Elasticsearch

# Initialize the LlamaCppEmbeddings with the path to your model
# llama = LlamaCppEmbeddings(model_path=_model_path, n_gpu_layers=33, verbose=False)
sbert = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")

_elastic_search_client = Elasticsearch("http://localhost:9200")

def calculate_embeddings_llamacpp(examples):
    # Initialize an empty dictionary to hold the results
    results = {'id': [], 'title': [], 'embedding': []}

    # Iterate over the range of the list length
    for i in range(len(examples['id'])):
        # Access each element by index
        text = examples['text'][i]
        # Split the text into paragraphs
        paragraphs = text.split("\n\n")
        embeddings = llama.embed_documents(paragraphs)
        # Calculate embeddings for each paragraph
        for j in range(len(embeddings)):
            # Append the results to the corresponding lists in the results dictionary
            results['id'].append(examples['id'][i] + "." + str(j))
            results['title'].append(examples['title'][i])
            results['embedding'].append(embeddings[j])
    
    # Return the results dictionary
    return results

def calculate_embeddings_transformers(examples):    
    # Initialize an empty dictionary to hold the results
    embeds = {'id': [], 'paragraph': [], 'embedding': []}

    # Iterate over the range of the list length
    for i in range(len(examples['id'])):
        # Access each element by index
        text = examples['text'][i]
        # Split the text into paragraphs
        example_paragraphs = text.split("\n\n")
        for j in range(len(example_paragraphs)):
            embeds['id'].append(examples['id'][i] + '.' + str(j)) 
            #embeds['id'].append(examples['id'][i]) 
            embeds['paragraph'].append(example_paragraphs[j])
        
    embeds['embedding'] = sbert.encode(embeds['paragraph'], batch_size=700,precision="int8")
    results = {'id': embeds['id'], 'embedding': embeds['embedding']}
    return results

def process_embeddings():
    _model_path = "models\\gist-small-embedding-v0.Q8_0.gguf"
    _data_path = "data\\wikimedia___wikipedia"

    wiki_dataset = datasets.load_dataset(_data_path)
    subset = wiki_dataset['train'].select(range(2100))
    
    for i in range(28):
        file_name = f"data\\gist_embeds_{i}.parquet"
        file_name = f"data\\gist_embeds_{i}_1_percent.parquet"
        # Check if the file already exists
        if not os.path.exists(file_name):
            # 6,343,736 wikipedia articles is divisible by 28 = 226,562
            # So we will make 28 files of 226k each
            print(f"Will now process for file number {i}")
            # magic_number = 226562 # all data in 28 batches
            magic_number = 226 # for testing, this is .1% of the batch and data size
            subset = wiki_dataset['train'].select(range(i * magic_number, (i+1) * magic_number))
            dataset = subset.map(calculate_embeddings_transformers, batched=True, batch_size=7000, remove_columns=['url','title','text'])
            writer = parquet.ParquetWriter(file_name, dataset.data.schema)
            for chunk in dataset.data.to_batches(10000):
                writer.write_batch(chunk, row_group_size=100)
            writer.close()
        else:
            print(f"Skipping {file_name}, already exists.")

def add_to_embedding_index(examples):
    for i in range(len(examples['id'])):
        _elastic_search_client.index(
            index="embedding_index",
            id=examples["id"],
            document=
            {
                "paragraph_id": examples['id'][i],
                "embedding.predicted_value": examples['embedding'][i]
            }
        )
    
    file_path = "data\\gist_embeds"
    parquet_files = [f for f in os.listdir(file_path) if f.endswith('.parquet')]
    dataframes = [pd.read_parquet(os.path.join(file_path, file)) for file in parquet_files]
    combined_df = pd.concat(dataframes, ignore_index=True)
    first = combined_df.iloc[0]
    # If I'm reading the docs correctly, this will set up with int8 HSNW
    _elastic_search_client.index(
        index="embedding_index",
        id=first["id"],
        document=
        {
            "paragraph_id": first['id'],
            "embedding.predicted_value": first['embedding']
        }
    )
    _elastic_search_client.get(index="embedding_index", id="12.0")
    _elastic_search_client.delete(index="embedding_index", id="12.0")
    _elastic_search_client.search(
        index="embedding_index",
        query={
            "knn": {
                "field": "embedding.predicted_value",
                "query_vector": sbert.encode("criticisms of anarchy")
            }
        }
    )


def index_embeddings():
    # load in the files saved during process_embeddings()
    file_path = "data\\gist_embeds"
    parquet_files = [f for f in os.listdir(file_path) if f.endswith('.parquet')]
    dataframes = [pd.read_parquet(os.path.join(file_path, file)) for file in parquet_files]
    combined_df = pd.concat(dataframes, ignore_index=True)
    _elastic_search_client = Elasticsearch("http://localhost:9200")
    
    mappings = {
        "properties": {
            "paragraph_id": {"type": "text"},
            "embedding.predicted_value": { 
                "type": "dense_vector", 
                "dims": 384
            }
        }
    }
    _elastic_search_client.indices.delete(index="embedding_index")
    _elastic_search_client.indices.create(index="embedding_index", mappings=mappings)

    embedding_dataset = Dataset.from_pandas(combined_df)
    embedding_dataset.map(add_to_embedding_index, batched=True, batch_size=100)
    #test_article = combined_df.iloc[1]
    
    #_elastic_search_client.get(index="embedding_index",id="0")
    #_elastic_search_client.search(size=1,index="embedding_index")


def index_lexical():
    # TODO move from the local_wikipedia module
    pass

index_embeddings()