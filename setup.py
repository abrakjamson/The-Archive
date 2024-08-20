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
from elasticsearch import Elasticsearch, exceptions, helpers
import logging

# Initialize the LlamaCppEmbeddings with the path to your model
# llama = LlamaCppEmbeddings(model_path=_model_path, n_gpu_layers=33, verbose=False)
class Setup:
    def __init__(self, model_path="../../models\\avsolatorio/GIST-small-Embedding-v0", elasticsearch_port=9200):
        self._model_path = model_path
        self._sbert = SentenceTransformer(model_path)
        self._elastic_search_client = Elasticsearch("http://localhost:" + str(elasticsearch_port))
 
    def _calculate_embeddings_llamacpp(self, examples):
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

    def _calculate_embeddings_transformers(self, examples):    
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
            
        embeds['embedding'] = self._sbert.encode(embeds['paragraph'], batch_size=700,precision="int8")
        results = {'id': embeds['id'], 'embedding': embeds['embedding']}
        return results

    def process_embeddings(self):
        """ Calculates embeddings for the wikipedia downloaded by download_wikipedia.
            It uses checkpointing to disk, as this is likely to take days to weeks.
        """
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
                dataset = subset.map(self._calculate_embeddings_transformers, batched=True, batch_size=7000, remove_columns=['url','title','text'])
                writer = parquet.ParquetWriter(file_name, dataset.data.schema)
                for chunk in dataset.data.to_batches(10000):
                    writer.write_batch(chunk, row_group_size=100)
                writer.close()
            else:
                print(f"Skipping {file_name}, already exists.")

    def _add_to_embedding_index(self, examples):      
        bulk_data = []
        def __generate_data():
            for i in range(len(examples['id'])):
                yield {
                    "_op_type": "index",
                    "_index": "embedding_index",
                    "_id": examples['id'][i],
                    "_source": {
                        "embedding.predicted_value": examples['embedding'][i]
                    }
                }
        # sends the entire batch over http to the ES server
        # must stay under 100 MB
        helpers.bulk(self._elastic_search_client, __generate_data())        

    def index_embeddings(self):
        # load in the files saved during process_embeddings()
        file_path = "..\..\\data\\gist_embeds"
        parquet_files = [f for f in os.listdir(file_path) if f.endswith('.parquet')]
        dataframes = [pd.read_parquet(os.path.join(file_path, file)) for file in parquet_files]
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # If I'm reading the docs correctly, this will set up with int8 HSNW
        # The ID is the article ID used in the dataset, ".",  and then 0-indexed paragraph number
        mappings = {
            "properties": {
                "paragraph_id": {"type": "text"},
                "embedding.predicted_value": { 
                    "type": "dense_vector",
                    "element_type": "byte",
                    "dims": 384
                }
            }
        }
        try:
            self._elastic_search_client.indices.create(index="embedding_index", mappings=mappings)
        except ConnectionError:
            logging.error("Could not connect to Elasticsearch server. Is it running?")
        except exceptions.BadRequestError:
            logging.warn("Embedding index is already created. It will not be modified.")

        embedding_dataset = Dataset.from_pandas(combined_df)

        # I expect to get bottlenecked on hard disk, so this doesn't need to be in parallel
        # batch of 10,000 should be around 10 MB to send at a time
        embedding_dataset.map(self._add_to_embedding_index, batched=True, batch_size=10000)

    def download_wikipedia(self):
        self._dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en",  cache_dir="..\\..\\data")

    def index_lexical(self):
        # TODO move from the local_wikipedia module
        pass

if __name__ == "__main__":
    setup_instance = Setup()
    setup_instance.download_wikipedia()
    setup_instance.index_lexical()
    setup_instance.process_embeddings()
    setup_instance.index_embeddings()

    # should return the embedding of the first paragraph of "Anarchy"
    # which is just the first article in the data! I'm not an anarchist
    setup_instance._elastic_search_client.get(index="embedding_index", id="12.0")

    # should return several embeddings with id == 12.*
    setup_instance._elastic_search_client.search(
            index="embedding_index",
            query={
                "knn": {
                    "field": "embedding.predicted_value",
                    # TODO add a calibration embedding when using for real
                    "query_vector": setup_instance._sbert.encode("criticisms of anarchy")
                }
            }
        )

