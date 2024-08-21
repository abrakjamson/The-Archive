"""
Copyright Abram Jackson 2024
"""

"""
This module uses the following packages that otherwise aren't needed by the project:
threadpoolctl, scipy, safetensors, regex, Pillow, joblib, scikit-learn, tokenizers, transformers, sentence-transformers, torch
"""

import asyncio
import logging
import os
import datasets
from datasets import Dataset
from pyarrow import parquet
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from elasticsearch import Elasticsearch, exceptions, helpers

# Initialize the LlamaCppEmbeddings with the path to your model
# llama = LlamaCppEmbeddings(model_path=_model_path, n_gpu_layers=33, verbose=False)
class Setup:
    def __init__(self, model="avsolatorio/GIST-small-Embedding-v0", elasticsearch_port=9200):
        logging.getLogger().setLevel(logging.WARN)
        self._sbert = SentenceTransformer(model,cache_folder="../../models/")
        self._elastic_search_client = Elasticsearch("http://localhost:" + str(elasticsearch_port))
 
    def download_models(self):
        # GIST model
        hf_hub_download(cache_dir="../../models",repo_id="avsolatorio/GIST-small-Embedding-v0", filename="model.safetensors")
        # Phi-3 mini
        hf_hub_download(cache_dir="../../models", repo_id="bartowski/Phi-3.1-mini-128k-instruct-GGUF", filename="Phi-3.1-mini-128k-instruct-IQ4_XS.gguf")

    def download_wikipedia(self):
        self._dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en",  cache_dir="..\\..\\data")
    
    def process_embeddings(self):
        """ Calculates embeddings for the wikipedia downloaded by download_wikipedia.
            It uses checkpointing to disk, as this is likely to take days to weeks and need restart.

            This function should be called after download_models()
        """
        _data_path = "../../data/wikimedia___wikipedia"

        wiki_dataset = datasets.load_dataset(_data_path)
        subset = wiki_dataset['train'].select(range(2100))
        
        for i in range(28):
            # file_name = f"data/gist_embeds_{i}.parquet"
            file_name = f"data/gist_embeds_{i}_one_percent.parquet"
            # Check if the file already exists
            if not os.path.exists(file_name):
                # 6,343,736 wikipedia articles is divisible by 28 = 226,562
                # So we will make 28 files of 226k each
                logging.info(f"Will now process for file number {i}")
                # batch_size = 226562 # all data in 28 batches
                batch_size = 2266 # for testing, this is 1% of the batch and data size
                subset = wiki_dataset['train'].select(range(i * batch_size, (i+1) * batch_size))
                dataset = subset.map(self._calculate_embeddings, 
                                     batched=True, 
                                     batch_size=7000, 
                                     remove_columns=['url','title','text'])
                # Writing in chunks to avoid putting it all in memory
                # 10,000 items should be less than 1 GB
                writer = parquet.ParquetWriter(file_name, dataset.data.schema)
                for chunk in dataset.data.to_batches(10000):
                    writer.write_batch(chunk, row_group_size=100)
                writer.close()
            else:
                print(f"Skipping {file_name}, already exists.")
    
    def _calculate_embeddings(self, examples):    
        """ Calculates a batch of articles into paragraph embeddings
        """
        # Create the longer lists of paragraphs
        embeds = {'id': [], 'paragraph': []}
        for i in range(len(examples['id'])):
            text = examples['text'][i]
            example_paragraphs = text.split("\n\n")
            for j in range(len(example_paragraphs)):
                # ID format is article_id + "." sequential paragraph_id
                embeds['id'].append(examples['id'][i] + '.' + str(j)) 
                embeds['paragraph'].append(example_paragraphs[j])

        # Calculate embeddings in batch.
        # From testing, a batch of 700 uses about 11 GB of video memory
        paragraph_embeddings = self._sbert.encode(embeds['paragraph'], batch_size=700,precision="int8")
        results = {'id': embeds['id'], 'embedding': paragraph_embeddings}
        return results  

    def index_embeddings(self):
        """ Adds the embeddings that were saved to disk to Elasticsearch's index
        """
        # load in the files saved during process_embeddings()
        file_path = "../../data/gist_embeds"
        parquet_files = [f for f in os.listdir(file_path) if f.endswith('.parquet')]
        dataframes = [pd.read_parquet(os.path.join(file_path, file)) for file in parquet_files]
        combined_df = pd.concat(dataframes, ignore_index=True)
        embedding_dataset = Dataset.from_pandas(combined_df)

        # If I'm reading the docs correctly, this will set up with int8 HSNW
        # We'll see if 384 bytes per paragraph are going to blow up memory too bad
        #  if so, we can calculate 1 byte embeddings, but index only half-byte
        #  I don't want to have to re-architect to 1.7 bit
        # The ID is the article ID used in the dataset, ".", and then 0-indexed paragraph number
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
            return
        except exceptions.BadRequestError:
            logging.warn("Embedding index is already created. It will not be modified.")     

        # I expect to get bottlenecked on disk IO, so this doesn't need to be in parallel
        # batch of 10,000 should be around 10 MB to send over http at a time
        embedding_dataset.map(self._add_to_embedding_index, batched=True, batch_size=10000)

    def _add_to_embedding_index(self, documents):    
        """ Indexes a batch of paragraph embeddings into Elasticsearch
        """  
        bulk_data = []
        def __generate_data():
            for i in range(len(documents['id'])):
                yield {
                    "_op_type": "index",
                    "_index": "embedding_index",
                    "_id": documents['id'][i],
                    "_source": {
                        "embedding.predicted_value": documents['embedding'][i]
                    }
                }
        # sends the entire batch over http to the ES server
        # must stay under 100 MB
        helpers.bulk(self._elastic_search_client, __generate_data())   
    
    def _add_to_lexical_index(self, documents):
        """Adds a batch of documents to Elasticsearch's index
        """
        bulk_data = []
        def __generate_data():
            for i in range(len(documents['id'])):
                yield {
                    "_op_type": "index",
                    "_index": "lexical_index",
                    "_id": documents['id'][i],
                    "_source": {
                        "title": documents["title"],
                        "text": documents["text"]
                    }
                }
        # sends the entire batch over http to the ES server
        # must stay under 100 MB
        helpers.bulk(self._elastic_search_client, __generate_data())      

    def index_lexical(self):
        """ Adds the downloaded Wikpedia dataset into Elasticsearch for a lexical search.
        Includes only the id, title, and article text.
        The article text is stripped down to language and punctuation.
        See more details at HuggingFace: https://huggingface.co/datasets/wikimedia/wikipedia

        This function should be called after download_wikipedia()
        """
        if self._dataset is None:
            logging.warning("Dataset not loaded into memory. Checking cache")
            if os.path.exists("../../data/wikimedia___wikipedia/"):
                self._dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en",  cache_dir="..\\..\\data")
            else:
                logging.error("Dataset is not downloaded. Did you call download_wikipedia() first?")
                return
        
        mappings = {
            "properties": {
                "title": {"type": "text"},
                "text": {"type": "text"}
            }
        }
        try:
            self._elastic_search_client.indices.create(index="wiki_index", mappings=mappings)
        except ConnectionError:
            logging.error("Could not connect to ElasticSearch server. Is it running?")
            return
        except exceptions.BadRequestError:
            logging.warn("Lexical index is already created. It will not be modified.")
        
        self._dataset.map(self._add_to_lexical_index, batched=True, batch_size=1000)

    def create_elasticsearch_snapshot(self):
        """Backs up the index to the configured repository.
        You must have configured a repository already before calling this method.
        https://www.elastic.co/guide/en/elasticsearch/reference/current/snapshots-filesystem-repository.html
        """
        repository_name = "searchrepo"
        # path.repo must be set in elasticsearch.yml, and the node must be restarted
        self._elastic_search_client.snapshot.create_repository(
            name = repository_name,
            body = {
                "type": "fs",
                'settings': {
                            'chunk_size': '1GB',
                            'compress': True,
                            'location': "snapshots"
                        }
            }
        )

        # Works in background. Check the ES console output for completion
        self._elastic_search_client.snapshot.create(
            repository=repository_name,
            indices="embedding_index",
            snapshot="embedding_point1percent"
        )

        self._elastic_search_client.snapshot.get(repository="searchrepo",snapshot="embeddeing_point1percent")

        self._elastic_search_client.cluster.put_settings(
            persistent={
                "cluster.routing.allocation.disk.watermark.low": "97%",
                "cluster.routing.allocation.disk.watermark.high": "98%",
                "cluster.routing.allocation.disk.watermark.flood_stage": "99%"
            }
        )

        """
        self._elastic_search_client.snapshot.restore(
            snapshot="embedding_point1percent",
            repository=repository_name
            indices="embedding_index"
        )
        """

@staticmethod
def clean_up_everything():
    """ Removes all data, models, and indexes.
    """
    #TODO implemnent clean_up_everything
    #_elastic_search_client.indices.delete(index="embedding_index")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    setup_instance = Setup()
    setup_instance.download_models()
    setup_instance.download_wikipedia()
    setup_instance.index_lexical()
    setup_instance.process_embeddings()
    setup_instance.index_embeddings()

    repository = setup_instance._elastic_search_client.snapshot.get_repository()

    # should return the embedding of the first paragraph of the first article in the data (Anarchy)
    elasticsearch_get_result = setup_instance._elastic_search_client.get(index="embedding_index", id="12.0")
    if elasticsearch_get_result is not None:
        logging.info("Getting a record from the Elasticsearch embedding index succeeded.")
    else:
        logging.error("Getting an embedding record from Elasticsearch search failed. Check that the index is created and the index has documents")

    # should return several embeddings with id == 12.*
    elasticsearch_search_embedding = setup_instance._elastic_search_client.search(
            index="embedding_index",
            query={
                "knn": {
                    "field": "embedding.predicted_value",
                    # TODO add a calibration embedding when using for real
                    "query_vector": setup_instance._sbert.encode("criticisms of anarchy")
                }
            }
        )
    
    if elasticsearch_search_embedding is not None:
        logging.info("Searching by embedding succeeded.")
    else:
        logging.error("Searching by embedding failed. Check that the model is downloaded and the Elasticsearch index was created correctly")