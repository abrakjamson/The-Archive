""" 
Copyright Abram Jackson 2024
All rights reserved
 """


import logging
from typing import List
import datasets
from datasets import load_dataset
import os
import sys

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

class Wikipedia_Lexical(BaseRetriever):
    """ Makes a search by keyword to ElasticSearch """   

    _elastic_search_client = Elasticsearch("http://localhost:9200/")

    def __init__(self):
        super().__init__()

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """ Searches local Wikipedia pages in Elastic search for the invoked string.
        Should be used with Langchain's invoke mechanism instead of being called directly. """

        # Exact match
        # client.search(index="wiki_index", query={"match": {"text": {"query": "Anarchis*"}}})
        # Fuzzy search 
        # client.search(index="wiki_index", query={"fuzzy": {"text": {"value": "anarchist"}}})
        # Weighting the title more
        # client.search(index="wiki_index", query={"multi_match": {"query": "Anarchy", "fields": ["title^3", "text"]}})
        # Fuzzy and weighted title (combine two should cluases in a boolean expression, with auto-configured fuzziness and title weighted 3x)
        # client.search(index="wiki_index", query={ "bool": {"should": [{"fuzzy": {"title": {"value": "anarchist", "fuzziness": "AUTO", "boost": 3}}}, {"fuzzy": {"text": {"value": "anarchist","fuzziness": "AUTO"}}}]}})
        search_results = self._execute_elasticsearch_query(query)
        the_hits = search_results.body["hits"]["hits"][:2]
        document_results = []
        for obj in the_hits:
            new_obj = {'page_content': obj['_source']}
            document_results.append(new_obj)
        return document_results
    
    def _execute_elasticsearch_query(self, user_query):
        search_results = self._elastic_search_client.search(
            index="wiki_index", 
            query={
                "multi_match":
                {
                    "query": user_query, 
                    "fields": ["title^3", "text"]
                }
            }
        )
        return search_results
    
class Wikipedia_Semantic(BaseRetriever):
    """ Makes a semantic query to Elasticsearch """

    _elastic_search_client = Elasticsearch("http://localhost:9200/")
    _script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    _sbert = SentenceTransformer(
        "avsolatorio/GIST-small-Embedding-v0",
        cache_folder=os.path.join(_script_dir, "../../models/"))
    
    # Load a few documents to provide calibration embeddings
    _calibration_articles = datasets.load_dataset(
        "wikimedia/wikipedia", 
        "20231101.en",  
        cache_dir = os.path.join(_script_dir, "../../data/"),
        split='train[:20]')
    _calibration_paragraphs = [sub for string in _calibration_articles['text'] for sub in string.split("\n\n")]
    _calibration_embeddings = _sbert.encode(_calibration_paragraphs, precision="int8")
  
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """ Searches the embedding_index of Elasticsearch with cosine similarity of the query
            Should be invoked through Langchain invoke. """
        """
        query_embedding = self._sbert.encode(query)
        quantized_embedding = quantize_embeddings(query_embedding,calibration_embeddings=self._calibration_embeddings, precision='int8')
        """
        embeds = datasets.load_dataset(
            "wikimedia/wikipedia", 
            "20231101.en",  
            cache_dir=os.path.join(self._script_dir, "../../data/"),
            split='train[:20]')
        paragraphs = [sub for string in embeds['text'] for sub in string.split("\n\n")]
        paragraphs.insert(0, query)
        quantized_embedding = self._sbert.encode(paragraphs, precision="int8")
        search_results = self._elastic_search_client.search(
            index="embedding_index",
            query={
                "knn": {
                    "field": "embedding.predicted_value",
                    "query_vector": quantized_embedding[0],
                }
            }
        )
        the_hits = search_results.body["hits"]["hits"][:10]
        document_results = []
        for obj in the_hits:
            new_obj = {'page_content': self.paragraph_id_to_paragraph(obj['_id'])}
            logging.info(obj['_id'])
        return document_results

    def paragraph_id_to_article(self, paragraph_id):
        """ Gets the entire article of the specified paragraph ID"""
        article = self._elastic_search_client.get(
            index="wiki_index",
            id = paragraph_id.split('.')[0]
        )
        return article.body
    
    def paragraph_id_to_paragraph(self, paragraph_id):
        """ Gets the text of the paragraph for the specified paragraph ID"""
        es_record = self.paragraph_id_to_article(paragraph_id)
        article_text = es_record['_source']['text']
        logging.info(es_record['_source']['title'])
        paragraphs = article_text.split('\n\n')
        paragraph_number = int( paragraph_id.split('.')[1] )
        return paragraphs[paragraph_number]
        
if __name__ == "__main__":
    wikipedia_semantic = Wikipedia_Semantic()
    search_results = wikipedia_semantic._get_relevant_documents("criticisms of anarchy")
    print(search_results)

    
from elasticsearch import Elasticsearch
_elastic_search_client = Elasticsearch("http://localhost:9200/")
search_results = _elastic_search_client.get(
    index="embedding_index",
    id = "12.19")
print(search_results)
_elastic_search_client.indices.get(
    index="embedding_index",
)
_elastic_search_client.count(
    index="embedding_index",
)