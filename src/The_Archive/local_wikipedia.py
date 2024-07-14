import logging
from typing import List
from datasets import load_dataset

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from elasticsearch import Elasticsearch

class Local_Wikipedia(BaseRetriever):

    _hat = "hat"
    _dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:1%]",  cache_dir="data")
    _elastic_search_client = Elasticsearch("http://localhost:9200/")
    mappings = {
            "properties": {
                "title": {"type": "text"},
                "text": {"type": "text"}
            }
        }
#    _elastic_search_client.indices.create(index="wiki_index", mappings=mappings)
    
    def __init__(self):
        """ Downloads and loads EN Wikipdia, adds it to Elastic Search
        Loads HF's Wikipedia into a list Dataset, downloading if necessary
        https://huggingface.co/datasets/legacy-datasets/wikipedia"""
        super().__init__()

    def do_indexing(self):
        self._dataset.map(self._index_document)
    
    def _index_document(self, article: Document):
        # client.index(index="wiki_index", id=item["id"], document={"title": item["title"],text": item["text"]},) 
        self._elastic_search_client.index(
            index="wiki_index",
            id=article["id"],
            document=
            {
                "title": article["title"],
                "text": article["text"]
            }
        )        

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
        the_hits = search_results.body["hits"]["hits"]
        top_three = the_hits[:2]
        # TODO need to turn this into a dictionary of documents I think
        document_results = []
        
        for obj in top_three:
            new_obj = {'page_content': obj['_source']}
            document_results.append(new_obj)
        return document_results
    
    def _execute_elasticsearch_query(self, user_query):
        search_results = self._elastic_search_client.search(
            index="wiki_index", 
            query={ 
                "bool": {
                    "should": [
                        {
                            "fuzzy": {
                                "title": {
                                    "value": user_query, 
                                    "fuzziness": "AUTO", 
                                    "boost": 3}
                            }
                        }, 
                        {
                            "fuzzy": {
                                "text": {
                                    "value": user_query,
                                    "fuzziness": "AUTO"}
                            }
                        }
                    ]
                }
            }
        )
        return search_results
        
