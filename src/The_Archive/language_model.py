""" 
Copyright Abram Jackson 2024
All rights reserved
 """

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
)
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.llms import LlamaCpp
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

import logging

from local_wikipedia import Local_Wikipedia

class Langauge_Model():

    def __init__(self, model_name="test", ):
        logging.info("LLM init with model: " + model_name)
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])        
        self.llm = LlamaCpp(
            model_path=model_name,
            temperature=0.75,
            max_tokens=2000,
            top_p=1,
#            callback_manager=callback_manager,
            verbose=False,
            n_ctx=8192,
            n_gpu_layers=100
        ) 
        self.search_prompt = """<|system|>Give a comma-separated list of search keywords most likely to find an answer to the user's question. Do NOT answer the question.<|end|>\n<|user|>When was Obama born?<|end|>\n<|assistant|>Barack Obama,United States Presidents,Family of Barack Obama<|end|>\n<|user|>How can I make charcoal?<|end|>\n<|assistant|>Charcoal,Charcoal Kiln,Retort (Chemistry)<|end|>\n<|user|>{user_question}<|end|>\n<|assistant|>"""
        self.context_prompt = """<|system|>Directly answer the user's question without any extra information. Be as concise as possible. Use only the following context information for the user's question\n{context}<|end|>
<|user|>{user_question}<|end|>
<|assistant|>"""

    def process_query(self, query):
        """Primary funtion to process the user's query with database RAG and LLM.
        
        Arguments:
        query: A string of the user's prompt
        
        returns a string of the LLM's response"""
        search_template = PromptTemplate.from_template(self.search_prompt)
        context_template = PromptTemplate.from_template(self.context_prompt)
        # use this line to use online wikipedia search instead
        # retriever = WikipediaRetriever(lang="en", doc_content_chars_max=10000, top_k_results=2)
        retriever = Local_Wikipedia()
        retriever.do_indexing() #TODO move to setup.py
        output_parser = StrOutputParser()

        # Find the best Wikipedia articles to answer the question
        logging.info("Suggested Wikipedia searches:\n")
        search_results = (
            search_template | 
            self.llm | 
            retriever).invoke(query)
        context_results = ''.join(str(d)[:12000] for  d in search_results) # list of dicts -> single string
        for article in search_results:
            a = article
            b = a['page_content']
            c = b['title']
            logging.info(article['page_content']['title'])
        # Answer the question using the Wikipedia articles
        logging.info("Generating answer:\n")
        answer = (
            context_template | 
            self.llm | 
            output_parser).invoke({"context": context_results, "user_question": query})
        return f"Answer:\n {answer}"