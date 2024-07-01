""" 
Copywright Abram Jackson 2024
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
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

import logging
from database import Database

class Langauge_Model():

    def __init__(self, database, model_name="test", ):
        logging.info("LLM init with model: " + model_name)
        self._database = database
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = LlamaCpp(
            model_path="C:/Users/abram/.cache/lm-studio/models/microsoft/Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf",
            temperature=0.75,
            max_tokens=2000,
            top_p=1,
            callback_manager=callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
        )

    def process_query(self, query):
        """Primary funtion to process the user's query with database RAG and LLM.
        
        Arguments:
        query: A string of the user's prompt
        
        returns a string of the LLM's response"""
        self._database.search_by_keyword(query)

        template = """<|system|>\n
        Directly answer the user's question without any extra information.
        Be as concise as possible.<|end|>\n
        <|user|>\n{user_question}<|end|>\n
        <|assistant|>\n"""
        prompt_template = PromptTemplate.from_template(template)
        constructed_prompt = prompt_template.format(user_question=query)
        
        self.llm.invoke(constructed_prompt)

        return ("User query was: \n" + query)
    