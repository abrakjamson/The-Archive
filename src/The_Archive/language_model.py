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

from local_wikipedia import Wikipedia_Lexical, Wikipedia_Semantic

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
        # newlines for phi go after the speaker and after the end token
        self.search_prompt = """<|system|>
        Give a comma-separated list of search keywords most likely to find an answer to the user's question. Do NOT answer the question.<|end|>
        <|user|>
        When was Obama born?<|end|>
        <|assistant|>
        Barack Obama,United States Presidents,Family of Barack Obama<|end|>
        <|user|>
        How can I make charcoal?<|end|>
        <|assistant|>
        Charcoal,Charcoal Kiln,Retort (Chemistry)<|end|>
        <|user|>
        {user_question}<|end|>
        <|assistant|>
        """
        self.context_prompt = """<|system|>
        Directly answer the user's question.
        Use only the following context information for the user's question:
        {context}<|end|>
        <|user|>
        {user_question}<|end|>
        <|assistant|>
        """

        self.answer_prompt = """<|system|>
        Directly answer the user's question based on two other assistants' attempted answers.
        Answer only based on the two assistants' answers. First assistant's answer:
        {lexical_context}
        
        Second assistant's answer:
        {semantic_context}<|end|>
        <|user|>
        {user_question}<|end|>
        <|assistant|>
        """

    def process_query(self, query):
        """Primary funtion to process the user's query with database RAG and LLM.
        
        Arguments:
        query: A string of the user's prompt
        
        returns a string of the LLM's response"""
        search_template = PromptTemplate.from_template(self.search_prompt)
        context_template = PromptTemplate.from_template(self.context_prompt)
        combined_template = PromptTemplate.from_template(self.answer_prompt)
        # use this line to use online wikipedia search instead
        # retriever = WikipediaRetriever(lang="en", doc_content_chars_max=10000, top_k_results=2)
        retriever = Wikipedia_Lexical()

        output_parser = StrOutputParser()

        # Lexical: Find the best Wikipedia articles to answer the question
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
        
        # Answer the question using the lexical Wikipedia articles
        logging.info("Generating lexical answer\n")
        lexical_answer = (
            context_template | 
            self.llm | 
            output_parser).invoke({"context": context_results, "user_question": query})
        logging.info(lexical_answer)

        # Semantic: Find the best Wikipediaparagraps to answer the question
        retriever = Wikipedia_Semantic()
        semantic_paragraphs = retriever.invoke(query)
        
        # Answer the question using the semantic Wikipedia paragraphs
        logging.info("Generating semantic answer\n")
        semantic_answer = (
            context_template | 
            self.llm | 
            output_parser).invoke({"context": semantic_paragraphs, "user_question": query})
        logging.info(semantic_answer)
        
        # Summarize from the best of the lexical and semantic answers
        combined_answer = (
            combined_template |
            self.llm |
            output_parser).invoke({
                "lexical_context": lexical_answer,
                "semantic_context": semantic_answer,
                 "user_question": query })
        logging.info(combined_answer)

        return f"Answer:\n {combined_answer}"
    
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    debug_language_model = Langauge_Model(model_name= r"C:\Users\abram\.cache\lm-studio\models\lmstudio-community/gemma-2-2b/gemma-2-2b-it-Q8_0.gguf")
    answer = debug_language_model.process_query("Why did communism fail in China?")
    print(answer)