""" 
Copyright Abram Jackson 2024
All rights reserved
 """

import logging

from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.llms import LlamaCpp
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from ..The_Archive.local_wikipedia import Wikipedia_Lexical, Wikipedia_Semantic
from ..The_Archive.prompt_templates import Prompt_Templates

class Language_Model():

    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def __init__(self, model_name=r"C:\Users\abram\.cache\lm-studio\models\bartowski\Phi-3.5-mini-instruct-GGUF\Phi-3.5-mini-instruct-Q4_K_M.gguf", ):
        logging.debug("LLM init with model: " + model_name)
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
        logging.debug("Model is loaded")
        
        ArchivePrompts = Prompt_Templates() # can specify different model token formats here
        self.search_prompt = ArchivePrompts.search_prompt()
        self.context_prompt = ArchivePrompts.context_prompt()
        self.answer_prompt = ArchivePrompts.answer_prompt()

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
            logging.debug(article['page_content']['title'])
        
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
    debug_language_model = Language_Model(model_name= r"C:\Users\abram\.cache\lm-studio\models\bartowski\Phi-3.5-mini-instruct-GGUF\Phi-3.5-mini-instruct-Q4_K_M.gguf")
    answer = debug_language_model.process_query("criticisms of anarchy")
    #answer = debug_language_model.process_query(r"In contrast, Edmund Burke's 1756 work A Vindication of Natural Society, argued in favour of anarchist society in a defense of the state of nature. Burke insisted that reason was all that was needed to govern society and that \"artificial laws\" had been responsible for all social conflict and inequality, which led him to denounce the church and the state. Burke's anti-statist arguments preceded the work of classical anarchists and directly inspired the political philosophy of William Godwin.")
    print(answer)