import io
import logging
import os
import sys
import json

import datasets
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser


class Generate_Essay_Quesitions:
    """ Creates a file with test questions based on the first 20 articles of Wikipedia.

    1. Gets the article text from the data
    2. Asks an LLM to create 5 questions for each
    3. Saves the results in JSONL
    """
    pass

    def __init__(self):
        self._elasticSearchClient = Elasticsearch("http://localhost:9200/")
        self._script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.llm = LlamaCpp(
            model_path = r"C:\Users\abram\.cache\lm-studio\models\bartowski\Phi-3.5-mini-instruct-GGUF\Phi-3.5-mini-instruct-Q4_K_M.gguf",
            temperature=0.75,
            max_tokens=2000,
            top_p=1,
            verbose=False,
            n_ctx=16384,
            n_gpu_layers=100
        )
        self.generation_prompt = """<|system|>
Give five JSON Lines of essay questions based on the article text.
The entire response should be formatted as JSONL without any additional commentary.
If the article is unsuitable for essay questions, return only a newline.<|end|>
<|user|>
{{"article_id": "12", "text": "Anarchy is a form of society without... (full text omitted from conversation history)"}}<|end|>
<|assistant|>
{{"article_id": "12", "question": "Describe three major criticisms of Anarchy"}}
{{"article_id": "12", "question": "What are the origins of the word 'anarchy'?"}}
{{"article_id": "12", "question": "Why is anarchy considered a social disorder?"}}
{{"article_id": "12", "question": "Describe how John Calvin compared anarchy to tyranny and why."}}
{{"article_id": "12", "question": "What is the 'state of nature' according to John Locke?"}}<|end|>
<|user|>
{{"article_id": {article_id}, "title": {article_title}, "text": {article_text}}}<|end|>
<|assistant|>"""

    def generate(self):
        articles = datasets.load_dataset(
            "wikimedia/wikipedia", 
            "20231101.en",
            cache_dir = os.path.join(self._script_dir, "../../data/"),
            split = 'train[:20]')
        generation_template = PromptTemplate.from_template(self.generation_prompt)
        output_parser = StrOutputParser()
        questions_file_path = os.path.join(self._script_dir, "../../data/eval/questions.jsonl")
        with open (questions_file_path, 'w') as file:
            for i in range(len(articles)):
                questions =  (
                generation_template | 
                self.llm | 
                output_parser).invoke({
                    "article_id": articles['id'][i],
                    "article_title":articles['title'][i],
                    "article_text": articles['text'][i][:14000]})
                json_questions = self.parse_questions(questions)
                for json_obj in json_questions:
                    json_line = json.dumps(json_obj, ensure_ascii=False)
                    file.write(json_line + '\n')

    
    def parse_questions(self, LLM_output):
        """Attempts to parse the LLM-provided answers. Malformed JSONL or LLM commentary is dropped"""
        jsonl_list = []
        lines = LLM_output.strip().split('\n')
        error_count = 0
        for line in lines:
            try:
                # the LLM is liable to add spaces or commas to the lines
                line = line.lstrip(" \t\n\r\f\v,").rstrip(" \t\n\r\f\v,")
                json_obj = json.loads(line)
                if "article_id" in json_obj and "question" in json_obj:
                    # using an int instead of string to represent the ID is a common error
                    json_obj['article_id'] = str(json_obj['article_id'])
                    jsonl_list.append(json_obj)
                else:
                    raise ValueError("Invalid format")
            except json.JSONDecodeError:
                error_count += 1
            except ValueError:
                error_count += 1
        logging.debug(f"Found {len(jsonl_list)} rows of valid JSON in {len(lines)} rows of LLM output")
        return jsonl_list

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    generate_test_questions = Generate_Essay_Quesitions()
    generate_test_questions.generate()