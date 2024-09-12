""" 
Copyright Abram Jackson 2024
All rights reserved
"""

import io
import logging
import os
import sys
import json

import datasets
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import yaml

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from src.The_Archive.prompt_templates import Prompt_Templates

class Generate_Questions:
    """ Creates a file with test questions based on the first 20 articles of Wikipedia.

    1. Gets the article text from the data
    2. Asks an LLM to create 5 questions for each
    3. Saves the results in JSONL
    """
    pass

    def __init__(self):
        self._elasticSearchClient = Elasticsearch("http://localhost:9200/")
        self._script_dir = os.getenv('PROJECT_ROOT')
        self.llm = LlamaCpp(
            model_path = r"C:\Users\abram\.cache\lm-studio\models\bartowski\Phi-3.5-mini-instruct-GGUF\Phi-3.5-mini-instruct-Q4_K_M.gguf",
            temperature=0.75,
            max_tokens=2000,
            top_p=1,
            verbose=False,
            n_ctx=16384,
            n_gpu_layers=100
        )
        templates = Prompt_Templates()
        self.essay_generation_prompt = templates.essay_generation_prompt()
        self.retrieval_generation_prompt = templates.retrieval_generation_prompt()

    def generate_retrieval_questions(self):
        """
        Instructs the LLM to generate factual questions based on a Wikipedia article
        Writes out JSONL.
        """
        articles = datasets.load_dataset(
            "wikimedia/wikipedia", 
            "20231101.en",
            cache_dir = os.path.join(self._script_dir, "data/"),
            split = 'train[:50]')
        generation_template = PromptTemplate.from_template(self.retrieval_generation_prompt)
        output_parser = StrOutputParser()
        questions_file_path = os.path.join(self._script_dir, "data/eval/retrieval_questions.jsonl")
        with open (questions_file_path, 'w', encoding='utf-8') as file:
            for i in range(len(articles)):
                questions =  (
                generation_template | 
                self.llm | 
                output_parser).invoke({
                    "article_id": articles['id'][i],
                    "article_title":articles['title'][i],
                    "article_text": articles['text'][i][:14000]})
                json_result = self.parse_json_questions(questions)
                for json_obj in json_result:
                    json_line = json.dumps(json_obj, ensure_ascii=False)
                    file.write(json_line + '\n')
    
    def jsonl_to_yaml_config(self, jsonl_file):
        yaml_data = {'tests': []}
        with open(jsonl_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                json_obj = json.loads(line)
                yaml_entry = {
                    'vars': {
                        'inquiry': json_obj['question']
                    },
                    'assert': [
                        {
                            'type': 'icontains',
                            'value': json_obj['answer']
                        }
                    ]
                }
                yaml_data['tests'].append(yaml_entry)
    
        # These need to be copied into the actual Promptfoo config, ./promptfooconfig.yaml
        # TODO do that in code
        with open("tests.yaml", 'w', encoding='utf-8') as outfile:
            yaml.dump(yaml_data, outfile, default_flow_style=False, sort_keys=False)

    def generate_essay_questions(self):
        articles = datasets.load_dataset(
            "wikimedia/wikipedia", 
            "20231101.en",
            cache_dir = os.path.join(self._script_dir, "data/"),
            split = 'train[:20]')
        generation_template = PromptTemplate.from_template(self.essay_generation_prompt)
        output_parser = StrOutputParser()
        questions_file_path = os.path.join(self._script_dir, "data/eval/essay_questions.jsonl")
        with open (questions_file_path, 'w', encoding='utf-8') as file:
            for i in range(len(articles)):
                questions =  (
                generation_template | 
                self.llm | 
                output_parser).invoke({
                    "article_id": articles['id'][i],
                    "article_title":articles['title'][i],
                    "article_text": articles['text'][i][:14000]})
                json_questions = self.parse_json_questions(questions)
                for json_obj in json_questions:
                    json_line = json.dumps(json_obj, ensure_ascii=False)
                    file.write(json_line + '\n')

    
    def parse_json_questions(self, LLM_output):
        """
        Attempts to parse the LLM-provided answers. Malformed JSONL or LLM commentary is dropped.
        """
        jsonl_list = []
        lines = LLM_output.strip().split('\n')
        error_count = 0
        for line in lines:
            try:
                # the LLM is liable to add spaces or commas to the lines
                line = line.lstrip(" \t\n\r\f\v,").rstrip(" \t\n\r\f\v,")
                # the prompt uses brackets to work around a langchain parsing bug
                if len(line) == 0:
                    raise ValueError("Invalid format")
                if line[0] == "[":
                    line = "{" + line[1:]
                if line[-1] == "]":
                    line = line[:-1] + "}"
                json_obj = json.loads(line)
                # essay question format
                if "article_id" in json_obj and "question" in json_obj:
                    # using an int instead of string to represent the ID is a common error
                    json_obj['article_id'] = str(json_obj['article_id'])
                    jsonl_list.append(json_obj)
                # retrieval questino format
                elif "question" in json_obj and "answer" in json_obj:
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
    generate_test_questions = Generate_Questions()
    #generate_test_questions.generate_essay_questions()
    generate_test_questions.generate_retrieval_questions()
    generate_test_questions.jsonl_to_yaml_config(os.path.join(generate_test_questions._script_dir, "data/eval/retrieval_questions.jsonl"))