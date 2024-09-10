with open("logfile.log", "a") as log_file: log_file.write("running app_test.py\n")
print("running app_test.py")

import sys
import os
from typing import Dict, Any

# Promptfoo will call this file as a script, so we need to add the source into sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
with open("logfile.log", "a") as log_file: log_file.write("importing language model\n")
from src.The_Archive.language_model import Language_Model
with open("logfile.log", "a") as log_file: log_file.write("importing prompt templates\n")
from src.The_Archive.prompt_templates import Prompt_Templates

import json


def call_api(prompt, options: Dict[str, any], context):
    """
    Tests The Archive's reasoning loop without application surroundings.
    A prompt to test is required.
    Available options are 'prompt type' and 'model'

    - 'prompt type' one of 'lexical', 'context', or 'synthesis'
    - 'model' is a Huggingface model path
    """
    with open("logfile.log", "a") as log_file: log_file.write("running call_api\n")
    print("running call_api")
    prompt_type = options.get('prompt type', None)
    model = options.get('model', None)
    if model is not None:
        archive = Language_Model(model_name = model)
    else:
        archive = Language_Model()

    # change the default prompt to the prompt to test
    if prompt_type == "lexical":
        archive.search_prompt = prompt
    elif prompt_type == "context":
        archive.context_prompt = prompt
    elif prompt_type == "synthesis":
        archive.answer_prompt = prompt

    output = archive.process_query(prompt)

    result = {
        "output": output
    }
    print(result)
    return result

if __name__ == "__main__":
    # null test of just functionality
    with open("logfile.log", "a") as log_file: log_file.write("running script directly\n")
    print("running script directly")
    templates = Prompt_Templates()
    prompt_to_test = templates.search_prompt()
    model_name= r"C:\Users\abram\.cache\lm-studio\models\bartowski\Phi-3.5-mini-instruct-GGUF\Phi-3.5-mini-instruct-Q4_K_M.gguf"
    call_api(templates.search_prompt(), {"prompt type": "lexical", "model": model_name}, context= None)

with open("logfile.log", "a") as log_file: log_file.write("end of app_test.py\n")