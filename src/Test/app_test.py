""" 
Copyright Abram Jackson 2024
All rights reserved
"""

import sys
import os
from typing import Dict, Any
import logging
import json

# TODO finish sorting out paths
if 'PROJECT_ROOT' not in os.environ:
    os.environ['PROJECT_ROOT'] = 'C:/Users/abram/Documents/The-Archive'
# Promptfoo will call this file as a script, so we need to add the source into sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.The_Archive.language_model import Language_Model
from src.The_Archive.prompt_templates import Prompt_Templates


def call_api(prompt, options: Dict[str, any], context):
    """
    Tests The Archive's reasoning loop without application surroundings.
    Exptected to be run by Promptfoo.
    A prompt to test is required.
    Available options are 'prompt type' and 'model'

    - 'prompt type' one of 'lexical', 'context', or 'synthesis'
    - 'model' is a Huggingface model path
    """

    # TODO Explore using scenarios or different provider functions instead of this method
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

    # The user's actual question is in the configuration
    inquiry = context['vars'].get('inquiry', "No user question found.")
    output = archive.process_query(inquiry)

    result = {
        "output": output
    }
    print(result)
    return result

if __name__ == "__main__":
    # test functionality with same prompt
    templates = Prompt_Templates()
    prompt_to_test = templates.search_prompt()
    model_name= r"C:\Users\abram\.cache\lm-studio\models\bartowski\Phi-3.5-mini-instruct-GGUF\Phi-3.5-mini-instruct-Q4_K_M.gguf"
    call_api(templates.search_prompt(), {"prompt type": "lexical", "model": model_name}, context= None)
