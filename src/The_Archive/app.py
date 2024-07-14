""" 
Copyright Abram Jackson 2024
All rights reserved
 """

from language_model import Langauge_Model

import logging

logging.getLogger().setLevel(logging.ERROR)

def display_answer(llm_result):
    """Displays the result of the LLM processing with citations
    
    In the future this function may output voice to the speaker.

    Arguments:
    llm_result: the response from the LLM chain
    """
    print(llm_result)

def execution_loop():
    """Runs the processing of the program"""
    while True:
        # unmodified text input. Later may get voice or correct spelling
        user_input = input()
        result = model.process_query(user_input)
        display_answer(result)

print("Please wait for the model and database to load...")
# Instantiate
# model = Langauge_Model(loaded_database, model_name= r"C:\Users\abram\.cache\lm-studio\models\rubra-ai\Phi-3-mini-128k-instruct-GGUF\rubra-phi-3-mini-128k-instruct.Q4_K_M.gguf")
model = Langauge_Model(model_name= r"C:\Users\abram\.cache\lm-studio\models\bartowski\Phi-3.1-mini-128k-instruct-GGUF\Phi-3.1-mini-128k-instruct-IQ4_XS.gguf")

print("Ready")
execution_loop()