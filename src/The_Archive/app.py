""" 
Copywright Abram Jackson 2024
All rights reserved
 """

from database import Database
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
loaded_database = Database()
model = Langauge_Model(loaded_database, model_name="Phi-3-mini-4k-instruct-q4.gguf")
print("Ready")
execution_loop()