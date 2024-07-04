""" 
Copywright Abram Jackson 2024
All rights reserved
 """

import logging
from abc import ABC, abstractmethod

class Abstract_Database(ABC):
    def __init__(self):
        logging.info("Database init")
    
    @abstractmethod
    def index_documents(self, path):
        pass

    @abstractmethod
    def search(self, query):
        pass