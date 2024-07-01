""" 
Copywright Abram Jackson 2024
All rights reserved
 """

import logging

class Database:
    def __init__(self):
        logging.info("Database init")
    
    def search_by_keyword(self, keyword):
        print("Keyword search:" + keyword) # placeholder