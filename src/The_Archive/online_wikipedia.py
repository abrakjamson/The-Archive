""" 
Copywright Abram Jackson 2024
All rights reserved
 """
 
import logging
import requests

from bs4 import BeautifulSoup

from abstract_database import Abstract_Database


class Online_Wikipedia(Abstract_Database):

    def __init__(self):
        pass

    def index_documents(self, path):
        """No action. Later this could be a language of wikipedia or a category
        """
        pass

    def search(self, query):
        """Uses the en.wikipedia.org existing search engine and methodology to return the top 3 articles.
        """
        base_url = 'https://en.wikipedia.org/w/api.php'
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'srlimit': 1  # Limit the search results to the top 1 article
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            search_results = response.json().get('query', {}).get('search', [])
            top_articles = [result['title'] for result in search_results]
            top_page_text = self._get_page_text(top_articles[0])
            return top_page_text
        else:
            return f"An error occurred: {response.status_code}"
        
    def _get_page_text(self, page_title):
        """Gets the contents of a wikipedia article by title and converts it to markdown"""
        base_url = 'https://en.wikipedia.org/w/api.php'
        params = {
            'action': 'query',
            'titles': page_title,
            'format': 'json',
            'prop': 'extracts',
            'explaintext': 1,
            'formatversion': 2
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            # get the HTML into a mostly-text format, while keeping breaks and lists
 #           html_content = response.json().get('pages', {}).get('text', {}).get('*', '')
            extract_content = response.json()['query']['pages'][0]['extract']
            return extract_content[:8192]
            """
            soup = BeautifulSoup(html_content, 'html.parser')
            text = ''
            for e in soup.descendants:
                if isinstance(e, str):
                    text += e
                elif e.name in ['br',  'p', 'h1', 'h2', 'h3', 'h4']:
                    text += '\n'
                elif e.name == 'li':
                    text += '\n- '
            return text
            """
        else:
            return f"An error occurred: {response.status_code}"