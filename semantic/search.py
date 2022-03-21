# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import os
import logging
import requests


logger = logging.getLogger(__name__)


class CWS:
    def __init__(self, key:str):
        self.key = key
        self.url = 'https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/Search/WebSearchAPI'
        self.headers  = {'x-rapidapi-key': key,'x-rapidapi-host': 'contextualwebsearch-websearch-v1.p.rapidapi.com'}
    
    def search(self, query):
        page = 1
        snippets = []
        done = False
        total = total_count = 0
        while not done:
            querystring = {'q':query,'pageNumber':page,'pageSize':50,'autoCorrect':True}
            response = requests.request("GET", self.url, headers=self.headers, params=querystring)
            
            if response.status_code == 200:
                j = response.json()
                total_count = j["totalCount"]
                values= j['value']
                for value in values:
                    description = value['description']
                    snippets.append(description)
                total += len(values)
            else:
                done = True
            page += 1
            if total >= total_count*0.2:
                done = True
            logger.debug('(%s, %s, %s, %s)', page, total, total_count, int(total_count*0.2))

        return snippets


class CacheSearch:
    def __init__(self, ws, path):
        self.ws = ws
        self.path = path
    
    def search(self, query):
        filename = f'{self.path}/{query}.csv'
        snippets = []
        if os.path.exists(filename):
            logger.debug('Cache file %s', filename)
            with open(filename, 'rt', newline='', encoding='utf-8') as file:
                snippets = file.readlines()
        else:
            logger.debug('Cache file %s does not exist...', filename)
            snippets=self.ws.search(query)
            logger.debug('Snippets loaded from Search Engine')
            with open(filename, 'wt', newline='', encoding='utf-8') as file:
                file.writelines(snippets)
            logger.debug('Snippets stored in %s', filename)
        return snippets 
