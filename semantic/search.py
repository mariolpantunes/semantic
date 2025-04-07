# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import os
import gzip
import logging
import requests


logger = logging.getLogger(__name__)


class SearxNG:
    def __init__(self):
        self.url = 'https://search.hrun.mooo.com/'
    
    def search(self, query):
        page = 1
        snippets = []
        done = False
        total = total_count = 0
        while not done:
            querystring = {'q':query,'pageno':page,'format':'json', 'language':'en'}
            response = requests.request("GET", self.url, params=querystring) #headers=self.headers,

            previous_total = 0

            if response.ok:
                j = response.json()
                results= j['results']
                for r in results:
                    description = r['content']
                    snippets.append(description)
                previous_total = total
                total += len(results)
            else:
                done = True
            page += 1
            if total == previous_total:
                done = True
            logger.debug(f'({page}, {total} {previous_total})')

        return snippets


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
            #response = requests.request("GET", self.url, headers=self.headers, params=querystring)
            response = None
            status_code = 500

            if status_code == 200:
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
    def __init__(self, ws: SearxNG, path:str, limit:int=0):
        self.ws = ws
        self.path = path
        self.limit = limit
    
    def search(self, query):
        filename = f'{self.path}/{query}.csv'
        snippets = []
        logger.debug(f'Trying to get {filename}')
        if os.path.exists(filename):
            logger.debug('Cache file %s', filename)
            with gzip.open(filename, 'rt', newline='', encoding='utf-8') as file:
                snippets = file.readlines()
        else:
            logger.debug('Cache file %s does not exist...', filename)
            snippets=self.ws.search(query)
            logger.debug('Snippets loaded from Search Engine')
            with gzip.open(filename, 'wt', newline='', encoding='utf-8') as file:
                file.writelines(snippets)
            logger.debug('Snippets stored in %s', filename)

        if self.limit>0:
            return snippets[:self.limit]
        else:
            return snippets 
