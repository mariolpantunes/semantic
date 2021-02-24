# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import requests
import logging
import json


class CWS:
    def __init__(self, key):
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
            print('({}, {}, {}, {})'.format(page, total, total_count, total_count*0.2))

        return snippets