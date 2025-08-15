import os

import requests
import json

from transformers import BartTokenizer

Baseurl = "https://api.claude-Plus.top"

Skey = ""

url = Baseurl + "/v1/chat/completions"
headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {Skey}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'}



def getAnswer(modelname,query):
    payload = json.dumps({
        "model": modelname,
        "messages": [
            {
                "role": "system",
                "content": query
            },
        ]
    })


    response = requests.request("POST", url, headers=headers, data=payload)

    data = response.json()

    content = data

    print(content)
    return content['choices'][0]['message']['content']
