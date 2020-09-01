import requests
import json
import pandas as pd

url = 'http://0.0.0.0:5000/api/'

data = pd.read_csv('data/test_processed.csv', sep=';').sample(n=5).drop(columns='Survived').values.tolist()
j_data = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print(r, r.text)