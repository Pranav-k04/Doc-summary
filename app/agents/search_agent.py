import requests

def search_papers(query):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": 5, "fields": "title,url"}
    response = requests.get(url, params=params)
    data = response.json()
    return data.get("data", [])
