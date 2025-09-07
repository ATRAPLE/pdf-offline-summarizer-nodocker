import requests

url = "http://localhost:11434/api/generate"
data = {
    "model": "qwen2.5:7b-instruct",
    "prompt": "Olá"
}
response = requests.post(url, json=data)
print(response.text)
