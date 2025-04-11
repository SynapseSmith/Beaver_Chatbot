import requests

# 전송할 텍스트 메시지
text_message = "프랭크 버거 25개 중에 12개는 라지, 나머지는 레귤러"

# 서버 URL
url = 'http://127.0.0.1:5080/order'

# 텍스트 메시지를 보낼 때는 'data' 또는 'json' 파라미터를 사용할 수 있습니다.
# 여기서는 'json' 파라미터를 사용하여 JSON 형식으로 데이터를 전송합니다.
data = {
    'message': text_message
}

response = requests.post(url, json=data)

# 응답 출력
server_return = response.json()
print('-'*50)
print(server_return)
print('-'*50)