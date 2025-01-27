import requests
import json

datas = {
    'header' : {
        'interfaceID': 'AI-SDC-CAT-001',
        'interfaceMsg' : '메뉴 주문 추론',
        'storeNo': '123456'
    },
    'body': {
        'text' : '프랭크 버거 열 개 중에 세 개는 라지 나머지는 레귤러로'
    }        
    
}

headers = {
    'Content-Type': 'application/json; charset=utf-8'
}

# 우회주소 비버웍스에서 사용중이므로 테스트에 사용하지 말 것!
# url = 'https://c8db-203-229-206-42.ngrok-free.app/order'
url = 'http://127.0.0.1:5080/order'

datas = json.dumps(datas)
reponse = requests.post(url, data=datas, headers=headers)
server_return = reponse.json()
print('-'*50)
print(server_return)
print('-'*50)
