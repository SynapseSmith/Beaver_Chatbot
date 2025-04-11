import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from flask import Flask, request, Response, stream_with_context
import asyncio
import time
import pandas as pd
import json
import pickle
import ast
import logging
import copy
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from threading import Thread

# Initialize Flask app
app = Flask(__name__)

def load_logger(log_dir, log_level):
    logger = logging.getLogger(__name__)
    if log_level == 'INFO':
        lv = logging.INFO
    elif log_level == 'ERROR':
        lv = logging.ERROR
    elif log_level == 'DEBUG':
        lv = logging.DEBUG
    else:
        raise NotImplementedError
    logger.setLevel(lv)

    formatter = logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s] :: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir, encoding='utf-8-sig')
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

# Set logger
curtime = time.strftime("%Hh-%Mm-%Ss")
date = time.strftime("%Y-%m-%d")
log_folder = os.path.join('/home/user09/beaver/prompt_engineering/log', date)
if not os.path.exists(log_folder):
    os.mkdir(log_folder)

logdir = os.path.join(log_folder, curtime + '.log')
logger = load_logger(logdir, 'INFO')
logger.info(f'*** {curtime} START ***')
logger.info(f'*** PID: {os.getpid()} ***')

class args:
    store = '프랭크버거'
    output_dir = "/home/user09/beaver/log/"
    llama_v1_model_name_or_path = "/home/user09/beaver/log/sft/24-08-04_11-42-33_final/checkpoint-500"
    instruction_template = "/home/user09/beaver/instruction_templates/sft_v0.7.txt"
    embedding_model = "BAAI/bge-m3"
    retriever_k = 4
    retriever_bert_weight = 0.7
    cache_dir = "/nas/.cache/huggingface"
    model_revision = "main"
    config_name = None

def read_db(output_dir, db_type, name, hf=None):
    if db_type == "faiss":
        return FAISS.load_local(f"{output_dir.split('log')[0]+'data'}/db/{name}_faiss4", hf, allow_dangerous_deserialization=True)
    elif db_type == "docs":
        with open(f"{output_dir.split('log')[0]+'data'}/db/{name}_docs4.pkl", "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("db_type should be either faiss or docs")

def get_retriever(db, docs):
    db_reteriver = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": args.retriever_k},
    )
    docs_reteriver = BM25Retriever.from_documents(docs)
    docs_reteriver.k = args.retriever_k
    retriever = EnsembleRetriever(
        retrievers=[db_reteriver, docs_reteriver],
        weights=[args.retriever_bert_weight, 1 - args.retriever_bert_weight],
    )
    return retriever

def format_docs(docs):
    return "\n".join(f"- {doc}".replace('"', '') for doc in docs[:args.retriever_k])

def invoke_format(example):
    before_slot = example['before_slot']
    text1 = " ".join([i.split('\'')[1] for i in before_slot.split("상품명': ")[1:]])
    text2 = "" if example['before_input'] == "이전 대화 없음" else example['before_input']
    text3 = example['current_input']
    text = text1 + " " + text2 + " " + text3
    return text

encode_kwargs = {'normalize_embeddings': True}
model_kwargs = {'device': 'cpu'}
hf = HuggingFaceBgeEmbeddings(
    model_name=args.embedding_model,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

retriever_dict = {
    "프랭크버거": get_retriever(read_db(args.output_dir, "faiss", "프랭크버거", hf), read_db(args.output_dir, "docs", "프랭크버거"))
}

menu_df = pd.read_csv("/home/user09/beaver/data/dataset_v11.0.csv")

# 각 매장별로 모든 메뉴를 딕셔너리로 저장.
all_menus_dict = {
    "프랭크버거": menu_df[menu_df['store'] == "프랭크버거"]['all_menus'].values[0]
}

def initialize_model():
    global prompt_chain, llm_chain, streamer, model  # 함수 밖에서도 변수의 유효범위를 유지하기 위해, 전역변수로 선언.

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.llama_v1_model_name_or_path,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.llama_v1_model_name_or_path)

    # streamer = TextStreamer(tokenizer, skip_prompt=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)  # 실시간으로 중간 결과를 출력하는 스트리머.

    model = AutoModelForCausalLM.from_pretrained(
        args.llama_v1_model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        device_map="auto",
    )

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        streamer=streamer,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        max_new_tokens=400
    )

    llm_chain = HuggingFacePipeline(pipeline=text_generation_pipeline)  # 허깅페이스의 pipeline을 감싸는 langchain의 wrapper

    with open(args.instruction_template, "r") as f:
        prompt_template = f.read()

    prompt_chain = PromptTemplate(  # 프롬프트 템플릿을 채워주는 역할.
        input_variables=["store", "menu_info", "side_info", "retriever", "before_input", "before_slot", "before_response", "current_input"],
        template=prompt_template
    )

class InputHistory:  # 이전 대화, 이전 슬롯, 이전 응답, 현재 대화를 보관하는 클래스.
    before_input = None
    before_slot = None
    before_response = None
    current_input = None

def initialize_history():
    global endoforder, showslot, logger
    logger.info(' ** Initialize history **')
    showslot = False
    endoforder = False
    InputHistory.before_input = "이전 대화 없음"
    InputHistory.before_slot = "{'매장포장여부': None, '주문내역': [None], '결제수단': None}"
    InputHistory.before_response = "이전 대화 없음"
    InputHistory.current_input = None

def update_history(response):
    updated_before_input = InputHistory.current_input
    updated_before_response, updated_before_slot = response.split("현재 주문슬롯: ")
    updated_before_response = updated_before_response.split("현재 응대: ")[1].strip()
    updated_before_slot = updated_before_slot.split("<|eot_id|>")[0].strip()

    InputHistory.before_input = updated_before_input
    InputHistory.before_slot = updated_before_slot
    InputHistory.before_response = updated_before_response

def run_llm_chain(message):  # Thread의 target 매개변수로 입력되는 함수.
    return llm_chain.invoke(message)  # 입력된 message를 langchain 파이프라인을 통해 처리해, 언어 모델로부터 출력 생성.

is_first = True # 응대 문구의 첫 토큰인지 여부
is_initial = False  # 초기화 되었는지 여부
showslot = False # 주문 종료 전 슬롯 확인 용도
endoforder = False  # 주문 종료 여부

args.model_name_or_path = args.llama_v1_model_name_or_path

initialize_model()  # config, tokenizer, streamer, model, pipline, langchain wrapper, prompt template 한꺼번에 초기화
initialize_history()  # 이전 대화, 이전 슬롯, 이전 응답, 현재 대화 초기화

code_info = pd.read_excel('/home/user09/beaver/data/code_info.xlsx')
code_info = code_info.set_index('prodNM')['code'].to_dict()

option_code_info = pd.read_excel('/home/user09/beaver/data/option_code_info.xlsx')
option_code_info = option_code_info.set_index('optionprodGrpNm')['optionprodGrpCd'].to_dict()

with open('/home/user09/beaver/data/dataset/menu_info_converted.json', 'r', encoding='utf-8') as f:
    menu_info = json.load(f)

with open('/home/user09/beaver/data/dataset/side_info_converted.json', 'r', encoding='utf-8') as f:
    side_info = json.load(f)

pay_dict = {'99': '미납', 'BP': '비버페이', 'CA': '현금', 'CD': '카드', 'CP': '매장쿠폰', 'DP': '선불권', 
            'ET': '기타결제', 'GV': '모바일상품권', 'MB': '회원결제', 'MP': '만나서결제', 'PC': '간편결제', 'PO': '포인트'}
pay_inv_dict = {v: k for k, v in pay_dict.items()}

easy_pay_dict = {'A':'알리페이' , 'B':'비플페이', 'C':'앱카드', 'D':'PAYON', 'E':'페이북', 'F':'EMV', 'H':'삼성페이', 'I':'식권대장', 
            'J':'앱결제', 'K':'카카오페이', 'L':'올리브식권', 'M':'SMS', 'N':'네이버페이', 'O':'OCR', 'P':'페이코', 'R':'일반',
            'T':'단말기결제', 'U':'만나서결제', 'V':'기타', 'W':'위챗페이', 'Z':'제로페이'}
easy_pay_inv_dict = {v: k for k, v in easy_pay_dict.items()}

deliver_dict = {'포장': 'P', '매장': 'S', '배달': 'D'}

@app.route('/order', methods=['POST'])
def order():
    try: 
        global is_initial, endoforder, showslot, model, llm_chain, logger, pay_inv_dict, easy_pay_inv_dict, deliver_dict
        
        payload = request.get_json(force=True)
        header = payload['header']
        body = payload['body']
        message = body['text']
        
        logger.info(payload)
        
        response_dict = dict()
        response_dict['header'] = header
        response_dict['body'] = dict()
        
        is_response_yielding = True
        
        # @stream_with_context
        def generate(is_response_yielding):
            global is_initial, endoforder, showslot, is_first, menu_info, side_info
            
            if is_initial == False:
                is_initial = True
                logger.info(' --- Start order ----')
            
            if is_initial and message == '초기화':
                initialize_history()
                return

            InputHistory.current_input = message
            
            inputs = {
                'store': args.store,
                'menu_info': menu_info,
                'side_info': side_info,
                'before_input': InputHistory.before_input,
                'before_slot': InputHistory.before_slot,
                'before_response': InputHistory.before_response,
                'current_input': InputHistory.current_input
            }
            
            retriever = retriever_dict[inputs['store']]
            retriever_result = format_docs([f"{i.page_content}: {i.metadata['옵션']}" for i in retriever.invoke(invoke_format(inputs))])
            inputs['retriever'] = retriever_result

            start_time = time.time()
            model_input = prompt_chain.invoke(inputs)

            t= Thread(target=run_llm_chain, args=(model_input,))
            t.start()
            
            history = ""
            slot = None
            new_slot = {}
            response_text = ''
            yield_start_time = time.time()
            start = False
            for new_text in streamer:
                text_for_yield = None
                if not start:
                    start = True
                    print(time.time() - yield_start_time)
                 
                if not new_text:
                    continue

                history += new_text
                    
                if '응대:' in history and is_response_yielding:
                    response_text += new_text
                    streaming_text = new_text.replace('응대: ', '')
                    if streaming_text in ['', ' ']:
                        continue
                   
                    if is_first:
                        print(f'{time.time() - start_time}')
                        is_first = False
                        text_for_yield = f'<s>{streaming_text}\n'
                        logger.info(f'*** Yield Time taken: {time.time() - yield_start_time}s ***')
                    else:
                        if '\n' in new_text:
                            is_response_yielding = False
                            text_for_yield = f"{streaming_text.strip()}<e>\n".replace('<|showslot|>', '').replace('<|endoforder|>', '')
                        else:
                            text_for_yield = f'{streaming_text}\n'
                            
                    yield text_for_yield
                        
                    logger.info(f' >> Yield streaming text: {streaming_text} <<')
                    
            is_first = True
            
            if slot is None and "현재 주문슬롯:" in history: # 현재 주문 슬롯이 없으면? 
                slot = history
                slot_str = slot.split("현재 응대:")[1].split("현재 주문슬롯:")[1].strip()
                slot_str = slot_str.replace('<|eot_id|>', '').replace('<|nooptions|>', "'<|nooptions|>'")
                slot = ast.literal_eval(slot_str)
                
                if slot['주문내역'][0]:
                    if slot['주문내역'][0]['옵션'][0] == '<|nooptions|>':
                        temp = list(slot['주문내역'][0]['옵션'][0])
                        temp = ''.join(temp[1:-1])
                        slot['주문내역'][0]['옵션'][0] = temp
                    
                new_slot['orderList'] = []
                for order in slot['주문내역']:
                    try:
                        if order is None:
                            continue

                        new_order = {
                            'prodCd': str(code_info.get(order['상품명'])),
                            'prodNm': order['상품명'],
                            'prodQty': int(order['수량']),
                            'optionprodList': []
                        }
                        if order['옵션'] == '<|nooptions|>':
                            new_slot['orderList'].append(new_order)
                            del new_order['optionprodList']
                            continue
                        
                        for option in order['옵션']:
                            for k, v in option.items():
                                if k in ['사이드 선택', '사이즈 선택', '맛 선택', '아이스볼 선택', '치즈 선택', '음료선택', '음료 선택', '온도 선택', '수량 선택']:
                                    new_order['optionprodList'].append({
                                        'optionprodCd': str(code_info.get(v)),
                                        'optionprodNm': v,
                                        'optionprodQty': 1 if v else None,
                                        'optionprodGrpCd': str(option_code_info.get(k)),
                                        'optinoprodGrpNm': k
                                    })
                    
                        new_slot['orderList'].append(new_order)
                        
                    except TypeError as e:
                        logger.error(f" !!! Error raised when generating the slots >>> \n{e}")
                        continue
                        
                new_slot['delivmthdCd'] = deliver_dict.get(slot['매장포장여부'], None)
                
                if slot['결제수단'] in easy_pay_inv_dict:
                    new_slot['paymntmnCd'] = 'PC'
                    new_slot['easypaymnttypeCd'] = easy_pay_inv_dict.get(slot['결제수단'], None)
                    
                else:
                    new_slot['paymntmnCd'] = pay_inv_dict.get(slot['결제수단'], None)
                    new_slot['easypaymnttypeCd'] = 'Null'
                    
                new_slot['orderCheck'] = 'False'
                
            if not showslot and "<|showslot|>" in history:
                showslot = True
                new_slot['orderCheck'] = 'True'
                
                text = ''
                for i, slot in enumerate(new_slot['orderList']):
                    if 'prodNm' in slot:
                        text += f"{slot['prodNm']} {slot['prodQty']}개 "
                        # if slot['optionprodList']:
                        if 'optionprodList' in slot:
                            options = [
                                f"{option['optionprodNm']} 사이즈" if option['optionprodNm'] in ['R', 'L'] else str(option['optionprodNm'])
                                for option in slot['optionprodList']
                            ]
                            text += f"의 옵션은 {'와 '.join(options)} "
                            text = text.replace("개 의", "개의")
                    if i == len(new_slot['orderList']) - 1:
                        text += "주문하셨어요. "
                    else:
                        text += "주문하셨고, "

                dining_option = "매장 식사로" if new_slot['delivmthdCd'] == 'S' else "포장으로"
                checking_list_ment = f'주문하신 내역은 {dining_option} {text}'
                logger.info(f"  >>>> {checking_list_ment}")
                
                yield '<s>' + checking_list_ment + '<e>\n'
                        
            if '<|endoforder|>' in history:
                endoforder = True
                new_slot['orderCheck'] = 'True'
                        
            response_text = response_text.split('응대: ')[1].split('<|eot_id|>')[0]
            
            if '<|showslot|>' in response_text:
                logger.info(' **** <|showslot|> Token is generated ****')
                response_text = response_text.replace('<|showslot|>', '')
                
            if '<|endoforder|>' in response_text:
                logger.info(' **** <|endoforder|> Token is generated, order finished ****')
                response_text = response_text.replace('<|endoforder|>', '')
                
                
            logger.info(f"\n\n >> Response: {response_text}\n\n  >>> Slot: {new_slot}\n")
            
            end_time = time.time() - start_time
            history += f"\n\n[all end time >> {end_time:.2f}s]"
            
            update_history(history)
            logger.info(f'*** Time taken: {end_time:.2f}s ***')
            logger.info(f'-'*100)
            
            # InterfaceID 002
            if endoforder:
                response_dict['header']['interfaceID'] = 'AI-SDC-CAT-002'
                response_dict['body'].update(new_slot)
                initialize_history()
                logger.info('-- [InterfaceID 002] ---')
                yield json.dumps(response_dict) # 앞에서 yield로 생성된 토큰을 바로 보내는 이상 return이 먹히지 않음. yield로 해야함
                # return json.dumps(response_dict)
                # return slot_json, response_json
            
            # InterfaceID 001```
            else:
                response_dict['body']['text'] = response_text
                logger.info(response_dict)
                logger.info('-- [InterfaceID 001] ---')
                # yield json.dumps(response_dict) # 앞에서 yield로 생성된 토큰을 바로 보내는 이상 return이 먹히지 않음. yield로 해야함
                # return json.dumps(response_dict)
                # return resonse_json
        
        return Response(stream_with_context(generate(is_response_yielding)), content_type='application/json')

    except Exception as e:
        logger.error(f'\n\n !!!!! Error raised >>> \n{e}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)