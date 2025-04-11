import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import warnings
warnings.filterwarnings("ignore")

import time
import pandas as pd
import numpy as np
import gradio as gr
from threading import Thread
import gc
import ast

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import DataFrameLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer, TextIteratorStreamer
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from transformers import pipeline
from peft import AutoPeftModelForCausalLM
import pickle
import torch

class args:
    store = None
    output_dir = "/home/user06/beaver/log/"
    llama_v1_model_name_or_path = "/home/user06/beaver/log/sft/24-05-09_01-04-28/checkpoint-500"
    llama_v2_model_name_or_path = "/home/user06/beaver/log/sft/24-05-09_17-07-08/checkpoint-500"
    model_name_or_path = None
    instruction_template = "instruction_templates/sft_v0.3.txt"
    is_quantization = False
    embedding_model="BAAI/bge-m3"
    retriever_k = 4
    retriever_bert_weight = 0.7
    cache_dir = "/nas/.cache/huggingface"
    model_revision = "main"
    config_name=None
        
def read_db(output_dir, db_type, name, hf=None):
        if db_type == "faiss":
            return FAISS.load_local(f"{output_dir.split('log')[0]+'data'}/db/{name}_faiss", hf, allow_dangerous_deserialization=True)
        elif db_type == "docs":
            with open(f"{output_dir.split('log')[0]+'data'}/db/{name}_docs.pkl", "rb") as f:
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
    "아비꼬": get_retriever(read_db(args.output_dir, "faiss", "아비꼬", hf), read_db(args.output_dir, "docs", "아비꼬")),
    "고봉민김밥": get_retriever(read_db(args.output_dir, "faiss", "고봉민김밥", hf), read_db(args.output_dir, "docs", "고봉민김밥")),
    "프랭크버거": get_retriever(read_db(args.output_dir, "faiss", "프랭크버거", hf), read_db(args.output_dir, "docs", "프랭크버거")),
    "피자먹다": get_retriever(read_db(args.output_dir, "faiss", "피자먹다", hf), read_db(args.output_dir, "docs", "피자먹다")),
    "포트캔커피": get_retriever(read_db(args.output_dir, "faiss", "포트캔커피", hf), read_db(args.output_dir, "docs", "포트캔커피"))
}

menu_df = pd.read_csv("/home/user06/beaver/data/dataset_v8.csv")
all_menus_dict = {
    "프랭크버거": menu_df[menu_df['store'] == "프랭크버거"]['all_menus'].values[0],
    "고봉민김밥": menu_df[menu_df['store'] == "고봉민김밥"]['all_menus'].values[0],
    "포트캔커피": menu_df[menu_df['store'] == "포트캔커피"]['all_menus'].values[0],
    "피자먹다": menu_df[menu_df['store'] == "피자먹다"]['all_menus'].values[0],
    "아비꼬": menu_df[menu_df['store'] == "아비꼬"]['all_menus'].values[0]
}

def initialize_model():
    global prompt_chain, llm_chain, streamer, model  # 함수 밖에서도 변수의 유효범위를 유지하기 위해, 전역변수로 선언.
    
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    
    if args.is_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config = config,
        cache_dir = args.cache_dir,
        revision = args.model_revision,
        device_map = "auto",
        quantization_config = bnb_config if args.is_quantization else None
    )
    
    text_generation_pipeline = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        return_full_text = False,
        streamer = streamer,
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        max_new_tokens = 400
    )
    
    llm_chain = HuggingFacePipeline(pipeline=text_generation_pipeline)  # 허깅페이스의 pipeline을 감싸는 langchain의 wrapper

    with open(args.instruction_template, "r") as f:
        prompt_template = f.read()
        # prompt_template = prompt_template.replace("{{", "{").replace("}}", "}")  # LLaMA의 입력으로 요구되는 '{{', '}}'를 하나의 중괄호로 교체.
        
    prompt_chain = PromptTemplate(
        input_variables=["store", "all_menus", "retriever", "before_input", "before_slot", "before_response", "current_input"],
        template=prompt_template
    )
    
class InputHistory:  # 이전 대화, 이전 슬롯, 이전 응답, 현재 대화를 보관하는 클래스.
    before_input=None
    before_slot=None
    before_response=None
    current_input=None

def initialize_history():
    InputHistory.before_input = "이전 대화 없음"
    InputHistory.before_slot = "{'매장포장여부': None, '주문내역': [None], '결제수단': None}"
    InputHistory.before_response = "이전 대화 없음"
    InputHistory.current_input = None

def update_history(response):
    # (Example Sample)
    # 관련 메뉴:
    # - 카라멜 마끼아또: {'[필수 주문] 샷추가 ': ['샷추가 1회', '샷추가 2회'], '[필수 주문] 온도 ': ['ICE', 'HOT'], '[선택 주문] 원두 변경 ': ['스페셜티 원두 변경'], '[필수 주문] 컵선택 ': ['일반컵', '캔포장500ml', '캔포장(얼음x)355ml']}
    # - 1L 카라멜 마끼아또: {'[선택 주문] 보틀변경 ': ['보틀변경'], '[필수 주문] 샷추가 ': ['샷추가 1회', '샷추가 2회'], '[필수 주문] 온도 ': ['Only ICE']}
    # - 1L 디카페인 카라멜 마끼아또: {'[선택 주문] 보틀변경 ': ['보틀변경'], '[필수 주문] 온도 ': ['Only ICE']}
    # - 디카페인 카라멜마끼아또: {'[선택 주문] 샷추가(디카페인) ': ['디카페인 샷추가 1회', '디카페인 샷추가 2회'], '[필수 주문] 온도 ': ['ICE', 'HOT'], '[필수 주문] 컵선택 ': ['일반컵', '캔포장500ml', '캔포장(얼음x)355ml']}

    # 이전 사용자 입력: 이전 대화 없음
    # 이전 주문슬롯: {'매장포장여부': None, '주문내역': [None], '결제수단': None}
    # 이전 응대: 이전 대화 없음

    # 현재 사용자 입력: 카라멜 마끼아또 주문할게요<|eot_id|><|start_header_id|><|im_start|>assistant<|end_header_id|>
    # 현재 주문슬롯: {'매장포장여부': None, '주문내역': [{'상품명': '카라멜 마끼아또', '옵션': [{'온도': None, '수량': None}, {'컵선택': None, '수량': None}, {'샷추가': None, '수량': None}], '수량': 1}], '결제수단': None}
    # 현재 응대: 안녕하세요 고객님! 카라멜 마끼아또 주문 받았습니다. 카라멜 마끼아또의 온도와 샷 추가 여부, 컵 종류를 선택해주세요! 그리고 포장 여부와 개수를 선택해주세요.<|eot_id|>.
    
    updated_input = InputHistory.current_input
    updated_slot, updated_response = response.split("현재 응대: ")
    # updated_response: 안녕하세요 고객님! 카라멜 마끼아또 주문 받았습니다. 카라멜 마끼아또의 온도와 샷 추가 여부, 컵 종류를 선택해주세요! 그리고 포장 여부와 개수를 선택해주세요.<|eot_id|>. 
    updated_slot = updated_slot.split("현재 주문슬롯: ")[1].strip() 
    # updated_slot: {'매장포장여부': None, '주문내역': [{'상품명': '카라멜 마끼아또', '옵션': [{'온도': None, '수량': None}, {'컵선택': None, '수량': None}, {'샷추가': None, '수량': None}], '수량': 1}], '결제수단': None}
    updated_response = updated_response.split("<|eot_id|>")[0].strip()
     # updated_response: 안녕하세요 고객님! 카라멜 마끼아또 주문 받았습니다. 카라멜 마끼아또의 온도와 샷 추가 여부, 컵 종류를 선택해주세요! 그리고 포장 여부와 개수를 선택해주세요.
    
    # 이전 대화, 이전 슬롯, 이전 응답 모두를 현재 입력으로 업데이트.
    InputHistory.before_input = updated_input 
    InputHistory.before_slot = updated_slot
    InputHistory.before_response = updated_response
    
def run_enhanced_llm_chain(message):
    llm_chain.invoke(message)  # 입력된 message를 langchain 파이프라인을 통해 처리해, 언어 모델로부터 출력 생성.

is_initial = False  # 초기화 되었는지 여부
endoforder = False  # 주문 종료 여부
args.model_name_or_path = args.llama_v1_model_name_or_path 

initialize_model()  # config, tokenizer, streamer, model, pipline, langchain wrapper, prompt template 한꺼번에 초기화
initialize_history()  # 이전 대화, 이전 슬롯, 이전 응답, 현재 대화 초기화

def func(message, _):
    global is_initial, model, llm_chain, endoforder

    # 초기화되기 전, 매장 명을 정확히 입력할 경우, 주문 시작.
    if (is_initial == False) and (message in ['피자먹다', '고봉민김밥', '아비꼬', '포트캔커피', '프랭크버거']):
        args.store = message
        is_initial = True
        endoforder = False
        yield f"{message} 매장을 선택하셨습니다. 주문을 시작합니다."  
        return
    
    # 틀린 매장명이 입력된 경우.
    elif is_initial == False:
        yield "**고봉민김밥, 포트캔커피, 프랭크버거** 중 이용하실 매장을 선택해주세요."
        return

    # 매장명은 정확히 입력했지만 '다시'가 입력된 경우, 초기화 전으로 되돌리고 슬롯들을 모두 초기화.
    if (is_initial == True) and (message == '다시'):
        is_initial = False
        initialize_history()
        yield "다시 시작합니다. **고봉민김밥, 포트캔커피, 프랭크버거** 중 이용하실 매장을 선택해주세요."
        return
    
    # 매장명은 정확히 입력했지만 '초기화'가 입력된 경우, 초기화 상태는 True로 두되 슬롯들만 모두 초기화.
    if (is_initial == True) and (message == '초기화'):
        initialize_history()
        yield "슬롯을 초기화 했습니다."
        return
        
    InputHistory.current_input = message
    
    # https://docs.baseten.co/deploy/examples/03-llm-with-streaming
    inputs = {
        'store': args.store,
        'all_menus': all_menus_dict[args.store],
        'before_input': InputHistory.before_input,
        'before_slot': InputHistory.before_slot,
        'before_response': InputHistory.before_response,
        'current_input': InputHistory.current_input
    }
    
    retriever = retriever_dict[inputs['store']]  # 매장명과 대응하는 retriever 추출.
    
    # def format_docs(docs):
    #   return "\n".join(f"- {doc}".replace('"', '') for doc in docs[:args.retriever_k])
    
    # def invoke_format(example):
    #   before_slot = example['before_slot']
    #   text1 = " ".join([i.split('\'')[1] for i in before_slot.split("상품명': ")[1:]])
    #   text2 = "" if example['before_input'] == "이전 대화 없음" else example['before_input']
    #   text3 = example['current_input']
    #   text = text1 + " " + text2 + " " + text3
    #   return text
    retriever_result = format_docs([f"{i.page_content}: {i.metadata['옵션']}" for i in retriever.invoke(invoke_format(inputs))])
    inputs['retriever'] = retriever_result  # inputs 딕셔너리에 retriever에서 추출된 content 추가.
    # print(f"\n\n{retriever_result}")
    
    model_input = prompt_chain.invoke(inputs)
    # print(model_input)
    
    t = Thread(target=run_enhanced_llm_chain, args=(model_input,))
    t.start()
        
    history = ""
    
    slot = None
    start_time = time.time()
    for new_text in streamer:
        history += new_text
        if slot == None and "현재 응대:" in history:
            slot = history
            slot = ast.literal_eval(slot.split("주문슬롯:")[1].split("현재 응대:")[0].strip())

            print("\n##########################################")
            print("매장포장여부:", slot['매장포장여부'])
            print("주문내역:")
            if slot['주문내역'] != [None]:
                for prod_dict in slot['주문내역']:
                    print(f"  - {prod_dict['상품명']}:")
                    print(f"    - 수량:{prod_dict['수량']}")
                    print(f"    - 옵션:",)
                    for prod_option in prod_dict['옵션']:
                        print("      - ", end="")
                        for k, v in prod_option.items():
                            print(f"{k}: {v}", end=" ")
                        print()
            else:
                print("  없음")
            print("결제수단:", slot['결제수단'])    
            print("##########################################\n")
        
        
        if not endoforder and "<|endoforder|>" in history:
            endoforder = True
            print("\n#############################################")
            print("########## 주문 완료 POST API 요청 ##########")
            print("#############################################")
        
        yield history
        
    end_time = time.time() - start_time
    # history += f"\n\n[generatestart end time >> {end_time:.2f}s]"
    history += f"\n\n[all end time >> {end_time:.2f}s]"
    yield history
    update_history(history)
    # print("this is history", history)
    
    print(end_time)
    time.sleep(0.1)

demo = gr.ChatInterface(
    fn=func, 
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder='주문이나 상품에 관한 질문을 하세요!', container=False, scale=7),
    title="Beaver Works x CUK NLP",
    description='## LLM-based Universal Store Chatbot\n\n- **고봉민김밥, 포트캔커피, 프랭크버거** 중 이용하실 매장을 선택해주세요.\n- "**다시**"를 입력하면 매장과 슬롯이 초기화됩니다.\n- "**초기화**"를 입력하면 슬롯이 초기화됩니다.',
    theme="gradio/soft",
    retry_btn=None,
    undo_btn="Delete",
    clear_btn="Clear",
)

demo.launch(share=True, debug=True)

####################################################################

""" 
생성형_AI_기반_주문_챗봇_v0.3 버전 시연영상입니다.

1. 잘못된 주문 안 채우는 예시
2. 유사한 주문 채우는 예시
3. 정보 추출하는 예시 (약함)
3. 슬롯 잘 채우는 예시
4. 다 채울경우 주문 리스트 제시하는 예시
5. 주문 리스트 확인할 경우 결제하는 예시

[프랭크버거 예시]
새우버거 하나 주세요
프랑크버거 세트 하나 주세요
########### 사이드 종류는 뭐가 있어?
스파이시텐더로 주고 라지사이즈에 콜라로 줘
SG 불고기 버거도 하나 추가해줘
작은 사이즈로 주세요
카드로 결제하고 매장에서 먹을게요
그렇게 주세요

[고봉민김밥 예시]
킹크랩김밥 하나 줘
    김치 알밥 하나 줄래?
    쏘시지 김밥 한줄 줘
와사비 참치 김밥 하나 줄래?
김밥 말고 식사류 중에 뭐있어?
생와사비 참치 김밥에 돈가스 옵션 추가해줘
결제는 카드로 하고 포장할게
맞아

[포트캔커피 예시]
빽스치노 하나 줘
    바나나 라떼 한잔 줄래?
참외 주스 하나 주문할게요
선택할 수 있는 추가 옵션은 뭐있어?
그럼 캔으로 주고 시원한걸로 마시고 갈게
결제는 카카오페이로 할게
맞아
"""
