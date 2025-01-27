from threading import Thread
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import time
import pandas as pd
import json
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import pickle
import ast

app = Flask(__name__, template_folder='../templates')

class args:
    store = '프랭크버거'
    output_dir = "/home/user06/beaver/log/"
    llama_v1_model_name_or_path = "/home/user06/beaver/log/sft/24-07-18_15-14-56/checkpoint-500"
    instruction_template = "/home/user06/beaver/data/sft_v0.6.txt"
    embedding_model = "BAAI/bge-m3"
    retriever_k = 4
    retriever_bert_weight = 0.7
    cache_dir = "/nas/.cache/huggingface"
    model_revision = "main"
    config_name = None

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
    "프랭크버거": get_retriever(read_db(args.output_dir, "faiss", "프랭크버거", hf), read_db(args.output_dir, "docs", "프랭크버거"))
}

menu_df = pd.read_csv("/home/user06/beaver/data/dataset_v9.5.csv")

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

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)  # 실시간으로 중간 결과를 출력하는 스트리머.
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
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
        input_variables=["store", "menu_info", "retriever", "before_input", "before_slot", "before_response", "current_input"],
        template=prompt_template
    )

class InputHistory:  # 이전 대화, 이전 슬롯, 이전 응답, 현재 대화를 보관하는 클래스.
    before_input = None
    before_slot = None
    before_response = None
    current_input = None

def initialize_history():
    InputHistory.before_input = "이전 대화 없음"
    InputHistory.before_slot = "{'매장포장여부': None, '주문내역': [None], '결제수단': None}"
    InputHistory.before_response = "이전 대화 없음"
    InputHistory.current_input = None

def update_history(response):
    updated_before_input = InputHistory.current_input
    updated_before_slot, updated_before_response = response.split("현재 응대: ")
    updated_before_slot = updated_before_slot.split("현재 주문슬롯: ")[1].strip()
    updated_before_response = updated_before_response.split("<|eot_id|>")[0].strip()

    InputHistory.before_input = updated_before_input
    InputHistory.before_slot = updated_before_slot
    InputHistory.before_response = updated_before_response

def run_llm_chain(message):  # Thread의 target 매개변수로 입력되는 함수.
    llm_chain.invoke(message)   # 입력된 message를 langchain 파이프라인을 통해 처리해, 언어 모델로부터 출력 생성.

is_initial = False  # 초기화 되었는지 여부
endoforder = False  # 주문 종료 여부
args.model_name_or_path = args.llama_v1_model_name_or_path

initialize_model()  # config, tokenizer, streamer, model, pipline, langchain wrapper, prompt template 한꺼번에 초기화
initialize_history()  # 이전 대화, 이전 슬롯, 이전 응답, 현재 대화 초기화

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/order', methods=['POST'])
def order():
    global is_initial, endoforder, model, llm_chain
    
    with app.app_context():
        data = request.json
        message = data.get('message')
        
    def generate_response(message):
        global is_initial, endoforder
        
        if is_initial == False:
            is_initial = True
            yield "프랭크버거 매장을 방문해 주셔서 감사합니다. 주문을 시작합니다\n\n"
            return
        
        if is_initial and message == '초기화':
            initialize_history()
            yield "슬롯을 초기화 했습니다.\n\n"
            return

        InputHistory.current_input = message

        with open('/home/user06/beaver/data/dataset/menu_info_converted.json', 'r', encoding='utf-8') as f:
            menu_info = json.load(f)
        menu_info = json.dumps(menu_info, ensure_ascii=False, indent=4)
        
        inputs = {
            'store': args.store,
            'menu_info': menu_info,
            'before_input': InputHistory.before_input,
            'before_slot': InputHistory.before_slot,
            'before_response': InputHistory.before_response,
            'current_input': InputHistory.current_input
        }
        retriever = retriever_dict[inputs['store']]
        retriever_result = format_docs([f"{i.page_content}: {i.metadata['옵션']}" for i in retriever.invoke(invoke_format(inputs))])
        inputs['retriever'] = retriever_result
        
        processing_start_time = time.time()
        
        model_input = prompt_chain.invoke(inputs)
        
        t = Thread(target=run_llm_chain, args=(model_input,))
        t.start()
        # run_llm_chain(model_input)
        
        history = ""
        slot = None
        
        first_token_start_time = None 
        token_latency = None 
        
        for new_text in streamer:
            if not new_text:
                continue
            
            if first_token_start_time is None:
                first_token_start_time = time.time()
                token_latency = first_token_start_time - processing_start_time
                print(f"First token latency: {token_latency:.3f}s")
            
            history += new_text
            yield new_text + '\n'
                
            if slot is None and "현재 응대:" in history:
                slot = history
                slot_str = slot.split("주문슬롯:")[1].split("현재 응대:")[0].strip()
                slot_str = slot_str.replace("<|nooptions|>", "'<|nooptions|>'")
                slot = ast.literal_eval(slot_str)
                
                if isinstance(slot, dict) and '주문내역' in slot and '옵션' in slot['주문내역']:
                    slot['주문내역']['옵션'] = slot['주문내역']['옵션'].replace("'<|nooptions|>'", "<|nooptions|>")
                    
                print('================ slot ================')
                print(slot)

                print("\n##########################################")
                print("매장포장여부:", slot['매장포장여부'])
                print("주문내역:")
                if slot['주문내역'] != [None]:
                    for prod_dict in slot['주문내역']:
                        print(f"  - {prod_dict['상품명']}:")
                        print(f"    - 수량:{prod_dict['수량']}")
                        print(f"    - 옵션:")
                        if isinstance(prod_dict['옵션'], dict):
                            for prod_option in prod_dict['옵션']:
                                print("      - ", end="")
                                for k, v in prod_option.items():
                                    print(f"{k}: {v}", end=" ")
                                print()
                        else:
                            print("      - ", end="")
                            print(prod_dict['옵션'], end=" ")
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
        
        total_token_latency = time.time() - first_token_start_time
        print(f"All tokens total latency: {total_token_latency:.3f}s\n")
        update_history(history)
        time.sleep(0.1)
        
        yield history

    return Response(generate_response(message), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
