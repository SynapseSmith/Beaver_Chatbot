import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from flask import Flask, request, jsonify, Response, stream_with_context
import time
import pandas as pd
import json
import pickle
import logging
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa
import pygame
import pyttsx3
from TTS.api import TTS
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine
import io
import threading

# Initialize Flask app
app = Flask(__name__)

recognized_text = None
processing_speech = False
recognizer = sr.Recognizer()

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
log_folder = os.path.join('/home/user09/beaver/prompt_engineering/ars_log', date)
if not os.path.exists(log_folder):
    os.mkdir(log_folder)

logdir = os.path.join(log_folder, curtime + '.log')
logger = load_logger(logdir, 'INFO')
logger.info(f'*** {curtime} START ***')
logger.info(f'*** PID: {os.getpid()} ***')

class args:
    store = '프랭크버거'
    output_dir = "/home/user09/beaver/log/"
    llama_v1_model_name_or_path = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"
    instruction_template = "/home/user09/beaver/instruction_templates/sft_v0.6.txt"
    embedding_model = "BAAI/bge-m3"
    retriever_k = 4
    retriever_bert_weight = 0.7
    cache_dir = "/nas/.cache/huggingface"
    model_revision = "main"
    config_name = None

def read_db(output_dir, db_type, name, hf=None):
    if db_type == "faiss":
        return FAISS.load_local(f"{output_dir.split('log')[0]+'data'}/db/{name}_faiss11", hf, allow_dangerous_deserialization=True)
    elif db_type == "docs":
        with open(f"{output_dir.split('log')[0]+'data'}/db/{name}_docs11.pkl", "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("db_type should be either faiss or docs")

def get_retriever(db, docs):
    db_retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": args.retriever_k},
    )
    docs_retriever = BM25Retriever.from_documents(docs)
    docs_retriever.k = args.retriever_k
    retriever = EnsembleRetriever(
        retrievers=[db_retriever, docs_retriever],
        weights=[args.retriever_bert_weight, 1 - args.retriever_bert_weight],
    )
    return retriever

def format_docs(docs):
    return "\n".join(f"- {doc}".replace('"', '') for doc in docs[:args.retriever_k])

def invoke_format(example):
    text2 = "" if example['before_input'] == "이전 대화 없음" else example['before_input']
    text3 = example['current_input']
    text = text2 + " " + text3
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
        input_variables=["store", "menu_info", "side_info", "retriever", "before_input", "before_response", "current_input"],
        template=prompt_template
    )

class InputHistory:  # 이전 대화, 이전 슬롯, 이전 응답, 현재 대화를 보관하는 클래스.
    before_input = None
    before_response = None
    current_input = None

def initialize_history():
    global logger
    logger.info(' ** Initialize history **')
    InputHistory.before_input = "이전 대화 없음"
    InputHistory.before_response = "이전 대화 없음"
    InputHistory.current_input = None

def update_history(response):
    InputHistory.before_input = InputHistory.current_input
    InputHistory.before_response = response

def run_llm_chain(message):  # Thread의 target 매개변수로 입력되는 함수.
    return llm_chain.invoke(message)  # 입력된 message를 langchain 파이프라인을 통해 처리해, 언어 모델로부터 출력 생성.

is_initial = False  # 초기화 되었는지 여부
args.model_name_or_path = args.llama_v1_model_name_or_path

initialize_model()  # config, tokenizer, streamer, model, pipeline, langchain wrapper, prompt template 한꺼번에 초기화
initialize_history()  # 이전 대화, 이전 슬롯, 이전 응답, 현재 대화 초기화

def text_to_speech(text):
    print(text)
    if not text.strip():
        print("No text provided for TTS.")
        return

    def play_audio(audio):
        play(audio)  # 실시간으로 오디오를 재생
    
    def tts_thread(text):
        print(text)
        try:
            # Google TTS를 사용하는 방법
            tts = gTTS(text=text, lang='ko')
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio = AudioSegment.from_file(fp, format="mp3")

            # 비동기적으로 오디오를 재생
            threading.Thread(target=play_audio, args=(audio,)).start()

        except Exception as e:
            print(f"Error in TTS: {e}")

    # 비동기적으로 TTS 작업을 수행
    threading.Thread(target=tts_thread, args=(text,)).start()


# Function to handle continuous speech input and processing
def process_speech():
    """Continuously capture audio from the microphone and perform speech-to-text."""
    global recognized_text, processing_speech
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        logger.info("Ready to receive speech input...")

        while True:
            try:
                if not processing_speech:  # Check if the system is not currently processing a speech input
                    logger.info("Listening for speech...")
                    audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    text = recognizer.recognize_google(audio_data, language="ko-KR")
                    logger.info(f"Recognized text: {text}")
                    recognized_text = text  # Store the recognized text

                    # Start a new thread to process the recognized text
                    threading.Thread(target=order).start()

            except sr.UnknownValueError:
                logger.error("Speech Recognition could not understand audio")

            except sr.RequestError as e:
                logger.error(f"Could not request results from Google Speech Recognition service; {e}")
                break

            except Exception as e:
                logger.error(f"An error occurred: {e}")
                break
        
with open('/home/user09/beaver/data/dataset/menu_info_converted.json', 'r', encoding='utf-8') as f:
    menu_info = json.load(f)

with open('/home/user09/beaver/data/dataset/side_info_converted.json', 'r', encoding='utf-8') as f:
    side_info = json.load(f)

@app.route('/order', methods=['POST'])
def order():
    global processing_speech, recognized_text
    processing_speech = True
    
    try:
        if not recognized_text:
            return jsonify({'error': 'No speech has been recognized yet'}), 400

        # Use recognized_text to process the request
        InputHistory.current_input = recognized_text
        
        inputs = {
            'store': args.store,
            'menu_info': menu_info,
            'side_info': side_info,
            'before_input': InputHistory.before_input,
            'before_response': InputHistory.before_response,
            'current_input': InputHistory.current_input
        }
        
        retriever = retriever_dict[inputs['store']]
        retriever_result = format_docs([f"{i.page_content}: {i.metadata['옵션']}" for i in retriever.invoke(invoke_format(inputs))])
        inputs['retriever'] = retriever_result

        start_time = time.time()
        model_input = prompt_chain.invoke(inputs)
        
        history = run_llm_chain(model_input)

        if history:
            logger.info(f'Model response: {history}')
            text_to_speech(history)
        else:
            logger.error("Model did not generate a valid response")
            history = "죄송합니다, 요청을 처리할 수 없습니다."

        update_history(history)

        end_time = time.time() - start_time
        logger.info(f'*** Time taken: {end_time:.2f}s ***')
        logger.info(f'-'*100)

        return jsonify({'response': history}), 200

    except Exception as e:
        logger.error(f'\n\n !!!!! Error raised >>> \n{e}')
        return jsonify({'error': str(e)}), 500
    
    finally:
        processing_speech = False
        
def run_flask_app():
    """Run the Flask app."""
    app.run(host='127.0.0.1', port=5077, debug=False)

if __name__ == '__main__':
    # Start the speech recognition in a separate thread
    speech_thread = threading.Thread(target=process_speech)
    speech_thread.daemon = True
    speech_thread.start()

    # Run the Flask app
    run_flask_app()