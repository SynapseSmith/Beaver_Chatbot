import json
import logging
import pickle
import os
import gc
import random
import sys
import copy
import time
import math
import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

import wandb
import torch
import datasets
import transformers
from transformers import (
    TrainingArguments, 
    HfArgumentParser, 
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer, 
    default_data_collator, 
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling, 
    set_seed,
    BitsAndBytesConfig,
    GPTQConfig
)
from peft import (
    get_peft_config, 
    PeftModel, 
    PeftConfig, 
    get_peft_model, 
    LoraConfig, 
    TaskType,
    prepare_model_for_kbit_training
)
from trl import DataCollatorForCompletionOnlyLM
import evaluate

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    train_file: str = field(
        default="/home/user06/beaver/data/dataset/beaver_v8_train.json",
        metadata={"help": "The input training data file (a dataset)."}
    )
    validation_file: str = field(
        default="/home/user06/beaver/data/dataset/beaver_v8_val.json",
        metadata={"help": "The input validating data file (a dataset)."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input testing data file (a dataset)."}
    )

@dataclass
class ModelArguments:
    project: str = field(
        default="beaver_instruction_tuning",
        metadata={"help": "The name of the project"}
    )
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "The model id"}
    )
    embedding_model: str = field(
        default="BAAI/bge-m3",
        metadata={"help": "The embedding model"}
    )
    instruction_template: str = field(
        default=None,
        metadata={"help": "The instruction template"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    max_len: int = field(
        default=2048,
        metadata={"help": "The maximum length of the input"}
    )
    cache_dir: Optional[str] = field(
        default="/nas/.cache/huggingface",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    is_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use Lora or not"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "The number of attention heads in Lora"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha value in Lora"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate in Lora"}
    )
    lora_target_modules: str = field(
        default=None,
        metadata={"help": "The target modules in Lora"}
    )
    is_retriever: bool = field(
        default=True,
        metadata={"help": "Whether to use retriever or not"}
    )
    retriever_k: int = field(
        default=4,
        metadata={"help": "The number of retrievals in RAG"}
    )
    retriever_bert_weight: float = field(
        default=0.7,
        metadata={"help": "The weight of retriever in RAG"}
    )
    WANDB_TOKEN: str = field(
        default="none",
        metadata={"help": "The wandb token"}
    )
    HF_TOKEN: str = field(
        default="none",
        metadata={"help": "The huggingface token"}
    )

cnt_instruction_template_1, cnt_instruction_template_2 = 0, 0
cnt_format_docs = 0
cnt_invoke_format = 0
cnt_compute_metrics = 0
cnt_preprocess_logits_for_metrics = 0

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    wandb.login(key=model_args.WANDB_TOKEN)
    os.system(f"huggingface-cli login --token={model_args.HF_TOKEN}")
    wandb.init(project=model_args.project, dir=training_args.output_dir)
    set_seed(training_args.seed)
    print(torch.cuda.device_count())
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load Data
    train_dataset = datasets.load_dataset('json', data_files=data_args.train_file)['train']
    eval_dataset = datasets.load_dataset('json', data_files=data_args.validation_file)['train']
    # eval_dataset = eval_dataset.select(range(2))
    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,  
        revision=model_args.model_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        padding_side="right"
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    print('==================== pad_token ====================')
    print(tokenizer.pad_token)   # <|end_of_text|>
    print(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))  # 128001

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        # quantization_config=GPTQConfig(bits=4, tokenizer=tokenizer) if model_args.is_lora else None,
        device_map=None if training_args.deepspeed else "auto",
    )
    
    #  # 특정 레이어 동결 (예: 상위 몇 개 레이어만 학습)
    # for name, param in model.named_parameters():
    #     print(name)
    #     if 'layers' in name:  # 'layers'가 이름에 포함된 경우
    #         parts = name.split('.')
    #         if len(parts) > 2 and parts[2].isdigit():
    #             layer_index = int(parts[2])  # 'layers.N'에서 N을 추출
    #             if layer_index < 20:  # 예시: 6번째 레이어까지 동결
    #                 param.requires_grad = False
    #                 print('frozen layers')
    
    # print('-' * 100)
    # print('-' * 100)
    # print('-' * 100)
    
                
    if model_args.is_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,  # (d x lora_r) * (lora_r x h)
            lora_alpha=model_args.lora_alpha, # W = W0 + lora_alpha * (B * A)
            lora_dropout=model_args.lora_dropout,  # A, B에서 사용되는 dropout probability.
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
            target_modules="all-linear" if model_args.lora_target_modules=="all-linear" else model_args.lora_target_modules.split(",")
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
        print('==================== model ====================')
        print(model)
        model = get_peft_model(model, lora_config)
        print(model)
        
    else:
        new_tokens = ["<|showslot|>", "<|endoforder|>", "<|nooptions|>"]
        num_added_toks = tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print('==================== tokenizer ====================')
        print("We have added", num_added_toks, "tokens")  # We have added 3 tokens
        print(f"<|showslot|>: {tokenizer.convert_tokens_to_ids('<|showslot|>')},\
            <|endoforder|>: {tokenizer.convert_tokens_to_ids('<|endoforder|>')},\
            <|nooptions|>: {tokenizer.convert_tokens_to_ids('<|nooptions|>')},\
            <|eot_id|>: {tokenizer.convert_tokens_to_ids('<|eot_id|>')}")
        # <|showslot|>: 128256, <|endoforder|>: 128257, <|nooptions|>: 128258
        print(len(tokenizer))  # 128259
        print(tokenizer.all_special_ids)  # [128000, 128001]
        print(tokenizer.decode(tokenizer.all_special_ids))
        # "bos_token_id": 128000,
        # "eos_token_id": [128001, 128009]
        
    # # Padding strategy
    # if data_args.pad_to_max_length:
    #     padding = "max_length"
    # else:
    #     # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    #     padding = False
    
    
    def read_db(output_dir, db_type, name, hf=None):
        if db_type == "faiss":
            print(f"{output_dir.split('log')[0]+'data'}/db/{name}_faiss11")
            return FAISS.load_local(f"{output_dir.split('log')[0]+'data'}/db/{name}_faiss11", hf, allow_dangerous_deserialization=True)
        elif db_type == "docs":
            print(f"{output_dir.split('log')[0]+'data'}/db/{name}_docs.pkl")
            with open(f"{output_dir.split('log')[0]+'data'}/db/{name}_docs10.pkl", "rb") as f:
                return pickle.load(f) 
        else:
            raise ValueError("db_type should be either faiss or docs")
    
    def get_retriever(db, docs):
        db_reteriver = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": model_args.retriever_k},
        )
        docs_reteriver = BM25Retriever.from_documents(docs)
        docs_reteriver.k = model_args.retriever_k
        retriever = EnsembleRetriever(
            retrievers=[db_reteriver, docs_reteriver],
            weights=[model_args.retriever_bert_weight, 1 - model_args.retriever_bert_weight],
        )
        return retriever
    
    def format_docs(docs):
        return "\n".join(f"- {doc}".replace('"', '') for doc in docs[:model_args.retriever_k])
    
    def invoke_format(example):
        before_slot = example['before_slot']
        text1 = " ".join([i.split('\'')[1] for i in before_slot.split("상품명': ")[1:]])
        text2 = "" if example['before_input'] == "이전 대화 없음" else example['before_input']
        text3 = example['current_input']
        text = text1 + " " + text2 + " " + text3
        return text
        
    if not training_args.deepspeed:        
        encode_kwargs={'normalize_embeddings': True}
        model_kwargs={'device':'cpu'}
        hf = HuggingFaceBgeEmbeddings(  # HuggingFace 모델을 사용하여 텍스트를 임베딩하는 데 사용되는 클래스
            model_name=model_args.embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        retriever_dict = {
            "아비꼬": get_retriever(read_db(training_args.output_dir, "faiss", "아비꼬", hf), read_db(training_args.output_dir, "docs", "아비꼬")),
            "고봉민김밥": get_retriever(read_db(training_args.output_dir, "faiss", "고봉민김밥", hf), read_db(training_args.output_dir, "docs", "고봉민김밥")),
            "프랭크버거": get_retriever(read_db(training_args.output_dir, "faiss", "프랭크버거", hf), read_db(training_args.output_dir, "docs", "프랭크버거")),
            "피자먹다": get_retriever(read_db(training_args.output_dir, "faiss", "피자먹다", hf), read_db(training_args.output_dir, "docs", "피자먹다")),
            "포트캔커피": get_retriever(read_db(training_args.output_dir, "faiss", "포트캔커피", hf), read_db(training_args.output_dir, "docs", "포트캔커피"))
        }
        
        print('==================== retriever_dict ====================')
        print(retriever_dict)

    file_path = '/home/user06/beaver/data/dataset/menu_info.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    menu_info_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    menu_info = menu_info_df.to_dict()
    print(menu_info)
    
    def preprocess_function(examples):
        with open(model_args.instruction_template, "r") as f:
                instruction_template = f.read()
                instruction_template = instruction_template.replace("{{", "{").replace("}}", "}")
        if not training_args.deepspeed:
            retriever = retriever_dict[examples['store']]
            retriever_result = format_docs([f"{i.page_content}: {i.metadata['옵션']}" for i in retriever.invoke(invoke_format(examples))])
        else:
            retriever_result = examples['retriever']

        global cnt_instruction_template_1, cnt_instruction_template_2
        if cnt_instruction_template_1 < 3:
            print('==================== before replacing, instruction_template ====================')
            print(instruction_template)
            cnt_instruction_template_1 += 1
        
            # 관련 메뉴:
          # {retriever}
            # 이전 사용자 입력: {before_input}
            # 이전 주문슬롯: {before_slot}
            # 이전 응대: {before_response}
            # 현재 사용자 입력: {current_input}<|eot_id|><|start_header_id|><|im_start|>assistant<|end_header_id|>
        
        instruction_template = instruction_template.replace("{store}", examples['store'])
        instruction_template = instruction_template.replace("{retriever}", retriever_result)
        instruction_template = instruction_template.replace("{menu_info}", str(menu_info))
        
        instruction_template = instruction_template.replace("{before_input}", examples['before_input'])
        instruction_template = instruction_template.replace("{before_slot}", examples['before_slot'])
        instruction_template = instruction_template.replace("{before_response}", examples['before_response'])
        instruction_template = instruction_template.replace("{current_input}", examples['current_input'])
        
        
        instruction_template += f"\n현재 주문슬롯: {examples['current_slot']}\n현재 응대: {examples['current_response']}"+"<|eot_id|>"
        
        if cnt_instruction_template_2 < 3:
            print('==================== after replacing, instruction_template ====================')
            print(instruction_template)
            cnt_instruction_template_2 += 1
            
        # 관련 메뉴:
        # - 베이컨 치즈 버거 세트: {'[필수 주문] 사이드 선택 ': ['치즈 스틱(3개)', '콘샐러드', '피넛 슈가볼', '스파이시 텐더(2개)', '더치즈볼(3개)', '자이언트 통닭다리(1개)', '게다리살 튀김(3개)', 'JG 치즈 스틱(체다)', 'JG 치즈 스틱(크림)', '버팔로 치킨 매운맛(윙2개+봉2개)', '버팔로 치킨 순한맛(윙2개+봉2개)', '치즈 프렌치 프라이', '코울슬로', '콘치즈볼(3개)', '통가슴살 후라이드(1개)', '프렌치 프라이', '후라이드 아이스볼 초코(1개)', '후라이드 아이스볼 초코(2개)', '후라이드 아이스볼 플레인(1개)', '후라이드 아이스볼 플레인(2개)', '후라이드 아이스볼(플레인1+초코1)'], '[필수 주문] 사이즈 선택 ': ['L', 'R'], '[필수 주문] 음료선택 ': ['콜라', '제로 콜라', '사이다', '오렌지', '아메리카노(ICE)', '아메리카노(HOT)', '밀크 쉐이크']}
        # - 베이컨 치즈 버거: {'[필수 주문] 사이즈 선택 ': ['L', 'R']}
        # - 치즈 버거 세트: {'[필수 주문] 사이드 선택 ': ['치즈 스틱(3개)', '콘샐러드', '피넛 슈가볼', '스파이시 텐더(2개)', '더치즈볼(3개)', '자이언트 통닭다리(1개)', '게다리살 튀김(3개)', 'JG 치즈 스틱(체다)', 'JG 치즈 스틱(크림)', '버팔로 치킨 매운맛(윙2개+봉2개)', '버팔로 치킨 순한맛(윙2개+봉2개)', '치즈 프렌치 프라이', '코울슬로', '콘치즈볼(3개)', '통가슴살 후라이드(1개)', '프렌치 프라이', '후라이드 아이스볼 초코(1개)', '후라이드 아이스볼 초코(2개)', '후라이드 아이스볼 플레인(1개)', '후라이드 아이스볼 플레인(2개)', '후라이드 아이스볼(플레인1+초코1)'], '[필수 주문] 사이즈 선택 ': ['L', 'R'], '[필수 주문] 음료선택 ': ['콜라', '제로 콜라', '사이다', '오렌지', '아메리카노(ICE)', '아메리카노(HOT)', '밀크 쉐이크']}
        # - 치즈 버거: {'[필수 주문] 사이즈 선택 ': ['L', 'R']}
        # 이전 사용자 입력: 이전 대화 없음
        # 이전 주문슬롯: {'매장포장여부': None, '주문내역': [None], '결제수단': None}
        # 이전 응대: 이전 대화 없음
        # 현재 사용자 입력: 베이컨 치즈 버거 단품 하나 주세요.<|eot_id|><|start_header_id|><|im_start|>assistant<|end_header_id|>
        # 현재 주문슬롯: {'매장포장여부': None, '주문내역': [{'상품명': '베이컨 치즈 버거', '옵션': [{'사이즈 선택': None, '수량': None}], '수량': 1}], '결제수단': None}
        # 현재 응대: 네 ,베이컨 치즈 버거 단품 추가해드리겠습니다. 고객님! L/R 중에 사이즈를 선택해주세요.<|eot_id|>
        
        # ========================================================================================================
        
        # 관련 메뉴:
        # - 베이컨 치즈 버거 세트: {'[필수 주문] 사이드 선택 ': ['치즈 스틱(3개)', '콘샐러드', '피넛 슈가볼', '스파이시 텐더(2개)', '더치즈볼(3개)', '자이언트 통닭다리(1개)', '게다리살 튀김(3개)', 'JG 치즈 스틱(체다)', 'JG 치즈 스틱(크림)', '버팔로 치킨 매운맛(윙2개+봉2개)', '버팔로 치킨 순한맛(윙2개+봉2개)', '치즈 프렌치 프라이', '코울슬로', '콘치즈볼(3개)', '통가슴살 후라이드(1개)', '프렌치 프라이', '후라이드 아이스볼 초코(1개)', '후라이드 아이스볼 초코(2개)', '후라이드 아이스볼 플레인(1개)', '후라이드 아이스볼 플레인(2개)', '후라이드 아이스볼(플레인1+초코1)'], '[필수 주문] 사이즈 선택 ': ['L', 'R'], '[필수 주문] 음료선택 ': ['콜라', '제로 콜라', '사이다', '오렌지', '아메리카노(ICE)', '아메리카노(HOT)', '밀크 쉐이크']}
        # - 베이컨 치즈 버거: {'[필수 주문] 사이즈 선택 ': ['L', 'R']}
        # - 치즈 버거 세트: {'[필수 주문] 사이드 선택 ': ['치즈 스틱(3개)', '콘샐러드', '피넛 슈가볼', '스파이시 텐더(2개)', '더치즈볼(3개)', '자이언트 통닭다리(1개)', '게다리살 튀김(3개)', 'JG 치즈 스틱(체다)', 'JG 치즈 스틱(크림)', '버팔로 치킨 매운맛(윙2개+봉2개)', '버팔로 치킨 순한맛(윙2개+봉2개)', '치즈 프렌치 프라이', '코울슬로', '콘치즈볼(3개)', '통가슴살 후라이드(1개)', '프렌치 프라이', '후라이드 아이스볼 초코(1개)', '후라이드 아이스볼 초코(2개)', '후라이드 아이스볼 플레인(1개)', '후라이드 아이스볼 플레인(2개)', '후라이드 아이스볼(플레인1+초코1)'], '[필수 주문] 사이즈 선택 ': ['L', 'R'], '[필수 주문] 음료선택 ': ['콜라', '제로 콜라', '사이다', '오렌지', '아메리카노(ICE)', '아메리카노(HOT)', '밀크 쉐이크']}
        # - 더블 치즈 버거 세트: {'[필수 주문] 사이드 선택 ': ['치즈 스틱(3개)', '콘샐러드', '피넛 슈가볼', '스파이시 텐더(2개)', '더치즈볼(3개)', '자이언트 통닭다리(1개)', '게다리살 튀김(3개)', 'JG 치즈 스틱(체다)', 'JG 치즈 스틱(크림)', '버팔로 치킨 매운맛(윙2개+봉2개)', '버팔로 치킨 순한맛(윙2개+봉2개)', '치즈 프렌치 프라이', '코울슬로', '콘치즈볼(3개)', '통가슴살 후라이드(1개)', '프렌치 프라이', '후라이드 아이스볼 초코(1개)', '후라이드 아이스볼 초코(2개)', '후라이드 아이스볼 플레인(1개)', '후라이드 아이스볼 플레인(2개)', '후라이드 아이스볼(플레인1+초코1)'], '[필수 주문] 사이즈 선택 ': ['L', 'R'], '[필수 주문] 음료선택 ': ['콜라', '제로 콜라', '사이다', '오렌지', '아메리카노(ICE)', '아메리카노(HOT)', '밀크 쉐이크']}

        # 이전 사용자 입력: 베이컨 치즈 버거 단품 하나 주세요.
        # 이전 주문슬롯: {'매장포장여부': None, '주문내역': [{'상품명': '베이컨 치즈 버거', '옵션': [{'사이즈 선택': None, '수량': None}], '수량': 1}], '결제수단': None}
        # 이전 응대: 네 ,베이컨 치즈 버거 단품 추가해드리겠습니다. 고객님! L/R 중에 사이즈를 선택해주세요.

        # 현재 사용자 입력: 그냥 단품 말고 세트로 주세요.<|eot_id|><|start_header_id|><|im_start|>assistant<|end_header_id|>
        # 현재 주문슬롯: {'매장포장여부': None, '주문내역': [{'상품명': '베이컨 치즈 버거 세트', '옵션': [{'사이드 선택': None, '수량': None}, {'사이즈 선택': None, '수량': None}, {'음료선택': None, '수량': None}], '수량': 1}], '결제수단': None}
        # 현재 응대: 네 세트로 주문 도와드리겠습니다 고객님! 베이컨 치즈 버거 세트의 사이즈, 사이드, 음료 옵션을 선택해주세요.<|eot_id|>
        
        if not model_args.is_retriever:
            drop_start_idx = instruction_template.find("관련 메뉴:")
            drop_end_idx = instruction_template.find("이전 사용자 입력:")
            instruction_template = instruction_template[:drop_start_idx] + "이전 사용자 입력:" + instruction_template[drop_end_idx:]
            
        outputs = tokenizer(instruction_template)
        
        return {"input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"]}
        
    print('==================== before removing columns ====================')
    print(train_dataset)
    # Dataset({
    #     features: ['store', 'before_slot', 'current_slot', 'before_input', 'before_response', 
    #                'current_input', 'current_response', 'retriever', 'all_menus'],
    #     num_rows: 1010
    # })

    remove_column_names = train_dataset.column_names
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=False,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
    train_dataset = train_dataset.remove_columns(remove_column_names)
    
    print('==================== after removing columns ====================')
    print(train_dataset)
    # Dataset({
    #     features: ['input_ids', 'attention_mask'],
    #     num_rows: 1010
    # })
    
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 2):
        logger.info(f"Sample {index} of the training set: {len(train_dataset[index]['input_ids'])} tokens.")
        # Sample 654 of the training set: 1019 tokens.
        logger.info(f"Sample {index} of the training set: {tokenizer.decode(train_dataset[index]['input_ids'])}.")
        # Sample 654 of the training set: <|begin_of_text|><|begin_of_text|><|start_header_id|>system<|end_header_id|>
        # 당신은 포트캔커피 매장의 챗봇으로, 사용자와 상호작용 하면서 필요한 정보를 수집하여 슬롯을 채우고 주문 과정을 완성하세요.
        # 기본 주문슬롯은 다음과 같습니다: {'매장포장여부': None, '주문내역': [None], '결제수단': None}
        # 사용자가 메뉴를 주문하면, 관련 메뉴 정보를 바탕으로 주문슬롯을 채우세요.
        # 특히 관련 메뉴의 [필수 선택] 옵션들은 [{옵션명: None, 수량: None}] 형태로 명확하게 주문슬롯에 추가하세요.
        # 필수 메뉴의 옵션이 없고 사용자가 옵션을 선택하지 않은 경우엔 <|nooptions|> 을 주문슬롯에 추가하세요.
        # 사용자가 주문 수량을 명시하지 않은경우 옵션의 수량은 1로 간주하세요.
        # 사용자가 정보만을 원할 경우, 모든 메뉴와 관련 메뉴를 참고하여 정보를 제공해 주고 주문슬롯을 채우지마세요.
        # 사용자의 주문이 관련 메뉴들의 메뉴명과 일치하지 않으면, 주문슬롯을 채우지 말고, 다시 입력을 요청하세요.
        # 만약 관련 메뉴 속에 메뉴명과 유사 사용자의 주문이 입력되면, 관련 메뉴의 메뉴명으로 주문슬롯을 채우고 유사한 메뉴를 추가했다고 말하세요.
        # 사용자가 메뉴는 선택했으나 옵션을 아직 선택하지 않았다면, 옵션을 선택할 수 있도록 안내해 주세요.
        # 매장포장여부, 주문내역, 결제수단 주문슬롯이 다 채워졌을 경우 <|showslot|> 을 출력하여 사용자에게 주문내역을 확인 시키세요.
        # <|showslot|>을 출력한 이후에 사용자가 확인했을 경우 결제 진행 대화와 함께 <|endoforder|> 을 출력하고 대화를 종료해주세요.<|eot_id|><|start_header_id|>user<|end_header_id|>
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
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        # Sample 654 of the training set: {'input_ids': [128000, 128000, 128006, 9125, 128007, 198, 65895, 83628, 34804, 99969, 29726, 108607, 242, 102821, 45780, 242, 120, 102293, 41953, 21028, 102783, 245, 103850, 229, 43139, 11, 41820, 26799, 81673, 59134, 48424, 68611, 27797, 55000, 104448, 126168, 61139, 18918, 29833, 102201, 83290, 112640, 117998, 18359, 104965, 41381, 35495, 127264, 104219, 112813, 107123, 33931, 92245, 627, 21121, 101948, 127264, 110076, 117998, 34804, 121686, 123977, 25, 5473, 101518, 41953, 101796, 41953, 58126, 64189, 1232, 2290, 11, 364, 55430, 52688, 96318, 101577, 1232, 510, 4155, 1145, 364, 89881, 38187, 24140, 101353, 1232, 2290, 534, 56154, 27797, 108181, 52491, 109687, 18918, 127264, 108302, 11, 106434, 52491, 109687, 61139, 18918, 82818, 120378, 43139, 127264, 110076, 117998, 18359, 104965, 41381, 51402, 627, 108159, 101709, 106434, 52491, 109687, 21028, 510, 110174, 24140, 87138, 60, 39623, 113, 93131, 104210, 18973, 36092, 113, 93131, 80732, 25, 2290, 11, 29833, 104690, 25, 2290, 26516, 106612, 87472, 17835, 104167, 111372, 102893, 127264, 110076, 117998, 19954, 69508, 92245, 627, 110174, 24140, 52491, 109687, 21028, 39623, 113, 71279, 63718, 47782, 35495, 41820, 108181, 39623, 113, 93131, 18359, 87138, 88525, 115932, 50152, 108733, 220, 128258, 117615, 127264, 110076, 117998, 19954, 69508, 92245, 627, 56154, 27797, 108181, 127264, 29833, 104690, 18359, 104167, 30426, 88525, 115932, 66406, 41381, 39623, 113, 93131, 21028, 29833, 104690, 34804, 220, 16, 17835, 105131, 55430, 92245, 627, 56154, 27797, 108181, 61139, 73653, 18359, 102467, 48936, 50152, 11, 107036, 52491, 109687, 81673, 106434, 52491, 109687, 18918, 119884, 83290, 61139, 18918, 108273, 34983, 56773, 35495, 127264, 110076, 117998, 18359, 104965, 41381, 22035, 100711, 51402, 627, 56154, 27797, 110257, 127264, 13094, 106434, 52491, 109687, 106001, 52491, 109687, 80732, 54780, 84656, 60798, 88525, 51796, 91040, 11, 127264, 110076, 117998, 18359, 104965, 41381, 22035, 101264, 35495, 11, 106327, 43449, 18359, 127296, 92245, 627, 73653, 103168, 106434, 52491, 109687, 105220, 19954, 52491, 109687, 80732, 54780, 101003, 56154, 41820, 110257, 127264, 13094, 43449, 65219, 33390, 11, 106434, 52491, 109687, 21028, 52491, 109687, 80732, 43139, 127264, 110076, 117998, 18359, 104965, 41381, 35495, 101003, 56154, 24486, 52491, 109687, 18918, 69508, 101528, 35495, 101264, 92245, 627, 56154, 27797, 108181, 52491, 109687, 16969, 87138, 102621, 112804, 39623, 113, 93131, 18359, 117686, 87138, 88525, 112269, 33390, 11, 39623, 113, 93131, 18359, 87138, 48936, 29833, 123644, 103603, 34983, 56773, 51402, 627, 101518, 41953, 101796, 41953, 58126, 64189, 11, 127264, 96318, 101577, 11, 83719, 38187, 24140, 101353, 127264, 110076, 117998, 13094, 50467, 104965, 103430, 106872, 18359, 50152, 220, 128256, 117615, 62226, 83290, 41820, 26799, 102244, 127264, 96318, 101577, 18359, 74959, 45618, 102474, 51402, 627, 128256, 18359, 62226, 24486, 111323, 19954, 41820, 108181, 74959, 102621, 18359, 50152, 83719, 38187, 111809, 62060, 57390, 81673, 106999, 220, 128257, 117615, 62226, 101360, 62060, 117216, 99458, 64356, 34983, 92769, 13, 128009, 128006, 882, 128007, 198, 127489, 52491, 109687, 512, 12, 103236, 51440, 28313, 250, 96677, 110833, 54059, 121509, 25, 5473, 58, 110174, 24140, 127264, 60, 100514, 115, 103948, 20565, 13906, 2570, 70482, 115, 103948, 20565, 220, 16, 62841, 518, 364, 70482, 115, 103948, 20565, 220, 17, 62841, 4181, 18814, 110174, 24140, 127264, 60, 106083, 49085, 13906, 2570, 5604, 518, 364, 39, 1831, 4181, 18814, 14901, 76232, 127264, 60, 102467, 103097, 88837, 13906, 2570, 25941, 104249, 123916, 102199, 102467, 103097, 88837, 4181, 18814, 110174, 24140, 127264, 60, 90195, 113, 14901, 76232, 13906, 2570, 125462, 124754, 518, 364, 108607, 242, 101796, 41953, 2636, 1029, 518, 364, 108607, 242, 101796, 41953, 7, 111249, 49531, 87, 8, 17306, 1029, 63987, 12, 220, 16, 43, 103236, 51440, 28313, 250, 96677, 110833, 54059, 121509, 25, 5473, 58, 14901, 76232, 127264, 60, 64432, 112150, 104449, 66406, 13906, 2570, 42771, 112150, 104449, 66406, 4181, 18814, 110174, 24140, 127264, 60, 100514, 115, 103948, 20565, 13906, 2570, 70482, 115, 103948, 20565, 220, 16, 62841, 518, 364, 70482, 115, 103948, 20565, 220, 17, 62841, 4181, 18814, 110174, 24140, 127264, 60, 106083, 49085, 13906, 2570, 7456, 41663, 63987, 12, 220, 16, 43, 105638, 101436, 104249, 32428, 103236, 51440, 28313, 250, 96677, 110833, 54059, 121509, 25, 5473, 58, 14901, 76232, 127264, 60, 64432, 112150, 104449, 66406, 13906, 2570, 42771, 112150, 104449, 66406, 4181, 18814, 110174, 24140, 127264, 60, 106083, 49085, 13906, 2570, 7456, 41663, 63987, 12, 105638, 101436, 104249, 32428, 103236, 51440, 28313, 250, 100711, 110833, 54059, 121509, 25, 5473, 58, 14901, 76232, 127264, 60, 100514, 115, 103948, 20565, 7, 90335, 101436, 104249, 32428, 8, 13906, 2570, 90335, 101436, 104249, 32428, 100514, 115, 103948, 20565, 220, 16, 62841, 518, 364, 90335, 101436, 104249, 32428, 100514, 115, 103948, 20565, 220, 17, 62841, 4181, 18814, 110174, 24140, 127264, 60, 106083, 49085, 13906, 2570, 5604, 518, 364, 39, 1831, 4181, 18814, 110174, 24140, 127264, 60, 90195, 113, 14901, 76232, 13906, 2570, 125462, 124754, 518, 364, 108607, 242, 101796, 41953, 2636, 1029, 518, 364, 108607, 242, 101796, 41953, 7, 111249, 49531, 87, 8, 17306, 1029, 663, 633, 13094, 66965, 41820, 26799, 43449, 25, 119444, 62060, 57390, 127409, 198, 13094, 66965, 127264, 110076, 117998, 25, 5473, 101518, 41953, 101796, 41953, 58126, 64189, 1232, 2290, 11, 364, 55430, 52688, 96318, 101577, 1232, 510, 4155, 1145, 364, 89881, 38187, 24140, 101353, 1232, 2290, 534, 13094, 66965, 113914, 67945, 25, 119444, 62060, 57390, 127409, 271, 35859, 80979, 41820, 26799, 43449, 25, 103236, 51440, 28313, 250, 96677, 110833, 54059, 121509, 127264, 48936, 58901, 36811, 128009, 128006, 27, 91, 318, 5011, 91, 29, 78191, 128007, 198, 35859, 80979, 127264, 110076, 117998, 25, 5473, 101518, 41953, 101796, 41953, 58126, 64189, 1232, 2290, 11, 364, 55430, 52688, 96318, 101577, 1232, 62208, 125399, 80732, 1232, 364, 118618, 28313, 250, 96677, 110833, 54059, 121509, 518, 364, 36092, 113, 93131, 1232, 62208, 102837, 49085, 1232, 2290, 11, 364, 24140, 104690, 1232, 2290, 2186, 5473, 124754, 14901, 76232, 1232, 2290, 11, 364, 24140, 104690, 1232, 2290, 2186, 5473, 70482, 115, 103948, 20565, 1232, 2290, 11, 364, 24140, 104690, 1232, 2290, 73541, 364, 24140, 104690, 1232, 220, 16, 73541, 364, 89881, 38187, 24140, 101353, 1232, 2290, 534, 35859, 80979, 113914, 67945, 25, 96270, 124409, 116534, 102424, 0, 103236, 51440, 28313, 250, 96677, 110833, 54059, 121509, 127264, 84696, 101760, 39331, 13, 103236, 51440, 28313, 250, 96677, 110833, 54059, 121509, 21028, 106083, 49085, 81673, 100514, 115, 69508, 84618, 64189, 11, 90195, 113, 99458, 99029, 18918, 87138, 34983, 92769, 0, 107536, 99969, 41953, 84618, 64189, 81673, 74623, 120045, 87138, 34983, 92769, 13, 128009], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}.
    
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=False,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
    eval_dataset = eval_dataset.remove_columns(remove_column_names)
    
    print('==================== eval_dataset ====================')
    print(eval_dataset)
    # Dataset({
    #     features: ['input_ids', 'attention_mask'],
    #     num_rows: 42
    # })

    max_length = max(len(i['input_ids']) for i in datasets.concatenate_datasets([train_dataset, eval_dataset]))
    print('==================== max_length ====================')
    print(max_length)  # max_length: 2722
    
    response_template_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    
    print('==================== response_template_ids ====================')
    print(response_template_ids)  # [27, 91, 318, 5011, 91, 29, 78191]
    # 모델이 응답을 생성해야 하는 부분을 명확히 하기 위한 토큰 시퀀스.
    
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )  # 여러 샘플을 하나의 배치로 묶는 과정에서, 각각의 샘플들을 적절히 패딩하고 
       # 마스킹하여 모델이 배치 단위로 학습 하도록 함.
    
    bleu = evaluate.load("bleu")
    acc = evaluate.load("accuracy")
    
    def preprocess_logits_for_metrics(logits, labels):
        global cnt_preprocess_logits_for_metrics
        if cnt_preprocess_logits_for_metrics < 3:
            print('==================== logits shape ====================')
            print(logits.shape)
            # torch.Size([1, 1347, 128259])
            # torch.Size([1, 826, 128259])
            # torch.Size([1, 2109, 128259])
            
            # tensor([[[ 1.2422,  6.8438,  3.9844,  ...,  1.6406, -1.2422, -1.1406],
            #         [ 1.2422,  6.8438,  3.9844,  ...,  1.6406, -1.2422, -1.1406],
            #         [-0.0322, -0.9180, -0.2432,  ...,  0.5781, -0.0566, 16.2500],
            #         ...,
            #         [ 1.6641, -3.8594,  1.7500,  ...,  8.3125,  0.6719, 27.2500],
            #         [ 0.7500, -8.7500,  2.3594,  ...,  7.1562,  1.8125, 29.1250],
            #         [ 1.9766, -3.2500,  3.3125,  ...,  8.1250,  1.2031, 28.8750]]],
            #    device='cuda:0')
            cnt_preprocess_logits_for_metrics += 1
            
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        
        # argmax to get the token ids
        return logits.argmax(dim=-1)  # logits.shape: torch.size([1, 1347])
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels,
        # after the argmax(-1) has been calculated by preprocess_logits_for_metrics
        # but we need to shift the labels
        labels = labels[:, 1:]
        preds = preds[:, :-1]
        global_mask = labels == -100
                
        # Tokenize the marker string "\n현재 응대:"
        marker_token_ids = tokenizer.encode("현재 응대:", add_special_tokens=False)
        print(marker_token_ids)
        # [35859, 80979, 113914, 67945, 25]
        
        # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        
        global cnt_compute_metrics
        if cnt_compute_metrics < 1:
            print('==================== labels, preds, global_mask ====================')
            print(len(labels), len(preds), len(global_mask))  # 42, 42, 42
            print(labels[:3])
            # [[-100 -100 -100 ... -100 -100 -100]
            #  [-100 -100 -100 ... -100 -100 -100]
            #  [-100 -100 -100 ... -100 -100 -100]]
            print(preds[:3])
            # [[128001 128001 128258 ...   -100   -100   -100]
            #  [128001 128001 128258 ...   -100   -100   -100]
            #  [128001 128001 128258 ...   -100   -100   -100]]
            print(global_mask[:3])  
            # [[ True  True  True ...  True  True  True]
            #  [ True  True  True ...  True  True  True]
            #  [ True  True  True ...  True  True  True]]
            cnt_compute_metrics += 1
        
        # Calculate slot BLEU (token-level)
        slot_bleu_scores = []
        cnt_label_ids = 0
        for label_ids, pred_ids, g_mask in zip(labels, preds, global_mask):
            if cnt_label_ids < 1:
                print('==================== g_mask, label_ids, pred_ids ====================')
                print(g_mask)
                # [ True  True  True ...  True  True  True]
                label_ids = label_ids[~g_mask]
                # marker_token_ids: [35859, 80979, 113914, 67945, 25]
                print(label_ids)
                # [128007    198  35859  80979 127264 110076 117998     25   5473 101518
                #   41953 101796  41953  58126  64189   1232   2290     11    364  55430
                #   52688  96318 101577   1232  62208 125399  80732   1232    364 101204
                #   39519    255  82233  87931  93292    518    364  36092    113  93131
                #   71018  13922 125166 102668  87138   1232   2290     11    364  24140
                # 104690   1232   2290  73541    364  24140 104690   1232    220     16
                #   73541    364  89881  38187  24140 101353   1232   2290    534  ▶ 35859
                #   80979 113914  67945     25 ◀ 103315 116534 102424      0  85355  39519
                #     255  82233  87931  93292  21028 101852  29726  18918 103123 101696
                #   43139 127264 101703  81673 124365 115284     13  85355  39519    255
                #   82233  87931  93292  21028 109055 102668  39623    113  93131  18359
                #   87138  34983  92769     13 128009]
                pred_ids = pred_ids[~g_mask]
                print(pred_ids)
                # [128007    198  35859  80979 127264 110076 117998     25   5473 101518
                # 41953 101796  41953  58126  64189   1232   2290     11    364  55430
                # 52688  96318 101577   1232  62208 125399  80732   1232    364 101204
                # 39519    255  82233  87931  93292    518    364  36092    113  93131
                # 1232  13922  56154 102668  87138   1232   2290     11    364  24140
                # 104690   1232   2290   2186    364  24140 104690   1232    220     16
                # 73541    364  89881  38187  24140 101353   1232   2290    534 ▶ 35859
                # 80979 113914  67945     25 ◀ 127264 116534 102424     11  85355  39519
                #     255  82233  87931  93292 105454  50152  29726  16969 116604 101696
                # 43139 116604 101703  81673  30446 115284     13  85355  39519    255
                # 82233  87931  93292  21028 109055 102668  39623    113  93131  18359
                # 87138  34983  92769     13 128009]
                cnt_label_ids += 1
            
            # Find the position of the marker in the label_ids
            label_idx = None
            pred_idx = None
            
            for i in range(len(label_ids) - len(marker_token_ids) + 1):
                if all(label_ids[i + j] == marker_token_ids[j] for j in range(len(marker_token_ids))):
                    label_idx = i
                    break
                  
            for i in range(len(pred_ids) - len(marker_token_ids) + 1):
                if all(pred_ids[i + j] == marker_token_ids[j] for j in range(len(marker_token_ids))):
                    pred_idx = i
                    break
                        
            if label_idx is not None and pred_idx is not None:
                label_slot = label_ids[label_idx+len(marker_token_ids):]
                pred_slot = pred_ids[pred_idx+len(marker_token_ids):]
                print('==================== label_slot, pred_slot ====================')
                
                # if len(pred_slot) > len(label_slot):
                #     pred_slot = pred_slot[:len(label_slot)]
                
                # valid_token_ids_range = range(tokenizer.vocab_size)
                # label_slot = [token_id for token_id in label_slot if token_id in valid_token_ids_range]
                # pred_slot = [token_id for token_id in pred_slot if token_id in valid_token_ids_range]
                
                label_slot = np.array(label_slot)
                label_slot = label_slot[label_slot != -100]
                
                pred_slot = np.array(pred_slot)
                pred_slot = pred_slot[pred_slot != -100]
                
                print(label_slot)
                print(pred_slot)
                # [103315 116534 102424 103651 108807 110076  17835 100514    238  61394
                # 30446  81673 118003 100514    238  61394  30446 127264 107094 101760
                # 39331     13 127264  67236 101577  13094 107625  34609 101272 117677
                #     30    220 128256 128009]
                # [103315 116534 102424     11 108807 110076  17835 100514    238  61394
                # 30446  81673 118003 100514    238  61394  30446 127264  34983 101760
                # 39331     13    220 101360 101577  13094 107625  34609 101272 117677
                #     30    220 128256 128009 128009]
                
                decoded_label_slot = tokenizer.decode(label_slot, skip_special_tokens=True)
                decoded_pred_slot = tokenizer.decode(pred_slot, skip_special_tokens=True)
                print('==================== decoded_label_slot, decoded_pred_slot ====================')
                print(decoded_label_slot)
                # 현재 주문슬롯: {'매장포장여부': None, '주문내역': [{'상품명': '프랭크 버거', '옵션':[{'사이즈 선택': None, '수량': None}], '수량': 1}], '결제수단': None}
                print(decoded_pred_slot, "\n\n")
                # 현재 주문슬롯: {'매장포장여부': None, '주문내역': [{'상품명': '프랭크 버거', '옵션':{'사즈 선택': None, '수량': None}, '수량': 1}], '결제수단': None}
                slot_bleu = bleu.compute(predictions=[decoded_pred_slot], references=[decoded_label_slot])
                slot_bleu_scores.append(slot_bleu['bleu'])
            else:
                slot_bleu_scores.append(0.0)
                    
        slot_bleu_avg = sum(slot_bleu_scores) / len(slot_bleu_scores) if slot_bleu_scores else 0.0

        # Decode labels and predictions to text
        labels_list = [labels_[~g_mask] for labels_, g_mask in zip(labels, global_mask)]
        preds_list = [preds_[~g_mask] for preds_, g_mask in zip(preds, global_mask)]
        print('==================== labels_list, preds_list ====================')
        print(labels_list[0])
        #     [array([128007,    198,  35859,  80979, 127264, 110076, 117998,     25,
        #      5473, 101518,  41953, 101796,  41953,  58126,  64189,   1232,
        #      2290,     11,    364,  55430,  52688,  96318, 101577,   1232,
        #     62208, 125399,  80732,   1232,    364, 101204,  39519,    255,
        #     82233,  87931,  93292,    518,    364,  36092,    113,  93131,
        #     71018,  13922, 125166, 102668,  87138,   1232,   2290,     11,
        #       364,  24140, 104690,   1232,   2290,  73541,    364,  24140,
        #    104690,   1232,    220,     16,  73541,    364,  89881,  38187,
        #     24140, 101353,   1232,   2290,    534,  35859,  80979, 113914,
        #     67945,     25, 103315, 116534, 102424,      0,  85355,  39519,
        #       255,  82233,  87931,  93292,  21028, 101852,  29726,  18918,
        #    103123, 101696,  43139, 127264, 101703,  81673, 124365, 115284,
        #        13,  85355,  39519,    255,  82233,  87931,  93292,  21028,
        #    109055, 102668,  39623,    113,  93131,  18359,  87138,  34983,
        #     92769,     13, 128009])]
        print(preds_list[0])
        #     [array([ 35859,  35859,  35859,  80979, 113914,  35859,    116,  35859,
        #     35859,  35859,  35859,  35859,  35859,  35859,  35859,  35859,
        #     35859,  35859,  35859,  35859,  35859,  35859,  35859,  35859,
        #     35859,  35859,  35859,  35859,  35859,  35859, 111932,    255,
        #     35859,  35859,  35859,  35859,  35859,  35859,  45780,  35859,
        #     35859,  35859,  35859,  35859,  35859,  35859,  35859,  35859,
        #     35859,  35859,  35859,  35859,  35859,  35859,  35859,  35859,
        #     35859,  35859,  35859,  35859,  35859,  35859,  35859,  35859,
        #     35859,  35859,  35859,  35859,  35859,  35859,  80979,  35859,
        #     35859,  35859,  35859, 122273,  35859,  35859,  35859, 111932,
        #       255,  35859,  35859,  35859,  35859,  35859, 100551,  35859,
        #     35859,  35859,  35859, 113914,  35859,  35859,  35859,  35859,
        #     35859,  35859, 111932,    255,  35859,  35859,  35859,  35859,
        #     35859, 102668,  35859,  45780,  35859,  35859, 113914,  35859,
        #     35859,  35859,  35859])]
        
        decoded_labels = tokenizer.batch_decode(labels_list, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(preds_list, skip_special_tokens=True)
        print('==================== decoded_labels, decoded_preds ====================')
        print(decoded_labels[:3])
        # ["\n현재 주문슬롯: {'매장포장여부': None, '주문내역': [{'상품명': '프랭크 버거', '옵션':[{'사이즈 선택': None, '수량': None}], '수량': 1}], '결제수단': None}\n현재 응대: 네 고객님! 프랭크 버거의 세트를 단품으로 주문 도와드리겠습니다. 프랭크 버거의 사이즈 옵션을 선택해주세요.", "\n현재 주문슬롯: {'매장포장여부': '포장', '주문내역': [{'상품명': '초코칩 쿠키', '옵션': <|nooptions|>, '수량': 1}, {'상품명': '치즈 쿠키', '옵션': <|nooptions|>, '수량': 1}], '결제수단': '네이버페이'}\n현재 응대: 고객님, 결제 전 주문 확인 부탁드리겠습니다! 이렇게 주문 도와드릴까요?<|showslot|>", "\n현재 주문슬롯: {'매장포장여부': '포장', '주문내역': [{'상품명': '치즈 버거 세트', '옵션': [{'사이드 선택': None, '수량': None}, {'사이즈 선택': 'L', '수량': 1}, {'음료선택': None, '수량': None}], '수량': 1}, {'상품명': '쉬림프 버거 세트', '옵션': [{'사이드 선택': None, '수량': None}, {'사이즈 선택': 'L', '수량': 1}, {'음료선택': None, '수량': None}], '수량': 1}], '결제수단': None}\n현재 응대: 네 고객님! 치즈 버거 세트에 L 사이즈 옵션과 쉬림프 버거 세트 L 사이즈 옵션으로 주문 도와드리겠습니다. 추가로 세트메뉴의 사이드와 음료를 선택해주세요!"]
        print(decoded_preds[:3])
        # ['��현재 응호�����������������������렌������혤�������������������������������현재����이버���렌������월���� 응������렌������즈혤��� 응����', '��현재 응혏��������������������������혩혿����혤�������������������즈혿����혤����������������������혜�현재 응������������������������', '��현재 응호�������������������������즈���트��혤������������������즈���������������혬���������������������������트��혤������������������즈���������������혬������������������������현재����트���즈���월���즈혤����름����트��즈혤��� 응������� 응월��� 응즈���� 응����']
        
        # Calculate BLEU score
        bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        # overall_accuracy = acc.compute(predictions=preds[~global_mask], references=labels[~global_mask])
        # Calculate accuracy, considering only non-ignored parts    
        return {**bleu_score, 'slot_bleu': slot_bleu_avg}


    output = f"{time.strftime('%y-%m-%d_%H-%M-%S', time.localtime())}-user06"
    training_args.output_dir = os.path.abspath((os.path.join(training_args.output_dir, output)))
    training_args.run_name=output
    wandb.run.name=output
    # Initialize our Trainer
    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_state()
    
    # evaluation
    eval_results = trainer.evaluate(metric_key_prefix="final_eval/")
    wandb.log(eval_results)
    # {'eval_loss': 0.6280598640441895, 'eval_bleu': 0.7248115180349223, 'eval_precisions': [0.8320610687022901, 0.7437574316290131, 0.6932570740517761, 0.6524390243902439], 'eval_brevity_penalty': 0.9964830048039435, 'eval_length_ratio': 0.9964891749561147, 'eval_translation_length': 3406, 'eval_reference_length': 3418, 'eval_slot_bleu': 0.838047941485266, 'eval_runtime': 32.161, 'eval_samples_per_second': 1.306, 'eval_steps_per_second': 1.306, 'epoch': 0.16}
    
    # test
    if data_args.test_file is not None:
        test_dataset = datasets.load_dataset('json', data_files=data_args.test_file)['train']
        with training_args.main_process_first(desc="test dataset map pre-processing"):
            test_dataset = test_dataset.map(
                preprocess_function,
                batched=False,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on test dataset",
            )
        test_dataset = test_dataset.remove_columns(remove_column_names)
        test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="final_test/")
        wandb.log(test_results)
    
    
if __name__ == "__main__":
    main()

