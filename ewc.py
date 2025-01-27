import torch
import torch.nn.functional as F
from transformers import Trainer

class EWCLora:
    def __init__(self, model, importance=1000, device='cuda'):
        self.model = model
        self.importance = importance
        self.device = device
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.prev_params = {}
        self.fisher = {}
        self.model.to(device)

    def compute_fisher(self, dataloader):
        # Fisher Information Matrix를 계산하여 가중치의 중요도를 평가
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        
        for batch in dataloader:
            if isinstance(batch, dict):
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v).to(self.device) for k, v in batch.items()}
            else:
                inputs = {"input_ids": batch[0].to(self.device), "attention_mask": batch[1].to(self.device)}

            # 빈 시퀀스가 있는지 확인하여 처리
            if inputs['input_ids'].size(0) == 0 or (len(inputs['input_ids'].shape) > 1 and inputs['input_ids'].size(1) == 0):
                print("Skipping empty input sequence")
                continue
              
            outputs = self.model(**inputs)
            loss = outputs.loss
            self.model.zero_grad()
            loss.backward()

            for n, p in self.params.items():
                if p.grad is not None:  # grad가 None이 아닌 경우에만 처리
                    fisher[n] += p.grad ** 2

        for n in fisher:
            fisher[n] /= len(dataloader)
        
        self.fisher = fisher

    def consolidate(self):
        # 이전 학습한 가중치 및 Fisher 정보 저장
        for n, p in self.params.items():
            self.prev_params[n] = p.clone().detach()
    
    def ewc_loss(self):
        # EWC 손실 계산
        loss = 0
        for n, p in self.params.items():
            if n in self.fisher and p.shape == self.prev_params[n].shape:
                loss += (self.fisher[n] * (p - self.prev_params[n]) ** 2).sum()
            else:
                print(f"Skipping parameter {n} due to shape mismatch or missing Fisher information.")
        return loss * self.importance


class EWCTrainer(Trainer):
    def __init__(self, *args, ewc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc = ewc

    def compute_loss(self, model, inputs, return_outputs=False):
        # 기본 손실 계산
        outputs = model(**inputs)
        loss = outputs.loss
        
        # EWCLora 손실 추가
        if self.ewc is not None:
            loss += self.ewc.ewc_loss()
        
        return (loss, outputs) if return_outputs else loss
