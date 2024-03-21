#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : predict_3.py
# @Time      : 2024/3/21 14:50
# @Author    : Carl.Zhang
# Function   : predict_3

import torch
from utils import tag2idx, idx2tag, VOCAB, MAX_LEN
from models import Bert_BiLSTM_CRF
from transformers import BertTokenizer

# BertBiLSTMCRF_MODEL_PATH = r'D:\A_models\NER\new_BBC_epoch_30_dp_04.pt'
BertBiLSTMCRF_MODEL_PATH = r'D:\A_models\NER\new_BBC_epoch_30_dp_04.pkl'
# BertBiLSTMCRF_MODEL_PATH = r'D:\A_models\NER\new_BBC_epoch_30_dp_04_cpu.pkl'
BERT_PATH = '../../bert-base-chinese'

# 输出有颜色的字体
class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class BBC_Predict:
    def __init__(self):
        # 加载模型
        print("Bert-BiLSTM-CRF model loading...")
        # self.device = torch.device(device)
        # self.model = Bert_BiLSTM_CRF(tag2idx)
        # self.model.load_state_dict(torch.load(crf_model))
        # self.model.to(device)
        self.model = Bert_BiLSTM_CRF(tag2idx)
        self.model.load_state_dict(torch.load(BertBiLSTMCRF_MODEL_PATH, map_location=torch.device('cpu')), False)
        print("Bert-BiLSTM-CRF model loaded!")
        print("BertTokenizer model loading...")
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH, return_attention_mask=False)
        print("BertTokenizer model loaded!")

    def _decode(self, text):
        Pad_length = 200
        x = ['[CLS]'] + list(text) + ['[SEP]']
        self.text = x
        y = ['[CLS]'] + ['O' for _ in range(len(text))] + ['[SEP]']
        # z = len(x)
        self.x = self.tokenizer.convert_tokens_to_ids(x)
        self.y = [tag2idx[tag] for tag in y]
        self.z = len(self.y)
        self.x = torch.LongTensor([self.x + [0]*(Pad_length - self.z)])
        self.y = torch.LongTensor([self.y + [0]*(Pad_length - self.z)])
        self.z =  (self.x > 0)


    def predict(self, text):
        self._decode(text)
        Y_hat = []
        with torch.no_grad():
            x, y, z = self.x, self.y, self.z
            y_hat = self.model(x, y, z, is_test=True)
            for j in y_hat:
                Y_hat.extend(j)
        y_pred = [idx2tag[i] for i in Y_hat]
        return y_pred

    def parse(self, label_pred):
        Disease, Symptom, Check, Drug = set(), set(), set(), set()
        # text_label = []
        # for i in range(1, len(self.text)-1):
        #     text_label.append((self.text[i], label_pred[i]))
        # print(f"{Color.BLUE}模型输出{Color.END}：{text_label}")
        # print(f"{Color.BLUE}模型输出{Color.END}：{label_pred[1:len(self.text)-1]}")
        index = 1
        while index < len(self.text) - 1:
            if label_pred[index] == 'B-DISEASE':
                disease = self.text[index]
                while index + 1 < len(self.text) - 1 and label_pred[index+1] == 'I-DISEASE':
                    disease += self.text[index+1]
                    index += 1
                Disease.add(disease)
            elif label_pred[index] == 'B-SYMPTOM':
                symptom = self.text[index]
                while index + 1 < len(self.text) - 1 and label_pred[index+1] == 'I-SYMPTOM':
                    symptom += self.text[index+1]
                    index += 1
                Symptom.add(symptom)
            elif label_pred[index] == 'B-CHECK':
                check = self.text[index]
                while index + 1 < len(self.text) - 1 and label_pred[index+1] == 'I-CHECK':
                    check += self.text[index+1]
                    index += 1
                Check.add(check)
            elif label_pred[index] == 'B-DRUG':
                drug = self.text[index]
                while index + 1 < len(self.text) - 1 and label_pred[index+1] == 'I-DRUG':
                    drug += self.text[index+1]
                    index += 1
                Drug.add(drug)
            else:
                index += 1
        return list(Disease), list(Symptom), list(Check), list(Drug)


if __name__ == "__main__":
    bbc = BBC_Predict()
    print(f"\n{Color.BOLD}BERT-BiLSM-CRF 命名实体识别：{Color.END}")
    while True:
        text = input("请输入：")
        # '罗红霉素和头孢能一起吃吗'
        label_pred = bbc.predict(text)
        print(f"{Color.BLUE}模型输出{Color.END}：{label_pred[1:len(text) + 1]}")
        Disease, Symptom, Check, Drug = bbc.parse(label_pred)
        print(f"{Color.GREEN}实体识别{Color.END}：\n\t{Color.CYAN}疾病{Color.END}: {Disease}\
        \t{Color.CYAN}症状{Color.END}: {Symptom}\
        \t{Color.CYAN}检查{Color.END}:{Check}\
        \t{Color.CYAN}药物{Color.END}:{Drug}")
        print("-----------------------------------------------------------------------------------------------------")
        # print(label_pred)


