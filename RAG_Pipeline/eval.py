import sys
sys.executable = 'miniconda3/envs/VLM/bin/python3.10'
sys.path += ['/home/hice1/jwessell6/VLM/hw1', '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10', 
    '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10/site-packages', '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10/lib-dynload']
import json
import os 
from pathlib import Path
from step3_FAISS_augmented_chat import RAG_question
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
import pickle
def main():
    llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", device="cuda:0")
    # specify how to quantize the model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    llava_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=quantization_config, device_map="auto")
    
    ##### CHANGE YOUR DIRECTORY STRUCTURE HERE #########
    base_dir = os.path.expanduser("~") + "/scratch/VLM_Data/"
    egs = os.listdir(base_dir + "raw_videos/training")
    with open(base_dir + "cleaned_db.json", 'r') as file:
        QA = json.load(file)
    db_path = base_dir + "eval_100/training/"
    ###################################################
    for i in range(len(egs)):
        video_code = egs[i]
        QA_pairs = QA[video_code]['QApairs']
        
        #skip over videos we could not properly process or ones we have already completed
        if not Path(db_path + video_code).exists() or Path(db_path + video_code + "/results.pkl").exists():
            continue

        with open(db_path + video_code + "/answers.pkl", "rb") as file:
            Q_embs = pickle.load(file)
        results = []
        for j in range(len(Q_embs)):
            question = QA_pairs[j]['question'] + " is the answer: " 
            for answer in QA_pairs[j]['alternatives']:
                question += answer
                question += ", "
            question += "or "
            question += QA_pairs[j]['answer']
            answer = QA_pairs[j]['answer']
            results.append((answer, RAG_question(db_path + video_code, question, Q_embs[j], llava_processor, llava_model)))
        with open(db_path + video_code + "/results.pkl", "wb") as file:
            pickle.dump(results, file)         
        
if __name__ == "__main__":
    main()