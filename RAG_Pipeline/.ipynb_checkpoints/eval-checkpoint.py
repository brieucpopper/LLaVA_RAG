import sys
sys.executable = 'miniconda3/envs/VLM/bin/python3.10'
sys.path += ['/home/hice1/jwessell6/VLM/hw1', '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10', 
    '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10/site-packages', '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10/lib-dynload']
import json
import os 
from pathlib import Path
from step3_FAISS_augmented_chat import RAG_question
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig
import torch
import pickle
import argparse

def main(f, blind):
    
    # specify how to quantize the model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")
    model.eval()
    ##### CHANGE YOUR DIRECTORY STRUCTURE HERE #########
    base_dir = os.path.expanduser("~") + "/scratch/VLM_Data/"
    egs = os.listdir(base_dir + "raw_videos/training")
    with open(base_dir + "cleaned_db.json", 'r') as file:
        QA = json.load(file)
    db_path = base_dir + f"eval_{f}/training/"
    filename = "/results.pkl"
    if blind:
        filename = "/results_blind.pkl"
    ###################################################
    fails = []
    for i in range(len(egs)):
        video_code = egs[i]
        QA_pairs = QA[video_code]['QApairs']
        
        #skip over videos we could not properly process or ones we have already completed. Not sure why the following video codes don't work
        if not Path(db_path + video_code).exists(): #or Path(db_path + video_code + filename).exists() or video_code in ["yLJMpYQg_gA", "_xOx9hkJoBk", "s99K_WyajB8"]:
            continue
        print(video_code)
        with open(db_path + video_code + "/answers.pkl", "rb") as file:
            Q_embs = pickle.load(file)
        questions = []
        embeddings = []
        answers = []
        results = []
        for j in range(len(Q_embs)):
            question = QA_pairs[j]['question']
            answers = []
            for answer in QA_pairs[j]['alternatives']:
                answers.append(answer)
            answers.append(QA_pairs[j]['answer'])
            try:
                results.append((answers[-1], RAG_question(db_path + video_code, question, Q_embs[j], answers, processor, model, blind)))
            except:
                fails.append(video_code)
                continue
        with open(db_path + video_code + filename, "wb") as file:
            pickle.dump(results, file)         
    with open(db_path + "failures.pkl", "wb") as errors:
        pickle.dump(fails, errors)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for step3")

    # Define expected arguments
    parser.add_argument('--f', type=int, help = 'Sampling rate of interest')
    parser.add_argument('-b', action = 'store_true', help='Whether to use image-blind evaluation')
    args = parser.parse_args()
    main(args.f, args.b)