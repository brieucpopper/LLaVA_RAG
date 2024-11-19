'''
During evaluation, we will need to embed the questions using bridgetower. This will be (essentially, up to randomness) the same for every iteration of evaluation that we run using any RAG-based system. So, we should do this step once then re-use the embeddings to save time in the long-run.
'''
import sys
#comment out if needed - pace ice path variables are different in-browser
sys.executable = 'miniconda3/envs/VLM/bin/python3.10'
sys.path += ['/home/hice1/jwessell6/VLM/hw1', '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10', 
    '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10/site-packages', '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10/lib-dynload']
import json
import os 
import pickle
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
import torch 
from PIL import Image
import numpy as np
from pathlib import Path

def main():
    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
    model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc", torch_dtype=torch.float16).to("cuda:0")
    base_dir = os.path.expanduser("~") + "/scratch/VLM_Data/"
    with open(base_dir + "cleaned_db.json", 'r') as file:
        QA = json.load(file)
    egs = os.listdir(base_dir + "raw_videos/training")
    for i in range(len(egs)):
        video_code = egs[i]
        if not Path(base_dir + "eval_100/training/" + video_code).exists():
            continue
        QA_pairs = QA[video_code]['QApairs']
        embeddings = []
        image = Image.fromarray(np.random.randint(0,255,(32,32,3),dtype=np.uint8))
        for j in range(len(QA_pairs)):
            question = QA_pairs[j]['question'] + " is the answer: " 
            for answer in QA_pairs[j]['alternatives']:
                question += answer
                question += ", "
            question += "or "
            question += QA_pairs[j]['answer']
            encoding = processor(image, question, return_tensors="pt").to("cuda:0")
            outputs = model(**encoding)
            embeddings.append(outputs['text_embeds'].cpu().detach().numpy().reshape(-1))
        embeddings = np.stack(embeddings, axis = 0)
        print(embeddings.shape)
        with open(base_dir + "eval_100/training/" + video_code + "/answers.pkl", "wb") as file:
            pickle.dump(embeddings, file)
if __name__ == "__main__":
    main()
