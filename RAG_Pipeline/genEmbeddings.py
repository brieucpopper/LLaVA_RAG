import sys
sys.executable = 'miniconda3/envs/VLM/bin/python3.10'
sys.path += ['/home/hice1/jwessell6/VLM/hw1', '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10', 
    '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10/site-packages', '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10/lib-dynload']
import json
import os 
from pathlib import Path
from step1_video_to_frames_csv import video_to_images_with_transcript
from step2_csv_to_FAISS_embedding import gen_embeddings
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning

from torchvision import models
import clip

import torch
import argparse
from tsc_sampling import select_frames
import shutil
import pandas as pd

def generate_frames(split: str, video_code: str, frame_interval = 100):
    base_dir = os.path.expanduser("~") + "/scratch/VLM_Data/" 
    dir_contents = os.listdir(base_dir + "raw_videos/" + split + "/" + video_code)
    if len(dir_contents) != 2:
        print(f"Files incomplete for {video_code}")
        return "", ""
    contents = f"{base_dir}raw_videos/{split}/{video_code}/"
    if ".srt" in dir_contents[0]:
        transcript_path = contents +  dir_contents[0]
        video_path =  contents + dir_contents[1]
    else:
        transcript_path =  contents + dir_contents[1]
        video_path =  contents + dir_contents[0]
    results_dir = f"{base_dir}eval_{frame_interval}/{split}/{video_code}"
    if not Path(results_dir).exists():
        Path(results_dir).mkdir(parents=True)
    csv = "frames.csv"
    video_to_images_with_transcript(video_path, results_dir, transcript_path, csv, frame_interval)
    return results_dir, csv 


def main(rate, tsc):    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
    model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc", torch_dtype=torch.float16).to("cuda:0")
    base_dir = os.path.expanduser("~") + "/scratch/VLM_Data/"
    with open(base_dir + "cleaned_db.json", 'r') as file:
        QA = json.load(file)

    egs = os.listdir(base_dir + "raw_videos/training")
    
    #if temporal scene clustering, set up alternate folders for copying later.
    if tsc:
        resnet = models.resnet101(pretrained=True).to(device)
        resnet.fc = torch.nn.Identity()
        resnet.avgpool = torch.nn.Identity()
        resnet.eval()
        _, preprocess = clip.load("ViT-B/32", device=device)
        alt_paths = []
        for i in [25, 50, 100]:
            alt_path = f"{base_dir}tsc_eval_{i}/training"
            if not Path(alt_path).exists():
                Path(alt_path).mkdir(parents=True)
            alt_paths.append(alt_path)
    
    for i in range(len(egs)):
        base_dir = os.path.expanduser("~") + "/scratch/VLM_Data/" 
        results_dir = f"{base_dir}eval_{rate}/training/{egs[i]}"        
        if Path(f"{results_dir}").exists():
            continue
        
        #if we perform temporal scene clustering, we need to run it in between generate_frames and gen_embeddings.
        #it is a bit more complex of a process. Since I only want to run it once, I use it to generate the directories for all three experiments I will run.
        if tsc:
            #perform normal sampling (1 frame per 10 in our current scenario)
            path, csv = generate_frames("training", egs[i], rate)
            if path == "" or csv == "":
                continue
            for j in range(len(alt_paths)):
                new_path = alt_paths[j] + f"/{egs[i]}"
                shutil.copytree(path, new_path)
                out_frames, out_results = select_frames(new_path + "/frames", 25, (int) (8/(2**j)), preprocess, resnet)
                frames = []
                for frame in out_frames:
                    frames.append(frame[frame.find('frame_'):])
                csv = pd.read_csv(new_path + "/frames.csv")
                #sample csv. Need to reset index.
                csv = csv[csv['image_path'].str.extract(r'([^/]+)$', expand=False).isin(frames)].reset_index(drop=True)
                csv.drop('index', axis=1, inplace=True)
                csv['index'] = csv.index
                csv.to_csv(new_path + "/frames.csv", index=False)
                for image in os.listdir(new_path + "/frames"):
                    if image not in frames:
                        os.remove(new_path + "/frames/" + image)
                gen_embeddings(new_path, 'frames.csv', model, processor)
            continue
        path, csv = generate_frames("training", egs[i], rate)
            
        if path == "" or csv == "":
            continue
        gen_embeddings(path, csv, model, processor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for step3")
    parser.add_argument('--f', type=int, default = 100, help='Frame sampling rate (100 = 1 frame per 100)')
    parser.add_argument('--tsc', action='store_true', help = "Whether or not to use temporal scene clustering between steps 1 and 2")
    args = parser.parse_args()
    main(args.f, args.tsc)
    