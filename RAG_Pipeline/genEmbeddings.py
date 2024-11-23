import sys
# sys.executable = 'miniconda3/envs/VLM/bin/python3.10'
# sys.path += ['/home/hice1/jwessell6/VLM/hw1', '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10', 
#     '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10/site-packages', '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10/lib-dynload']
import json
import os 
from pathlib import Path
from step1_video_to_frames_csv import video_to_images_with_transcript
from step2_csv_to_FAISS_embedding import gen_embeddings
from tsc_sampling import select_frames, update_csv
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
from torchvision import models
import torch
import clip
import glob
import argparse
import random

def generate_frames(split: str, video_code: str, frame_interval = 100, resnet = None, preprocess = None):
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

    folder_list=glob.glob(f"{results_dir}/frames")

    out_frames, output_result = select_frames(folder_list, f"{results_dir}/frames.csv", preprocess, resnet)

    return results_dir, csv 


def main():    
    parser = argparse.ArgumentParser(description="parser for step3")
    parser.add_argument('--f', type=int, default = 100, help='Frame sampling rate (100 = 1 frame per 100)')
    args = parser.parse_args()
    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
    model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc", torch_dtype=torch.float16).to("cuda:0")
    base_dir = os.path.expanduser("~") + "/scratch/VLM_Data/"
    with open(base_dir + "cleaned_db.json", 'r') as file:
        QA = json.load(file)

    egs = os.listdir(base_dir + "raw_videos/training")

    random.seed(10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet18_pretrained = models.resnet18(pretrained=True).to(device)
    resnet18_pretrained.fc = torch.nn.Identity()
    resnet18_pretrained.avgpool = torch.nn.Identity()
    resnet18_pretrained.eval()

    model2, preprocess = clip.load("ViT-B/32", device=device)

    for i in range(len(egs)):
        base_dir = os.path.expanduser("~") + "/scratch/VLM_Data/" 
        results_dir = f"{base_dir}eval_{args.f}/training/{egs[i]}"        
        if Path(f"{results_dir}/faiss_database.bin").exists():
            continue
        path, csv = generate_frames("training", egs[i], args.f, resnet=resnet18_pretrained, preprocess=preprocess)
        if path == "" or csv == "":
            continue
        gen_embeddings(path, csv, model, processor)

if __name__ == "__main__":
    main()
    