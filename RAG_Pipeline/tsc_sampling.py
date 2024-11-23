# Lib
import os
import glob
import json
import tqdm
import natsort
import random
import csv

from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import clip

from torchvision import models

from config import config

#Original from LVNet

class loading_img(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return preprocess(Image.open(self.img_list[idx]))


def update_csv(folder, out_frames, config_csv):
    with open(config_csv, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    selected_frame_names = {os.path.basename(frame) for frame in out_frames}
    updated_rows = [row for row in rows if os.path.basename(row['image_path']) in selected_frame_names]
    for new_index, row in enumerate(updated_rows):
        row['index'] = new_index
    with open(config_csv, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

# select frames
def select_frames(folder_list, csv_path, preprocess, resnet18_pretrained):
    for folder in folder_list: 
        img_list = natsort.natsorted(glob.glob(f"{folder}/*.jpg"))
        img_feats = []
        # print("henlo")
        # print(folder, folder_list)
        # os.listdir(folder)
        # print(img_list)

        # random.seed(10)

        # device = "cuda" if torch.cuda.is_available() else "cpu"

        # resnet18_pretrained = models.resnet18(pretrained=True).to(device)
        # resnet18_pretrained.fc = torch.nn.Identity()
        # resnet18_pretrained.avgpool = torch.nn.Identity()
        # resnet18_pretrained.eval()

        # model, preprocess = clip.load("ViT-B/32", device=device)        
        img_set = loading_img(img_list)
        img_loader = DataLoader(img_set, batch_size=64, shuffle=False, num_workers=0)

        for imgtensor in img_loader: img_feats.append(imgtensor)
        # print(folder_list, print(len(img_list)))
        print(img_feats)
        if len(img_feats) == 0:
            print("SKIPPED: Received No Frames")
            return None, None
        img_feats = torch.cat(img_feats, dim=0).to(device)

        with torch.no_grad():
            featuremap = resnet18_pretrained(img_feats)
            frame_num = featuremap.shape[0]

            dist_list = []
            for img_feat in featuremap: dist_list.append(torch.mean(torch.sqrt((featuremap-img_feat)**2), dim=-1))
            dist_list = torch.cat(dist_list).reshape(frame_num, frame_num)

            idx_list = [_ for _ in range(frame_num)]

            # print(dist_list)
            loop_idx = 0
            out_frames = []

            output_results = []
            while len(idx_list) > 5:
                dist_idx = idx_list.pop(0)

                data = dist_list[dist_idx, idx_list].softmax(dim=-1)
                mu, std = torch.mean(data), torch.std(data)
                pop_idx_list = torch.where(data < mu-std*(np.exp(1-loop_idx/config.divlam)))[0].detach().cpu().numpy()
                result = list(np.array(idx_list)[pop_idx_list])
                result.append(dist_idx)
                output_results.append(result)

                num_picks = config.num_picks
                if len(result) > num_picks:
                    idx_result_list = sorted(random.sample(result, num_picks)) 
                    img_list = np.array(img_list)
                    idx_result_list = np.array(idx_result_list)
                    out_frames.extend(img_list[idx_result_list])
                else:
                    idx_result_list = sorted(result)
                    img_list = np.array(img_list)
                    idx_result_list = np.array(idx_result_list)
                    out_frames.extend(img_list[idx_result_list])

                loop_idx += 1
                
                for pop_idx in reversed(pop_idx_list): idx_list.pop(pop_idx)
    
        update_csv(folder, out_frames, csv_path)
        for image in img_list:
            if image not in out_frames:
                os.remove(image)
    return out_frames, output_results


# Init
# random.seed(10)

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet18_pretrained = models.resnet18(pretrained=True).to(device)
resnet18_pretrained.fc = torch.nn.Identity()
resnet18_pretrained.avgpool = torch.nn.Identity()
resnet18_pretrained.eval()

model, preprocess = clip.load("ViT-B/32", device=device)

# output_results = []
# folder_list=glob.glob(config.img_folder)

# out_frames, output_result = select_frames(folder_list, config.csv, preprocess, resnet18_pretrained)
# output_results.append(output_result)
# print(output_results)    
