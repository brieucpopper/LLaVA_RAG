import json
import os
import numpy as np
from numpy.linalg import norm
import cv2
import pandas as pd
from tqdm import tqdm
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from termcolor import cprint

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="parser for step2")

# Define expected arguments
parser.add_argument('--i', type=str, help='path to the input folder')


# Parse the arguments
args = parser.parse_args()


INPUT_PATH = args.i
#should have frames folder, and the csv




processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc",cache_dir="/home/hice1/bpopper3/scratch/hug_face_files/hub_cache")
model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc",cache_dir="/home/hice1/bpopper3/scratch/hug_face_files/hub_cache")


def encode_single_image_text(image_path, text):
    print(image_path)
    print(text)
    image = Image.open(image_path)
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    return outputs['cross_embeds'].detach().numpy()
import pandas as pd

#load ./paired_image_data.csv
df = pd.read_csv(f'{INPUT_PATH}/image_transcript_mapping.csv')
#rows :             index,image_path,transcript_text,timestamp

#create empty FAISS index
import faiss

# Create embeddings for each row
embeddings = np.zeros((len(df), 512))
for _, row in tqdm(df.iterrows()):
    

    embedding = encode_single_image_text(row['image_path'], row['transcript_text'])
    #reshape to (512,)
    embedding = embedding.reshape(512)
    embeddings[row['index'],:] = embedding

    print(embedding.shape)

# Initialize FAISS index
dimension = 512  # dimension of your embeddings
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the FAISS index
index.add(embeddings)

# Create a mapping between FAISS ids and dataframe indices
id_to_index = {i: idx for i, idx in enumerate(df.index)}

# Save the FAISS index
faiss.write_index(index, f"{INPUT_PATH}/faiss_database.bin")
print('saved faiss index')