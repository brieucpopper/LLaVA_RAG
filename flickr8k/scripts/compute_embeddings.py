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

processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")


def encode_single_image_text(image_path, text):
    image = Image.open(image_path)
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    return outputs['cross_embeds'].detach().numpy()
import pandas as pd

#load ./paired_image_data.csv
df = pd.read_csv('./paired_image_data.csv')
#index,image_path,raw_text

#create empty FAISS index
import faiss



# Create embeddings for each row
embeddings = np.zeros((len(df), 512))
for _, row in tqdm(df.iterrows()):
    

    embedding = encode_single_image_text(f"./flickr8k_data/Flicker8k_Dataset/{row['image_path']}", row['raw_text'])
    #reshape to (512,)
    embedding = embedding.reshape(512)
    embeddings[row['index'],:] = embedding

    print("for iamge with caption")
    print(row['raw_text'])
    print(f" the embedding starts with {embedding[:2]}")

    print(embedding.shape)

# Initialize FAISS index
dimension = 512  # dimension of your embeddings
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the FAISS index
index.add(embeddings)

# Create a mapping between FAISS ids and dataframe indices
id_to_index = {i: idx for i, idx in enumerate(df.index)}

# Save the FAISS index
faiss.write_index(index, "faiss_index_long.bin")