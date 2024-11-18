import json
import os
import numpy as np
from numpy.linalg import norm
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from termcolor import cprint
import faiss
import argparse

import pandas as pd

def encode_single_image_text(image_path, text, processor, model):
    image = Image.open(image_path)
    encoding = processor(image, text, return_tensors="pt").to("cuda:0")
    outputs = model(**encoding)
    return outputs['cross_embeds'].cpu().detach().numpy()

#moved model and processor to main function call to save time on loading/reloading model and sending weights to GPU
def gen_embeddings(path: str, csv:str, model, processor):
    
    #load ./paired_image_data.csv
    df = pd.read_csv(f'{path}/{csv}')
    #rows :             index,image_path,transcript_text,timestamp

    #create empty FAISS index


    # Create embeddings for each row
    embeddings = np.zeros((len(df), 512))
    for _, row in tqdm(df.iterrows()):
        embedding = encode_single_image_text(row['image_path'], row['transcript_text'], processor, model)
        #reshape to (512,)
        embedding = embedding.reshape(512)
        embeddings[row['index'],:] = embedding

    print(embedding.shape)

    # Initialize FAISS index
    dimension = 512  # dimension of your embeddings
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to the FAISS index
    index.add(embeddings)

     #Create a mapping between FAISS ids and dataframe indices
    id_to_index = {i: idx for i, idx in enumerate(df.index)}

    # Save the FAISS index
    faiss.write_index(index, f"{path}/faiss_database.bin")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="parser for step2")
    # Define expected arguments
    parser.add_argument('--i', type=str, help='path to the input folder')
    parser.add_argument('--n', type=str, help='name of target csv file')
    # Parse the arguments
    args = parser.parse_args()
    gen_embeddings(args.i)
