#This script loads LLaVA so it's on a big GPU
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
from transformers import LlavaNextForConditionalGeneration, BitsAndBytesConfig
#import return_image_and_enhanced_query function from ./query.py

from query import return_image_and_enhanced_query
import argparse

# Create the parser

def RAG_question(path, question):
    QUERY = args.q
    conv,image = return_image_and_enhanced_query(QUERY, args.i)


    ########################## LOADING LLAVA ##########################
    #clear cuda
    torch.cuda.empty_cache()

    print("beginning to load LLaVA")
    #load on cuda:0

    llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", device="cuda:0")

    # specify how to quantize the model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    llava_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=quantization_config, device_map="auto")



    prompt = llava_processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = llava_processor(image,prompt, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = llava_model.generate(**inputs, max_new_tokens=200)

    print(f"enhanced answer:{llava_processor.decode(output[0], skip_special_tokens=True)}")


# start_secs = time.time()

# for i in tqdm(range(100)):
#     #takes about 5 second per iteration
#     print(f'elapsed time {time.time()-start_secs}')
#     prompt = llava_processor.apply_chat_template(conv, add_generation_prompt=True)
#     inputs = llava_processor(image,prompt, return_tensors="pt").to("cuda:0")

#     # autoregressively complete prompt
#     output = llava_model.generate(**inputs, max_new_tokens=100)

#     cprint(f"enhanced answer:{llava_processor.decode(output[0], skip_special_tokens=True)}",on_color='on_red')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for step3")

    # Define expected arguments
    parser.add_argument('--i', type=str, help='path to the input folder')
    parser.add_argument('--q', type=str, help='query string')

    args = parser.parse_args()