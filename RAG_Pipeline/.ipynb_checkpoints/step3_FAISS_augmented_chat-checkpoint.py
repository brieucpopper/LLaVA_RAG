#This script loads LLaVA so it's on a big GPU
import json
import os
import numpy as np
from numpy.linalg import norm
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from PIL import Image
import requests
from termcolor import cprint
#import return_image_and_enhanced_query function from ./query.py

from query import return_image_and_enhanced_query
import argparse

# Create the parser

def batch_inputs(texts, images):
    input_ids = []
    attention_mask = []
    pixel_values = []
    image_sizes = []
    for eg in texts:
        input_ids.append(torch.squeeze(eg['input_ids']))
        attention_mask.append(torch.squeeze(eg['attention_mask']))
    for eg in images:
        pixel_values.append(torch.squeeze(torch.tensor(np.array(eg['pixel_values']))))
        image_sizes.append(torch.squeeze(torch.tensor(np.array(eg['image_sizes']))))
    results = {
        'input_ids' : torch.stack(input_ids).to("cuda:0"),
        'attention_mask' : torch.stack(attention_mask).to("cuda:0"),
        'pixel_values' : torch.stack(pixel_values).to("cuda:0"),
        'image_sizes' : torch.stack(image_sizes).to("cuda:0")
    }
    return results

def RAG_question(path, question, embeddings, answers, processor, model, blind = False):
    conv, video = return_image_and_enhanced_query(question, embeddings, answers, path, blind)
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(videos=[video], text = prompt, return_tensors="pt").to("cuda:0", torch.float16)
    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for step3")

    # Define expected arguments
    parser.add_argument('--i', type=str, help='path to the input folder')
    parser.add_argument('--q', type=str, help='query string')

    args = parser.parse_args()
    torch.cuda.empty_cache()
    llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", device="cuda:0")

    # specify how to quantize the model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    llava_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=quantization_config, device_map="auto")
    RAG_question(args.i, args.q, llava_processor, llava_model)