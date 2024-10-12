# imports
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

################# declare images for the examples ##############

url1='http://farm3.staticflickr.com/2519/4126738647_cc436c111b_z.jpg'
cap1='A motorcycle sits parked across from a herd of livestock'

url2='http://farm3.staticflickr.com/2046/2003879022_1b4b466d1d_z.jpg'
cap2='Motorcycle on platform to be worked on in garage'

url3='http://farm1.staticflickr.com/133/356148800_9bf03b6116_z.jpg'
cap3='this image depicts a cat'

img1 = {
  'flickr_url': url1,
  'caption': cap1,
  'image_path' : './shared_data/motorcycle_1.jpg'
}

img2 = {
    'flickr_url': url2,
    'caption': cap2,
    'image_path' : './shared_data/motorcycle_2.jpg'
}

img3 = {
    'flickr_url' : url3,
    'caption': cap3,
    'image_path' : './shared_data/cat_1.jpg'
}

img4 = {
    'caption' : "overall, I think it would be nice if everybody moves away from RAG 1.0 to frozen Frankenstein RAG and moves towards this much more optimized version RAG 2.0, so it's really about systems over models. It's not just your language model and your retriever and they're separate. It's about thinking from a systems perspective about the entire thing and the problem you're trying to solve. And so I think that really is the way that in deep learning things have always progressed, where if you optimize the system end-to-end, that's always going to win out.",
    'image_path' : './shared_data/stanford.jpg'
}


text1 = img1['caption']
text2 = img2['caption']
text3 = img3['caption']
test4 = img4['caption']

img1 = Image.open(img1['image_path'])
img2 = Image.open(img2['image_path'])
img3 = Image.open(img3['image_path'])
img4 = Image.open(img4['image_path'])

images=[img1, img2, img3, img4]
texts=[text1, text2, text3, test4]


###################################################################
processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")


def encode_single_image_text(image, text):

    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    return outputs['cross_embeds'].detach().numpy()


def encode_single_text(query):
    black_img= Image.new('RGB', (50, 50), (0, 0, 0)) #dummy image according to https://huggingface.co/BridgeTower/bridgetower-large-itm-mlm-itc/discussions/3
    encoding = processor(black_img, query, return_tensors="pt")
    outputs = model(**encoding)
    return outputs['text_embeds'].detach().numpy()


def create_RAG_db(images,texts):
    #return a dict with key idx and value as the cross modal embeddings, og images and texts

    cross_modal_embeddings = {}

    for idx, (image, text) in enumerate(zip(images, texts)):
        cross_modal_embeddings[idx] = {
            'cross_modal_embeddings': encode_single_image_text(image, text),
            'image': image,
            'text': text
        }
    return cross_modal_embeddings

def conversation_from_plain_text(plain_text):
    return [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": plain_text},
        ],
    },
]

def conversation_from_enhanced_query(image,text,original_query):
    return [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text" : f"This image is provided to guide your answer, you should refer to it to help you answer if needed. The image caption is '{text}'.The original user query you have to answer to is '{original_query}'"}
        ],
    }
]


#################################################################################

#now get simple query (plain - text) and get enhanced query (plain query + retrieved image and text)

def get_enhanced_query(query, db, k=1):

    # get query embedding
    embedding = encode_single_text(query)

    # get index in db of closest image-text pair with cosine similarity
    max_sim = -1
    max_idx = -1
    for idx in db.keys():
        cross_modal_embeddings = db[idx]['cross_modal_embeddings']
        sim = np.dot(embedding, cross_modal_embeddings.T) / (norm(embedding) * norm(cross_modal_embeddings))
        db[idx]['similarity'] = sim
        if sim > max_sim:
            max_sim = sim
            max_idx = idx
    return db[max_idx]['image'], db[max_idx]['text'], query

db = create_RAG_db(images,texts)
query = 'What are the 4 main points highlighted in the image? Also, what is the speaker saying about RAG 1.0 ?' ############################# MODIFY THE QUERY HERE

enhanced_query = get_enhanced_query(query, db)
cprint(f"default query: {query}",on_color='on_green')
cprint(f"enhanced query: {enhanced_query}",on_color='on_red')



# free cuda memory
torch.cuda.empty_cache()


#################################################################################

llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

llava_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
llava_model.to("cuda:0")

# ask it first default query ###############################################################

conversation = conversation_from_plain_text(query)
enhanced_conversation= conversation_from_enhanced_query(enhanced_query[0],enhanced_query[1],query)

cprint(f"Default convo: {conversation}",on_color='on_green')
cprint(f"Enhanced convo: {enhanced_conversation}",on_color='on_red')



prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = llava_processor(prompt, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
output = llava_model.generate(**inputs, max_new_tokens=100)


cprint(f"default answer:{llava_processor.decode(output[0], skip_special_tokens=True)}",on_color='on_green')






image = enhanced_query[0]
prompt2 = llava_processor.apply_chat_template(enhanced_conversation, add_generation_prompt=True)
inputs = llava_processor(image,prompt2, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
output2 = llava_model.generate(**inputs, max_new_tokens=100)

cprint(f"enhanced answer:{llava_processor.decode(output2[0], skip_special_tokens=True)}",on_color='on_red')





