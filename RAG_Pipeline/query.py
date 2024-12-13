import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import pandas as pd
from tqdm import tqdm
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from termcolor import cprint
import pandas as pd

#by default dont load if not needed
processor = None
model = None
is_loaded = False
def load_bridgetower():
    global index
    global df
    print('############################################# loading BRIDGETOWER ##############################################################')
    is_loaded = True
    global processor
    global model
    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
    model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")


def encode_text_query(text):
    global index
    global df
    if not is_loaded:
        load_bridgetower()
    #create random Image 32x32
    image = Image.fromarray(np.random.randint(0,255,(32,32,3),dtype=np.uint8))
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    #reshape to (512,)

    return outputs['text_embeds'].detach().numpy().reshape(-1)


def get_n_closest_index(embedding,n):
    global index
    global df
    num = n 
    if index.ntotal < n:
        #just return the best option
        D, I = index.search(np.array([embedding]), 1)
        return I[0][0]
    while num <= index.ntotal:
        D, I = index.search(np.array([embedding]), num)
        for idx in I[0]:
            #index found is divisible by n. This means that it is a frame we would have sampled if we had sampled at a larger multiple of our base rate (in this case, 25)
            if idx % n == 0:
                return idx
        #otherwise, double our number of searches
        num *= 2 
    return I[0][0]

def set_index(folder_path):
    global index
    global df
    index = faiss.read_index(folder_path+'/faiss_database.bin')
    df = pd.read_csv(folder_path+'/frames.csv')

def return_image_and_enhanced_query(query, text_embedding, answers, folder_path, sampling, blind):
    global index
    global df
    set_index(folder_path)
    #text query was pre-encoded offline. Embeddings will now be passed directly
    #text_embedding = encode_text_query(query)
    
    #return query with 1 closest image/text pair to enhance the query
    closest_index = get_n_closest_index(text_embedding, sampling)
    #print(f"Closest index is {closest_index[0]} for query '{query}'")
    images = []
    #Video input takes the closest match then a number of prior frames and a number of following frames.
    for i in range(closest_index - 4*sampling, closest_index + 5*sampling, sampling):
        if i < 0 or i >= len(df):
            continue
        if blind:
            images.append(Image.fromarray(np.random.randint(0,255,(32,32,3),dtype=np.uint8)))
            continue
        images.append(Image.open(df.iloc[i]['image_path']))
    return create_enhanced_conversation(df.iloc[closest_index]['transcript_text'], query, answers, images)


def create_enhanced_conversation(text, original_query, answers, video):
    global index
    global df
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text" : f"""A user has the following question regarding: 'What is 2+2'
            This transcript can also help you answer : 'Addition is a mathematical commutative operation that' .
            It is possible that the video helps you answer the question, but it is possible that it does not. Given the question, the video, and the transcript,
            what is the answer to the question? Your options are:
            Answer A: 4
            Answer B: 6
            Answer C: 9
            Answer D: 12
            Answer E: 15 """}
        ],
    },
    {
        "role": "assistant",
        "content": [
   
            {"type": "text", "text" : "A"}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text" : f"""The short video provided is a clip extracted from a larger video. A user has the following question
            regarding the video: {original_query}. Given the question, the video, and the transcript provided,
            what is the answer to the question? You options are:
            Answer A: {answers[0]}
            Answer B: {answers[1]}
            Answer C: {answers[2]}
            Answer D: {answers[3]}
            Answer E: {answers[4]}"""}
        ],
    }],video