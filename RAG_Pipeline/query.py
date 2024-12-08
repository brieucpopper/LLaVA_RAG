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
    print('loading BRIDGETOWER ##############################################################')
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
    D, I = index.search(np.array([embedding]), n)
    #I[0] has the indexes of the n closest images
    return I[0]

def set_index(folder_path):
    global index
    global df
    index = faiss.read_index(folder_path+'/faiss_database.bin')
    df = pd.read_csv(folder_path+'/image_transcript_mapping.csv')

def return_image_and_enhanced_query(question,answer_a,answer_b,answer_c,answer_d,folder_path):
    global index
    global df
    set_index(folder_path)
    text_embedding = encode_text_query(question)
    
    #return query with 1 closest image/text pair to enhance the query

    closest_index = get_n_closest_index(text_embedding,1)
    print(f"Closest index is {closest_index[0]} for query '{question}'")
    return create_enhanced_conversation(df.iloc[closest_index[0]]['transcript_text'],question,Image.open(df.iloc[closest_index[0]]['image_path']),answer_a,answer_b,answer_c,answer_d)


def create_enhanced_conversation(textt,question,image,answer_a,answer_b,answer_c,answer_d):
    global index
    global df
    
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text" : f"This image is provided to guide your answer, you should refer to it to help you answer if needed. The image caption is '{textt}'.The original user query you have to answer to is '{original_query}'"}
        ],
    }],image