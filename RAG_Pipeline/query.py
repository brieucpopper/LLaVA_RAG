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
    embedding = np.array(embedding, dtype=np.float32)
    D, I = index.search(np.array([embedding]), n)
    #I[0] has the indexes of the n closest images
    return I[0]

def set_index(folder_path):
    global index
    global df
    index = faiss.read_index(folder_path+'/faiss_database.bin')
    df = pd.read_csv(folder_path+'/frames.csv')

def return_image_and_enhanced_query(query, text_embedding, folder_path):
    global index
    global df
    set_index(folder_path)
    #text query was pre-encoded offline. Embeddings will now be passed directly
    #text_embedding = encode_text_query(query)
    
    #return query with 1 closest image/text pair to enhance the query

    closest_index = get_n_closest_index(text_embedding,1)
    #print(f"Closest index is {closest_index[0]} for query '{query}'")
    print(closest_index, df.shape, index.ntotal)
    if closest_index[0] == -1:
        return create_enhanced_conversation("", query, None)
    return create_enhanced_conversation(df.iloc[closest_index[0]]['transcript_text'],query,Image.open(df.iloc[closest_index[0]]['image_path']))


def create_enhanced_conversation(text,original_query,image):
    global index
    global df
    return [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text" : f"This image is provided to guide your answer, you should refer to it to help you answer if needed. The image caption is '{text}'.The original user query you have to answer to is '{original_query}'"}
        ],
    }
],image