import faiss

#load faiss_index.bin

index = faiss.read_index("faiss_index_long.bin")

#get info

print(index.ntotal)

#print 0th element in index

print(index.reconstruct(0)[:1])
print(index.reconstruct(1)[:1])
print(index.reconstruct(2)[:1])

print(index.reconstruct(999)[:1])


#for element 0,1,2,999 get norm linalg

import numpy as np

#find two closest elements to 0th element
D, I = index.search(np.array([index.reconstruct(5)]), 3)
print(D)
print(I)

import pandas as pd
#load ./flickr1k.csv
df = pd.read_csv("flickr1k.csv")

#cols are index,image_path,raw_text
#show image for I[0] and I[1]
#/home/hice1/bpopper3/scratch/VLM/flickr8k/flickr8k_data/Flicker8k_Dataset/3637013_c675de7705.jpg
import matplotlib.pyplot as plt
def show_img(path,local_to_save_path):
    
    path = f'/home/hice1/bpopper3/scratch/VLM/flickr8k/flickr8k_data/Flicker8k_Dataset/{path}'
    
    img = plt.imread(path)
    #save locally
    plt.imsave(local_to_save_path,img)

show_img(df.iloc[I[0][0]]['image_path'],'og.jpg')
show_img(df.iloc[I[0][1]]['image_path'],'closest.jpg')
show_img(df.iloc[I[0][2]]['image_path'],'closest2.jpg')