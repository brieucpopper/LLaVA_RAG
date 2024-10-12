#create a csv with


#1. index
#2. image path (jpg)
#3. raw text caption
#4. 512-dim image/text cross-modal embedding (can be initialized empty)

import pandas

#/home/hice1/bpopper3/scratch/VLM/flickr8k_data/Flickr8k.token.txt


FLICKR_TXT = './flickr8k_data/Flickr8k.token.txt'

#go over each line in the txt with a stride of 5 until 20 000

with open(FLICKR_TXT, 'r') as f:
    lines = f.readlines()

def extract_image_path(line):
    return line.split('#')[0]

def extract_text(line):
    return line.split('\t')[1].split('.')[0].replace('\n', ' ')

# example line
#989851184_9ef368e520.jpg#3	A black dog holds a small white dumbbell in its mouth .


#init empty dataframe with good amount of cols
arr=[]

for i in range(0, 5000, 5):
    image_path = extract_image_path(lines[i])
    text = extract_text(lines[i])
    arr.append({'index': i//5, 'image_path': image_path, 'raw_text': text})


df = pandas.DataFrame(arr)
#name columns
df.columns = ['index', 'image_path', 'raw_text']

df.to_csv('paired_image_data.csv', index=False)
    
    