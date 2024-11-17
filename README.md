This repo is a proof of concept on how to do simple RAG for a multimodal setting (using LLaVA).

Inspired by the deeplearning.ai MOOC https://learn.deeplearning.ai/login?callbackUrl=https%3A%2F%2Flearn.deeplearning.ai%2Fcourses%2Fmultimodal-rag-chat-with-videos

You can use the environment.yml conda env to run the scripts, you will need a GPU that fits the models in memory though


## Getting VQA to run with uniform sampling RAG

1. Create a CSV with columns ['index', 'image_path', 'transcript_text','timestamp'] thanks to step1
2. Run step 2 to go from CSV ---> FAISS embedding

## Proof of concept part (./simple_runnable_PoC)
the script proof_of_concept_RAG.py illustrates the whole process for only 4 images
It will find the crossmodal embeddings of the 4image/text pairs (512-dim vectors).
Then you ask LLaVA a pure-text question, your text question gets embedded into a 512 dim vector in the same space, and you find the most relevant image/text pair among the 4. In the end LLaVA answer not your original query, but an "enhanced" query that is the original query with the added retrieved image and text as context.

## Scaling with Flickr8k
To see how this proof of concept scales, I use the flickr8k dataset of paired images and text.

You can download the flickr8k data easily by running the dl_images.py script
then 


The steps are the following

1. create a csv with 1000 (or however you want) image/text pair info from flickr. The csv has index, image_path,raw_text as keys and is useful to sync up between scripts. For this run create_csv.py

2. compute the embeddings with Bridgetower (using the CSV as input). For this run the script compute_embeddings.py

3. OPTIONAL : Just convert a query into a "enhanced query". For this run script query.py (No LLaVA yet in that script, but utility functions to create an enhanced query)

4. Run scripts/chat_with_RAG after editing the query to see what the models answers to your query ! It will answer thanks to the query AND the most relevant image/text pair it found with RAG
