This repo is a proof of concept on how to do simple RAG for a multimodal setting (using LLaVA).

Inspired by the deeplearning.ai MOOC https://learn.deeplearning.ai/login?callbackUrl=https%3A%2F%2Flearn.deeplearning.ai%2Fcourses%2Fmultimodal-rag-chat-with-videos

You can use the environment.yml conda env to run the scripts, you will need a GPU that fits the models in memory though


## Proof of concept part (./simple_runnable_PoC)
the script proof_of_concept_RAG.py illustrates the whole process for only 4 images
It will find the crossmodal embeddings of the 4image/text pairs (512-dim vectors).
Then you ask LLaVA a pure-text question, your text question gets embedded into a 512 dim vector in the same space, and you find the most relevant image/text pair among the 4. In the end LLaVA answer not your original query, but an "enhanced" query that is the original query with the added retrieved image and text as context.

## Scaling with Flickr8k
To see how this proof of concept scales, I use the flickr8k dataset of paired images and text.

You can download the flickr8k data easily by running the dl_images.py script

