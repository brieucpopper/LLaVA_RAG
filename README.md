This repo is the repository for the group project done at Georgia Tech for our class CS 8803 VLM.
**We study how different sampling techniques combined with MultiModal RAG can change the performance of a LLaVA model answering various questions on videos (from [youcook](https://github.com/Jossome/YoucookQA))**

With Jack Wessell, Brandon Zhou, Azeez Ishaqui, and Brieuc Popper

## Basic high-level idea 


![gordon](https://github.com/user-attachments/assets/9856bf76-3fa2-412b-a374-1412bdf07f90)




For a given video we will keep only some frames in a RAG database. This can be done with uniform sampling (take one frame every 1 second for example). Then for each frame, we get the transcript of the video matching that frame, and encode the **image/text pair** into a 512-dimensional crossmodal embedding (with BridgeTower).

Then if you have a question based on that video, instead of LLaVA trying to answer without any knowledge of the video, the question will be embedded into a 512-dimensional query, then a nearest-neighbor search will provide the closest match (also possible to retrieve more than one image/text pair). LLaVA is then given this image/text pair as additional information in it's context window, in addition to the original question.

![rag](https://github.com/user-attachments/assets/4893de55-d34d-469d-a8c3-4ddcae213e00)

This enables the model to answer some questions based on the video.


## What we did in this project
We implemented the whole multimodal pipeline from scratch, using only LLaVA (for multimodal chat) and BridgeTower (to embed an image/text pair or just text into a 512-dimensional latent representation). Some of the interesting challenges we solved include :
 - Creating a RAG that relies not only on images (frames from the video) but also on an aligned transcript. The ablations confirm our models use both information from the transcripts AND from the visual frames.
 - Forced the model to answer A, B, C or D (the questions are multiple choice) with few-shot prompting combined with constrained output (done manually on LLaVA-1.6 by looking at the probability of each token post-softmax)
 - Implemented TSC (temporal scene clustering) to go from a lot of frames (high sampling rate) to a lot less frames by iteratively eliminating redundant frames
 - Using FAISS for efficient vector storage and retrieval of top-k nearest neighbors
 - Training on a GPU Cluster, with all the challenges associated (different GPUs available at different times, limited total script running time...)
 - Ran various experiments that confirm that when given an equal frame budget (e.g. 50 frames are allowed for a 5 minute video, our RAG performs better with TSC sampling than with uniform sampling). Ran various ablations etc.

## Main result

![image](https://github.com/user-attachments/assets/a377bf81-1a16-4297-8b4a-8a0f595c3eaa)


The notebook here : notebook_pipeline_custom_questions/handmade_questions_pipeline_GPU.ipynb has all the code to run the whole pipeline from start to end on our custom questions dataset (similar to YouCookQA)


