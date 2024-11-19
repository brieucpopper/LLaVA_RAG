Jack: How to use my script to run eval:

NOTE: At the top of each of my files are the lines "import sys" and some magic with the path/environment variables. This is necessary if you run on PACE-ICE in browser, which is how I developed these scripts. If you run locally or simply through ssh, these will get in you way so just commment out/delete them.

I have modified and re-factored the code Brieuc wrote to facilitate eval of RAG pipelines (the code could also be modified to work with standard baseline models as well with minimal effort. I will go over the directions for utilizing this repo here.

1). Run the script I added to ../yt-download (download_youcookii_videos.py). This will download all of the (available) youcookii videos to your target directory. I have this value hard-coded, so make sure to change to work with you directory.

2). Download the splits and questions/answers from https://github.com/Jossome/YoucookQA. This is a relatively large file so I did not push it alongside the rest of the Repo. 

3). Run the genEmbeddings.py file to both sample frames from your image using genEmbeddings.py. This takes one command line argument - the frame interval (default value 100). This script both samples frames from each video and creates the csv file and also generates the FAISS database.

4). Run question_embeddings.py. This step can also be done before step 3 as the two are independent of each other. When we actually perform RAG, we need to embed the question we are asking with bridgetower so that we can search the vector database we created in step 3. This is fine if we run eval once, but we will likely be doing so many times with different sampling rates and pipeline architectures. This step generates those embeddings once so that they can be re-used and save time in the future.

5). Run eval.py. This step will actually evaluate the RAG pipeline. It will go through the questions downloaded in part 2, generate the multiple choice structure, pull the embedding genereated in step 4, and query against the vector database from step 3. The generated answer and correct answer will be saved in a tuple and pickled. The actual calculation still needs to be performed, but since this takes longer to run than I can get on PACE, I made checkpoints with my work in this way.



---------------------------
For the Pipeline :
CALL ALL SCRIPTS FROM BASE DIRECTORY OF GIT REPO
example 

srun python RAG_Pipeline/step1_video_to_frames_csv.py




STEP 1 : Video ----> CSV
INPUT :
 - "--v" video_path (input video mp4)             
 - "--o" output_folder (desired output folder)
 - "--s" srt path (path to .srt transcript)
 - "--i" frame interval (for uniform sampling)


OUTPUT:
 - fills the output_folder with the CSV and a folder full of frames



STEP 2 :  CSV ----> FAISS
INPUT :
 - path to folder created by step 1 (--i)

OUTPUT :
 - writes to folder created by step 1 the FAISS


STEP 3 : CSV + FAISS + QUESTION ----> RAG-enhanced answer
INPUT:
 - user query (or question if we're doing VQA) (--q)
 - path to folder with FAISS And CSV (--i)

OUTPUT:
 - prints to terminal the LLAVA answer







-----------------------------------

module load anaconda3

conda activate vlm

cd tothegitrepo

srun python RAG_Pipeline/step1_video_to_frames_csv.py --v /home/hice1/bpopper3/scratch/LLaVA_RAG_PoC/yt-download/crepe.mp4 --o test_pipeline_withparser --s /home/hice1/bpopper3/scratch/LLaVA_RAG_PoC/yt-download/captions_videocooking.srt --i 100

srun python RAG_Pipeline/step2_csv_to_FAISS_embedding.py --i /home/hice1/bpopper3/scratch/LLaVA_RAG_PoC/test_pipeline_withparser

srun python RAG_Pipeline/step3_FAISS_augmented_chat.py --q "what color is the fruit next to the famous french dessert?" --i /home/hice1/bpopper3/scratch/LLaVA_RAG_PoC/test_pipeline_withparser

"How much salt should i add to the flour?"

"How many cups of milk should be added?"