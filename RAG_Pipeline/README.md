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