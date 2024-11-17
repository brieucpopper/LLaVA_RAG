For the Pipeline :
CALL ALL SCRIPTS FROM BASE DIRECTORY OF GIT REPO
example 

srun python RAG_Pipeline/step1_video_to_frames_csv.py


STEP 1 : Video ----> CSV
INPUT :
 - video_path (input video mp4)
 - output_folder (desired output folder)
 - srt path (path to .srt transcript)
 - csv output file name
 - frame interval (for uniform sampling)


OUTPUT:
 - fills the output_folder with the CSV and a folder full of frames



STEP 2 :  CSV ----> FAISS
INPUT :
 - path to folder created by step 1

OUTPUT :
 - writes to folder created by step 1 the FAISS


STEP 3 : CSV + FAISS + QUESTION ----> RAG-enhanced answer
INPUT:
 - user query (or question if we're doing VQA)
 - path to folder with FAISS And CSV

OUTPUT:
 - prints to terminal the LLAVA answer