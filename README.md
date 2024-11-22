This repo is the repository for the group project done at Georgia Tech for our class CS 8803 VLM.
**We study how different sampling techniques combined with MultiModal RAG can change the performance of a LLaVA model answering various questions on videos (from [youcook](https://github.com/Jossome/YoucookQA))**

With Jack Wessell, Brandon Zhou, Azeez Ishaqui, and Brieuc Popper

## Basic high-level idea 

![rag](https://github.com/user-attachments/assets/4893de55-d34d-469d-a8c3-4ddcae213e00)

For a given video we will keep only some frames in a RAG database. This can be done with uniform sampling (take one frame every 1 second for example). Then for each frame, we get the transcript of the video matching that frame, and encode the **image/text pair** into a 512-dimensional crossmodal embedding (with BridgeTower).

Then if you have a question based on that video, instead of LLaVA trying to answer without any knowledge of the video, the question will be embedded into a 512-dimensional query, then a nearest-neighbor search will provide the closest match (also possible to retrieve more than one image/text pair). LLaVA is then given this image/text pair as additional information in it's context window, in addition to the original question.

This enables the model to answer some questions based on the video.

## The rest of the Readme is TODO / work in progress. Refer to the README in ./RAG_Pipeline for more

