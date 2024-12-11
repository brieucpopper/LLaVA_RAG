# Script for downloading youcookii videos
# Written by Luowei Zhou, 09/23/2017
# Contact luozhou@umich.edu if you have trouble downloading some private/unavailable videos

# Requirement: install youtube-dl (https://github.com/rg3/youtube-dl/)

import os
from pytubefix import YouTube
from pytubefix.cli import on_progress

raw_path = os.path.expanduser("~") + "/scratch/VLM_Data"
dataset_root  = raw_path + '/raw_videos'
vid_file_lst = [raw_path + '/splits/train_list.txt', raw_path + '/splits/val_list.txt', raw_path + '/splits/test_list.txt']
split_lst = ['training', 'validation', 'testing']
if not os.path.isdir(dataset_root):
    os.mkdir(dataset_root)

missing_vid_lst = []

# download videos for training/validation/testing splits
for vid_file, split in zip(vid_file_lst, split_lst):
    if not os.path.isdir(os.path.join(dataset_root, split)):
        os.mkdir(os.path.join(dataset_root, split))
    with open(vid_file) as f:
        lines = f.readlines()
        for line in lines:
            rcp_type,vid_name = line.replace('\n','').split('/')

            # download the video
            vid_url = 'www.youtube.com/watch?v='+vid_name
            vid_prefix = os.path.join(dataset_root, split, rcp_type, vid_name) 
            #os.system(' '.join(("youtube-dl -o", vid_prefix, vid_url)))
            yt = YouTube(vid_url, on_progress_callback = on_progress)
            path = os.path.join(dataset_root, split, vid_name)
            if os.path.exists(path):
                continue
            try:
                os.mkdir(path)
                ys = yt.streams.get_highest_resolution()
                ys.download(path)
                caption = yt.captions.get_by_language_code('a.en')
                caption.save_captions(path + "/captions_videocooking.srt")
                print('[INFO] Downloaded {} video {}'.format(split, vid_name))
            except:
                missing_vid_lst.append('/'.join((split, line)))
                print('[INFO] Cannot download {} video {}'.format(split, vid_name))
  #write the missing videos to file
missing_vid = open('missing_videos.txt', 'w')
for line in missing_vid_lst:
    missing_vid.write(line)

# sanitize and remove the intermediate files
# os.system("find ../raw_videos -name '*.part*' -delete")
#os.system("find ../raw_videos -name '*.f*' -delete")
