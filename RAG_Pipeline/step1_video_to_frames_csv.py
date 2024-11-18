import cv2
import os
import csv
import pysrt
import argparse

# Create the parser


def video_to_images_with_transcript(video_path, output_folder, srt_path, csv_output, frame_interval=1):
    """
    Extract video frames as images, and create csv in format of [index, image_path, transcript]

    video_path: Path to the input video file
    output_folder: Directory to save the image frames
    srt_path: Path to the SRT subtitle file
    csv_output: Path to the output CSV file mapping images to transcript text
    frame_interval: Interval of frames to save
    """
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    subs = pysrt.open(srt_path)
    
    fps = video.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    saved_count = 0
    csv_data = []

    while True:
        success, frame = video.read()
        #mkdir f"{output_folder}/frames" with os
        os.makedirs(f"{output_folder}/frames", exist_ok=True)
        
        if not success:
            break

        if frame_count % frame_interval == 0:
            # Save the frame as image
            img_filename = f"{output_folder}/frames/frame_{frame_count:04d}.jpg"
            print(f"Saving frame {frame_count} to {img_filename}")
            cv2.imwrite(img_filename, frame)
            saved_count += 1

            # Map subtitle with timestamp

            timestamp = frame_count / fps

            transcript_text = ""
            for sub in subs:
                start_time = sub.start.ordinal / 1000
                end_time = sub.end.ordinal / 1000
                if start_time <= timestamp <= end_time:
                    transcript_text = sub.text
                    break
            #make the img_filename the full path
            curr_path = os.path.join(os.getcwd(), img_filename)
            print(curr_path)

            if transcript_text == "":
                transcript_text = "No transcript available"
            csv_data.append([int(frame_count/frame_interval), curr_path, transcript_text,timestamp])

        frame_count += 1

    video.release()

    with open(f'{output_folder}/{csv_output}', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['index', 'image_path', 'transcript_text','timestamp'])
        writer.writerows(csv_data)

    print(f"Saved {saved_count} frames from {video_path} to {output_folder}")
    print(f"Transcript mapping saved to {csv_output}")

def main(args):

    video_path = args.v
    output_folder = args.o
    srt_path = args.s
    frame_interval = args.i

    # Example usage:
    # video_path = "/home/hice1/bpopper3/scratch/LLaVA_RAG_PoC/yt-download/crepe.mp4"
    # output_folder = "test_uniform"
    # srt_path = "/home/hice1/bpopper3/scratch/LLaVA_RAG_PoC/yt-download/captions_videocooking.srt"
    csv_output = "image_transcript_mapping.csv"
    # frame_interval = 100  # Sample every 30 frames
    video_to_images_with_transcript(video_path, output_folder, srt_path, csv_output, frame_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for step1")

    # Define expected arguments
    parser.add_argument('--v', type=str, help='path to the video file (mp4)')
    parser.add_argument('--o', type=str, help='path to output folder')
    parser.add_argument('--s', type=str, help='path to SRT files (transcript)')

    parser.add_argument('--i', type=int, help='frame interval')

    # Parse the arguments
    args = parser.parse_args()
    main(args)