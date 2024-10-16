import cv2
import os
import csv
import pysrt

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
        
        if not success:
            break

        if frame_count % frame_interval == 0:
            # Save the frame as image
            img_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
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

            csv_data.append([int(frame_count/frame_interval), img_filename, transcript_text])

        frame_count += 1

    video.release()

    with open(csv_output, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['index', 'image_path', 'transcript_text'])
        writer.writerows(csv_data)

    print(f"Saved {saved_count} frames from {video_path} to {output_folder}")
    print(f"Transcript mapping saved to {csv_output}")

# Example usage:
video_path = "samplevideo.mp4"
output_folder = "output_images_folder"
srt_path = "samplevideo.srt"
csv_output = "image_transcript_mapping.csv"
frame_interval = 30  # Sample every 30 frames

video_to_images_with_transcript(video_path, output_folder, srt_path, csv_output, frame_interval)
