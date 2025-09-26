#script for extracting random frames from a folder of videos
#useful for getting data for training

#input: folder of videos
#output: folder of frames


import cv2
import os
import numpy as np

def extractRandomFrames(video_folder, output_folder, num_frames):
    """
    Extract random frames from a folder of videos
    """
    for video in os.listdir(video_folder):
        if video.endswith('.mp4'):
            video_path = os.path.join(video_folder, video)
            extractRandomFramesFromVideo(video_path, output_folder, num_frames)

def extractRandomFramesFromVideo(video_path, output_folder, num_frames):
    """
    Extract random frames from a video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure we don't request more frames than available
    num_frames = min(num_frames, total_frames)
    
    # Generate random frame indices
    random_frames = np.random.randint(0, total_frames, num_frames)
    
    # Get video name without extension for output folder
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)
    
    # Create output folder for this video
    os.makedirs(video_output_folder, exist_ok=True)
    
    print(f"Extracting {num_frames} random frames from {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
    
    extracted_count = 0
    for frame_idx in random_frames:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        if ret:
            # Generate filename with frame number and timestamp
            timestamp = frame_idx / fps
            filename = f"frame_{frame_idx:06d}_t{timestamp:.2f}s.jpg"
            output_path = os.path.join(video_output_folder, filename)
            
            # Save frame
            cv2.imwrite(output_path, frame)
            extracted_count += 1
        else:
            print(f"Warning: Could not read frame {frame_idx}")
    
    cap.release()
    print(f"Successfully extracted {extracted_count} frames to {video_output_folder}")
    return extracted_count

def main():
    """
    Main function with command line interface
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract random frames from videos")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input folder containing videos")
    parser.add_argument("--output", "-o", required=True,
                       help="Output folder for extracted frames")
    parser.add_argument("--frames", "-f", type=int, default=10,
                       help="Number of random frames to extract per video (default: 10)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                       help="Random seed for reproducible results")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Validate input folder
    if not os.path.exists(args.input):
        print(f"Error: Input folder does not exist: {args.input}")
        return
    
    # Create output folder
    os.makedirs(args.output, exist_ok=True)
    
    # Get list of video files
    video_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"Error: No video files found in {args.input}")
        return
    
    print(f"Found {len(video_files)} video files")
    print(f"Extracting {args.frames} random frames from each video")
    print(f"Output folder: {args.output}")
    print("-" * 50)
    
    total_extracted = 0
    for video_file in video_files:
        try:
            video_path = os.path.join(args.input, video_file)
            extracted = extractRandomFramesFromVideo(video_path, args.output, args.frames)
            total_extracted += extracted
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            continue
    
    print("-" * 50)
    print(f"Extraction complete! Total frames extracted: {total_extracted}")

if __name__ == "__main__":
    main()