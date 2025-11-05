import cv2, os, argparse

def extract_frames(video_path, out_dir, fps=2):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(video_fps / fps))
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            cv2.imwrite(os.path.join(out_dir, f"frame_{saved:05d}.jpg"), frame)
            saved += 1
        count += 1
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()
    extract_frames(args.video, args.out, args.fps)
