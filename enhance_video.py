import cv2
import numpy as np

def enhance_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_w, target_h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    print(f"Enhancing video: {input_path} -> {output_path}")
    print(f"Original: {width}x{height} | Target: {target_w}x{target_h}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(resized, -1, kernel)
        final = cv2.addWeighted(resized, 0.6, sharpened, 0.4, 0)

        out.write(final)

    cap.release()
    out.release()
    print("Video enhancement complete.")

if __name__ == "__main__":
    enhance_video("assets/traffic4.mp4", "assets/traffic4_hd.mp4")
