import cv2
import os


def video_to_images(
    video_path,
    output_dir,
    start_time=0,
    end_time=None,
    width=1920,
    height=1080
):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if end_time is None:
        end_time = duration

    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    saved_idx = 0

    while frame_idx < end_frame:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.resize(frame, (width, height))

        filename = os.path.join(
            output_dir,
            f"frame_{saved_idx:06d}.jpg"
        )

        cv2.imwrite(filename, frame)

        saved_idx += 1
        frame_idx += 1

    cap.release()

    print(f"Saved {saved_idx} images to '{output_dir}'")


if __name__ == "__main__":
    video_to_images(
        video_path="1_HQ.mkv",
        output_dir="validation_frames_raw",
        start_time=315,   # seconds
        end_time=375      # seconds
    )
