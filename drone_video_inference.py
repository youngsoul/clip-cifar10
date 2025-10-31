import argparse
import cv2
import joblib
from embetter.multi import ClipEncoder
from typing import List
from PIL import Image
import numpy as np

labels = ['truck', 'car', 'motorcycle']
label_descriptions = ['picture of a red truck with a black border with a brown box border', 'picture of a police car with a black border with a brown box border', 'picture of a motorcycle with a black border with a brown box border']

label_encodings = {}

def create_label_encodings(label_names: List = labels) -> dict:
    """Create a dictionary of target names and their encodings."""
    global label_encodings
    for i, label_name in enumerate(label_names):
        label_encodings[label_name] = ClipEncoder().fit_transform(label_descriptions[i])
    return label_encodings

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def check_for_label_similarity(label, image_encoding) -> float:
    target_embedding = label_encodings[label]
    similarity = cosine_similarity(image_encoding, target_embedding)

    return similarity[0]

    # best = np.argmax(sims)
    #
    # # threshold
    # if sims[best] < 0.25:
    #     print("none")
    # else:
    #     print(labels[best])

def process_video(video_path: str, show_window: bool = True, conf_threshold: float = 0.80):
    """
    Runs inference on a video stream frame-by-frame.

    - Encodes each frame as an image embedding using CLIP.
    - Predicts the class with the trained sklearn model (`model.joblib`).
    - Optionally overlays the prediction on the video.
    """
    videoCap = cv2.VideoCapture(video_path)

    if not videoCap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Load classifier once
    model = joblib.load("./model.joblib")
    if model is None:
        raise RuntimeError("Failed to load model")

    # Init encoder (load once, reuse!)
    encoder = ClipEncoder()

    try:
        while True:
            ret, frame_bgr = videoCap.read()
            if not ret:
                break

            # Convert BGR (OpenCV) -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Ensure contiguous uint8 array then to PIL.Image
            if frame_rgb.dtype != np.uint8:
                frame_rgb = frame_rgb.astype(np.uint8)
            pil_img = Image.fromarray(frame_rgb)

            # IMPORTANT: ClipEncoder/SentenceTransformer expects a list of PIL images for image embeddings
            # This returns a numpy array of shape (embedding_dim,) for a single item, or (n, d) for batch
            video_frame_encoding = encoder.transform([pil_img])
            # Ensure 2D shape (1, d) for sklearn
            if video_frame_encoding.ndim == 1:
                video_frame_encoding = video_frame_encoding.reshape(1, -1)

            # Predict label and confidence
            y_pred = model.predict(video_frame_encoding)[0]

            # y_pred == "frog" means the NONE class or the class not trained on
            if y_pred != "frog":
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(video_frame_encoding)[0]
                    # find probability of predicted class
                    try:
                        cls_index = list(model.classes_).index(y_pred)
                        conf = float(proba[cls_index])
                        if conf < conf_threshold:
                            y_pred = "unknown"
                            conf = 0.0
                        else:
                            if conf < 92.0:
                                # verify the label prediction is similar to the actual label
                                label_similarity = check_for_label_similarity(y_pred, video_frame_encoding)
                                if label_similarity <= 0.21:
                                    y_pred = "unknown"
                                    conf = label_similarity
                    except Exception:
                        conf = float(np.max(proba))
                else:
                    conf = 0.0
            else:
                y_pred = ""
            # Draw overlay
            if show_window:
                overlay = frame_bgr.copy()
                if y_pred != "":
                    text = f"{y_pred} ({conf:.2f})"
                else:
                    text = ""
                cv2.putText(
                    overlay,
                    text,
                    org=(20, 40),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                cv2.imshow("Raw Video Output", overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        videoCap.release()
        if show_window:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run CLIP-based video frame inference")
    parser.add_argument("--video", type=str, default="./videos/dexi_camera_all_classes_85_compression.mp4", help="Path to video file")
    parser.add_argument("--no-window", action="store_true", help="Disable preview window")
    args = parser.parse_args()

    create_label_encodings()
    process_video(args.video, show_window=not args.no_window)


if __name__ == '__main__':
    main()