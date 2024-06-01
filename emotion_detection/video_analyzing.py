import os
import cv2
from tqdm import tqdm
from PIL import Image
from .face_detection import detect_faces
from .recognition import recognize_face, get_face_encodings
from .config import VIDEO_FOLDER_PATH


def process_video(row, data, df):
    dialogue_id = row['Dialogue_ID']
    utterance_id = row['Utterance_ID']
    speaker = row['Speaker']

    video_file = f"dia{dialogue_id}_utt{utterance_id}.mp4"
    video_path = os.path.join(VIDEO_FOLDER_PATH, video_file)

    if not os.path.exists(video_path):
        print(f"Video file {video_file} does not exist.")
        return False  # Returning False for not recognized

    print(f"Processing Video {video_file}....")
    print("Speaker: " + speaker)

    vs = cv2.VideoCapture(video_path)
    face_recognized = False
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

    name = "Unknown"
    while vs.isOpened():
        ret, frame = vs.read()
        if not ret:
            break

        boxes, rgb = detect_faces(frame)
        if boxes is not None:
            for box in boxes:
                face_encodings = get_face_encodings(rgb, box)

                if face_encodings:
                    name = recognize_face(face_encodings[0], data)
                    if name == speaker:
                        face_recognized = True
                        break
        if face_recognized:
            break

    vs.release()

    if face_recognized:
        print("Face recognized as:", name)
    else:
        print("Face not recognized!")
    print("\n")
    return face_recognized
