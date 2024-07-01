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
        return False, {}, 0  # Returning False for not recognized and an empty dictionary

    vs = cv2.VideoCapture(video_path)
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    face_recognized = False
    face_detected = False
    face_matched = False

    # Dictionary to store the count of frames each name is detected
    name_frame_count = {}

    while vs.isOpened():
        ret, frame = vs.read()
        if not ret:
            break

        boxes, rgb = detect_faces(frame)
        if boxes is not None:
            face_detected = True
            for box in boxes:
                face_encodings = get_face_encodings(rgb, box)
                name = "Unknown"
                if face_encodings:
                    name = recognize_face(face_encodings[0], data)
                    if name != "Unknown":
                        face_recognized = True
                    if name not in name_frame_count:
                        name_frame_count[name] = 0
                    name_frame_count[name] += 1

                    if name == speaker:
                        face_matched = True
        else:
            face_detected = False

    vs.release()

    if not face_detected:
        print("Face not detected")
    elif not face_recognized:
        print("Face detected but not recognized")
    elif not face_matched:
        print(f"Face recognized as {name} but did not match the speaker {speaker}")
    else:
        print(f"Face matched with the speaker: {speaker}")

    return face_matched, name_frame_count, total_frames
