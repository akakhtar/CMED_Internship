import os
import cv2
from tqdm import tqdm
from collections import Counter
from PIL import Image
from .face_detection import detect_faces
from .emotion_detection import detect_emotions
from .recognition import recognize_face, get_face_encodings
from .config import VIDEO_FOLDER_PATH


def process_video(row, data, df):
    dialogue_id = row['Dialogue_ID']
    utterance_id = row['Utterance_ID']
    speaker = row['Speaker']
    print(f"{row['Speaker']} : {row['Utterance']}")

    video_file = f"dia{dialogue_id}_utt{utterance_id}.mp4"
    video_path = os.path.join(VIDEO_FOLDER_PATH, video_file)

    if not os.path.exists(video_path):
        print(f"Video file {video_file} does not exist.")
        return None, None, None, {}  # Returning an empty dictionary for the hash map

    vs = cv2.VideoCapture(video_path)
    all_emotions_freq = []
    all_emotions_weightedFreq = []

    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
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
                    # print("Face recognized as:", name)
                    # Update the hash map with the count of frames each name is detected
                    if name != "Unknown":
                        face_recognized = True
                    if name not in name_frame_count:
                        name_frame_count[name] = 0
                    name_frame_count[name] += 1

                if name == speaker:
                    face_matched = True
                    # print("Speaker matched with the recognized face.")
                    pil_image = Image.fromarray(rgb)
                    face, class_probabilities = detect_emotions(pil_image, box)
                    if face is not None:
                        detected_emotion = max(class_probabilities, key=class_probabilities.get)
                        if class_probabilities[detected_emotion] >= 0.80:
                            weight = (frame_idx / total_frames)
                            all_emotions_weightedFreq.append((detected_emotion, weight))
                            all_emotions_freq.append(detected_emotion)

        frame_idx += 1

    vs.release()

    emotion_counts_weightedFreq = Counter()
    for emotion, weight in all_emotions_weightedFreq:
        emotion_counts_weightedFreq[emotion] += weight

    emotion_counts_freq = Counter(all_emotions_freq)

    if not face_detected:
        print("Face not detected")
    elif not face_recognized:
        print("Face detected but not recognized")
    elif not face_matched:
        print(f"Face recognized as {name} but did not match the speaker {speaker}")
    else:
        print(f"Face matched with the speaker: {speaker}")
        print(f"Emotions: {emotion_counts_freq.most_common()}, of total frames : {total_frames}.")

    overall_emotion = max(emotion_counts_weightedFreq,
                          key=emotion_counts_weightedFreq.get) if all_emotions_weightedFreq else "neutral"
    overall_emotion_freq = max(emotion_counts_freq, key=emotion_counts_freq.get) if all_emotions_freq else "neutral"

    return overall_emotion_freq, overall_emotion
