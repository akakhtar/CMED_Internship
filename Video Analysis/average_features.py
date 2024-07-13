import time
import cv2
import os
import collections
import numpy as np
import pandas as pd
from scipy.spatial import distance
import boto3

class FaceAnalyzer:
    def __init__(self, known_faces_folder, region='us-east-1'):
        self.facesFolder = known_faces_folder
        self.rek = boto3.client('rekognition', region_name=region)
        self.reset_state()

    def reset_state(self):
        self.landmark_sums = collections.defaultdict(lambda: {'X': 0, 'Y': 0, 'count': 0})

    def calculate_distance(self, point1, point2):
        return distance.euclidean((point1['X'], point1['Y']), (point2['X'], point2['Y']))

    def ExtractFrames(self, video_path, interval=0.40):
        self.reset_state()
        print("Extraction of frames started...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("Error: FPS value is zero")
            return

        count = 0
        matched_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % int(interval * int(fps)) == 0:
                frame_name = f"frame{count:04d}.jpg"
                cv2.imwrite(frame_name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                print(f"Analysis of frame number : {count}")
                count += 1

                matched_faces = self.compareFaces(frame_name, 'ross')

                if matched_faces:
                    print(f"Face matched : {matched_faces['Name']}")
                    matched_count += 1
                    self.AnalyzeFrame(frame_name, matched_faces['BoundingBox'])
                else:
                    print("No faces matched in this frame!")

                os.remove(frame_name)

        cap.release()
        print(f"Total extracted and matched frames from {video_path} : {matched_count}")
        landmarks = self.get_average_landmarks()
        print("Average Landmarks:")
        print(landmarks)
        return self.average_distance(landmarks)

    def AnalyzeFrame(self, frame_name, boundingBox):
        print("Face analysis started.....")
        try:
            with open(frame_name, 'rb') as image_file:
                image_bytes = image_file.read()

            response = self.rek.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )
            for face_detail in response['FaceDetails']:
                if face_detail['BoundingBox'] == boundingBox:
                    for landmark in face_detail['Landmarks']:
                        type = landmark['Type']
                        self.landmark_sums[type]['X'] += landmark['X']
                        self.landmark_sums[type]['Y'] += landmark['Y']
                        self.landmark_sums[type]['count'] += 1

        except boto3.exceptions.botocore.exceptions.ClientError as e:
            print(f"Error analyzing frame {frame_name}: {e}")

    def compareFaces(self, target_frame, known_face_name):
        if not known_face_name.endswith('.jpg'):
            known_face_name += '.jpg'
        known_face_path = os.path.join(self.facesFolder, known_face_name)

        matched_faces = {}
        try:
            with open(known_face_path, 'rb') as source_image_file:
                source_image_bytes = source_image_file.read()

            with open(target_frame, 'rb') as target_image_file:
                target_image_bytes = target_image_file.read()

            response = self.rek.compare_faces(
                SourceImage={'Bytes': source_image_bytes},
                TargetImage={'Bytes': target_image_bytes},
                SimilarityThreshold=90
            )
            for face_match in response['FaceMatches']:
                matched_faces = {
                    'Name': os.path.basename(known_face_name).split('.')[0],
                    'BoundingBox': face_match['Face']['BoundingBox']
                }
                break
        except boto3.exceptions.botocore.exceptions.ClientError as e:
            print(f"Error comparing faces with {known_face_name}: {e}")
        return matched_faces

    def get_average_landmarks(self):
        average_landmarks = {}
        for type, sums in self.landmark_sums.items():
            average_landmarks[type] = {
                'X': sums['X'] / sums['count'],
                'Y': sums['Y'] / sums['count']
            }
        return average_landmarks

    def average_distance(self, landmarks):
        # Eyes
        eye_distances = [
            self.calculate_distance(landmarks['eyeLeft'], landmarks['eyeRight']),
            self.calculate_distance(landmarks['eyeLeft'], landmarks['leftPupil']),
            self.calculate_distance(landmarks['eyeRight'], landmarks['rightPupil']),
            self.calculate_distance(landmarks['leftEyeBrowLeft'], landmarks['leftEyeBrowRight']),
            self.calculate_distance(landmarks['rightEyeBrowLeft'], landmarks['rightEyeBrowRight']),
            self.calculate_distance(landmarks['leftEyeBrowLeft'], landmarks['leftEyeBrowUp']),
            self.calculate_distance(landmarks['leftEyeBrowRight'], landmarks['leftEyeBrowUp']),
            self.calculate_distance(landmarks['rightEyeBrowLeft'], landmarks['rightEyeBrowUp']),
            self.calculate_distance(landmarks['rightEyeBrowRight'], landmarks['rightEyeBrowUp']),
            self.calculate_distance(landmarks['leftEyeBrowUp'], landmarks['rightEyeBrowUp']),
        ]

        # Mouth
        mouth_distances = [
            self.calculate_distance(landmarks['mouthLeft'], landmarks['mouthRight']),
            self.calculate_distance(landmarks['nose'], landmarks['mouthLeft']),
            self.calculate_distance(landmarks['nose'], landmarks['mouthRight']),
        ]

        # Nose
        nose_distances = [
            self.calculate_distance(landmarks['nose'], landmarks['eyeLeft']),
            self.calculate_distance(landmarks['nose'], landmarks['eyeRight']),
            self.calculate_distance(landmarks['nose'], landmarks['mouthLeft']),
            self.calculate_distance(landmarks['nose'], landmarks['mouthRight']),
        ]

        # Jawline
        jawline_distances = [
            self.calculate_distance(landmarks['upperJawlineLeft'], landmarks['midJawlineLeft']),
            self.calculate_distance(landmarks['midJawlineLeft'], landmarks['chinBottom']),
            self.calculate_distance(landmarks['chinBottom'], landmarks['midJawlineRight']),
            self.calculate_distance(landmarks['midJawlineRight'], landmarks['upperJawlineRight']),
        ]

        # Average distances for each category
        average_eye_distance = np.mean(eye_distances)
        average_mouth_distance = np.mean(mouth_distances)
        average_nose_distance = np.mean(nose_distances)
        average_jawline_distance = np.mean(jawline_distances)

        # Combine all average distances into one feature vector
        feature_vector = np.array([
            average_eye_distance,
            average_mouth_distance,
            average_nose_distance,
            average_jawline_distance
        ])

        # Normalize by face size (e.g., distance between eye corners)
        reference_distance = self.calculate_distance(landmarks['eyeLeft'], landmarks['eyeRight'])
        normalized_feature_vector = feature_vector / reference_distance
        print("Normalized feature vector:", normalized_feature_vector)
        return average_eye_distance, average_mouth_distance, average_nose_distance, average_jawline_distance


def compute_average_reduced_features(video_folder, csv_file, known_faces_folder, region='us-east-1'):
    df = pd.read_csv(csv_file)
    analyzer = FaceAnalyzer(known_faces_folder, region)
    video_features = {}

    for index, row in df.iterrows():
     if pd.isna(row['average_eye_distance']):
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        video_file = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        video_path = os.path.join(video_folder, video_file)
        print(f"Processing video : {video_file}")
        if not os.path.exists(video_path):
            print(f"Video file {video_file} not found.")
            continue

        try:
            avg_eye_dist, avg_mouth_dist, avg_nose_dist, avg_jawline_dist = analyzer.ExtractFrames(video_path)
            df.at[index, 'average_eye_distance'] = avg_eye_dist
            df.at[index, 'average_mouth_distance'] = avg_mouth_dist
            df.at[index, 'average_nose_distance'] = avg_nose_dist
            df.at[index, 'average_jawline_distance'] = avg_jawline_dist
        except Exception as e:
            print(f"Error processing video {video_file}: {e}")

    print(f"Updated DataFrame:")
    print(df[['average_eye_distance', 'average_mouth_distance', 'average_nose_distance', 'average_jawline_distance']])

    # Save updated DataFrame back to CSV file
    df.to_csv(csv_file, index=False)
    print("CSV file updated")


known_faces_folder = 'Known_face/'
region = 'us-east-1'
video_folder = '../Dataset/Train'
csv_file = 'Ross_train.csv'
# local_video_path = 'dia0_utt0_local.mp4'
start_time = time.time()
# analyzer = FaceAnalyzer(known_faces_folder, region)
# analyzer.ExtractFrames(local_video_path)
compute_average_reduced_features(video_folder, csv_file, known_faces_folder, region)
end_time = time.time()
print(f"Total time taken {(end_time - start_time)/60} minutes.")
