import time
import cv2
import os
import collections
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import boto3


class FaceAnalyzer:
    def __init__(self, known_faces_folder, region='us-east-1'):
        self.facesFolder = known_faces_folder
        self.rek = boto3.client('rekognition', region_name=region)
        self.reset_state()

    def reset_state(self):
        self.landmark_sums = collections.defaultdict(lambda: {'X': 0, 'Y': 0, 'count': 0})

    def ExtractFrames(self, video_path, interval=0.25):
        self.reset_state()
        # print("Extraction of frames started...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {total_frames}")
        if fps == 0:
            print("Error: FPS value is zero")
            return

        count = 0
        matched_count = 0

        if total_frames < 24:
            interval = 0.10
        analyze_every_frame = total_frames < 10
        frame_interval = 1 if analyze_every_frame else int(interval * int(fps))


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_number % frame_interval == 0:
                frame_name = f"frame{count:04d}.jpg"
                cv2.imwrite(frame_name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                print(f"Frame number: {count}")
                count += 1

                matched_faces = self.compareFaces(frame_name, 'joey')

                if matched_faces:
                    # print(f"Face matched: {matched_faces['Name']}")
                    matched_count += 1
                    self.AnalyzeFrame(frame_name, matched_faces['BoundingBox'])
                else:
                    print("No faces matched in this frame!")

                os.remove(frame_name)

        cap.release()
        # print(f"Total extracted and matched frames from {video_path}: {matched_count}")
        landmarks = self.get_average_landmarks()
        return landmarks
    def AnalyzeFrame(self, frame_name, boundingBox):
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
                    'BoundingBox': face_match['Face']['BoundingBox'],
                    'Similarity': face_match['Similarity']
                }
                # print(f"Match found: {matched_faces['Name']} with similarity {matched_faces['Similarity']}%")
                break
        except boto3.exceptions.botocore.exceptions.ClientError as e:
            print(f"Error comparing faces with {known_face_name} possible with no faces found: {e}")
        return matched_faces

    def get_average_landmarks(self):
        average_landmarks = {}
        for type, sums in self.landmark_sums.items():
            average_landmarks[type] = {
                'X': sums['X'] / sums['count'],
                'Y': sums['Y'] / sums['count']
            }
        return average_landmarks


def compute_average_reduced_features(video_folder, csv_file, known_faces_folder, region='us-east-1'):
    df = pd.read_csv(csv_file)
    analyzer = FaceAnalyzer(known_faces_folder, region)
    landmark_features = {}  # Dictionary to store average landmarks

    for index, row in df.iterrows():
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']

        video_file = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        video_path = os.path.join(video_folder, video_file)
        print(f"Processing video : {video_file}")
        if not os.path.exists(video_path):
            print(f"Video file {video_file} not found.")
            continue

        try:
            # print(len(row['landmarks']))
            # if row['landmarks'] == '{}' or pd.isnull(row['landmarks']):
             average_landmarks = analyzer.ExtractFrames(video_path)
             landmark_features[index] = average_landmarks  # Store average landmarks
             print(f"Average Landmarks before addition to df : {landmark_features[index]} \n")
            # else:
            #     landmark_features[index] = row['landmarks']
            #     print(f"Facial landmarks not updated: {landmark_features[index]} \n")
        except Exception as e:
            print(f"Error processing video {video_file}: {e}")

    df['landmarks'] = pd.Series(landmark_features)  # Add average landmarks to DataFrame
    print(f"landmarks in data frame : {df['landmarks']}")

    # Save updated DataFrame back to CSV file
    df.to_csv(csv_file, index=False)
    print("CSV file updated")


known_faces_folder = '../Known_face/'
region = 'us-east-1'
video_folder = '../../Dataset/Train'
csv_file = '../Speakers/Train/Joey_train.csv'

start_time = time.time()
compute_average_reduced_features(video_folder, csv_file, known_faces_folder, region)
end_time = time.time()
print(f"Total time taken {(end_time - start_time) / 60} minutes.")