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
        self.all_landmarks = []
        self.reduced_features = []

    def ExtractFrames(self, video_path, interval=0.30):
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
        result = self.feature_reduction(self.all_landmarks)
        return result, landmarks  # Return both reduced features and landmarks

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
                    landmarks = []
                    for landmark in face_detail['Landmarks']:
                        landmarks.append(landmark['X'])
                        landmarks.append(landmark['Y'])
                    self.all_landmarks.append(landmarks)

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

    def feature_reduction(self, features):
        print(f"Features before reduction : {features}")
        if not features:
            print("No features to reduce.")
            return

        # Flatten the list of lists
        flattened_features = [item for sublist in features for item in sublist]

        # Convert to numpy array and reshape to the required shape for PCA
        flattened_features = np.array(flattened_features).reshape(len(features), -1)

        # Standardize the data
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(flattened_features)

        # Apply Kernel PCA
        kpca = KernelPCA(kernel='rbf', eigen_solver='arpack', n_components=1)
        reduced_features = kpca.fit_transform(standardized_features)
        print(f"Features after reduction: {reduced_features}")
        reduced_features = np.mean(reduced_features)
        print(f"Average reduced features: {reduced_features}")
        return reduced_features


def compute_average_reduced_features(video_folder, csv_file, known_faces_folder, region='us-east-1'):
    df = pd.read_csv(csv_file)
    analyzer = FaceAnalyzer(known_faces_folder, region)
    video_features = {}
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
            average_reduced_feature, average_landmarks = analyzer.ExtractFrames(video_path)
            video_features[index] = average_reduced_feature
            landmark_features[index] = average_landmarks  # Store average landmarks
            print(f"Features before addition to df : {video_features[index]} \n")
        except Exception as e:
            print(f"Error processing video {video_file}: {e}")

    df['facial_feature'] = pd.Series(video_features)
    df['landmarks'] = pd.Series(landmark_features)  # Add average landmarks to DataFrame
    print(f"features in data frame : {df['facial_feature']}")
    print(f"landmarks in data frame : {df['landmarks']}")

    # Save updated DataFrame back to CSV file
    df.to_csv(csv_file, index=False)
    print("CSV file updated")


known_faces_folder = 'Known_face/'
region = 'us-east-1'
video_folder = '../Dataset/Train'
csv_file = 'Ross_train.csv'

start_time = time.time()
compute_average_reduced_features(video_folder, csv_file, known_faces_folder, region)
end_time = time.time()
print(f"Total time taken {(end_time - start_time) / 60} seconds.")
