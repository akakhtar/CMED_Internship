import boto3
import time
import cv2
import os
import json
from PIL import Image, ImageDraw, ImageFont
import io
class FaceAnalyzer:
    def __init__(self, bucket, known_faces_folder,role_arn, sns_topic_arn, region='us-east-1'):
        self.bucket = bucket
        self.facesFolder = known_faces_folder
        self.roleArn = role_arn
        self.snsTopicArn = sns_topic_arn
        self.rek = boto3.client('rekognition', region_name = region)
        self.s3 = boto3.client('s3',region_name = region)
        self.startJobId = None

    def ExtractFrames(self, video_path, output_folder, interval=1):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("Error: FPS value is zero")
            return

        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % (interval*int(fps)) == 0:
                frame_name = os.path.join(output_folder,f"frame{count:04d}.jpg")
                cv2.imwrite(frame_name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                count += 1

                self.s3.upload_file(frame_name,self.bucket,frame_name)
                print(f"Uploaded {frame_name} to bucket {self.bucket}")
                temp_frame = frame_name
                # self.AnalyzeFrame(frame_name)

                matched_faces = self.compareFaces(temp_frame)
                for face in matched_faces:
                    print(f"Identified Faces: {face['Name']}")
                    if face['Name'] == 'chandler':
                        print("Face matched!")
                        self.AnalyzeFrame(frame_name)
                    else:
                        print("Face didn't matched!")


                self.s3.delete_object(Bucket=self.bucket, Key=frame_name)
                print(f"Deleted {frame_name} from bucket {self.bucket}")

                os.remove(frame_name)
                print(f"Deleted {frame_name} from local directory")

        cap.release()
        print(f"Extracted {count} frames from {video_path}")

    def AnalyzeFrame(self, frame_name):
        print("Frame analysis started...")
        try:
            response = self.rek.detect_faces(
                Image={'S3Object':{'Bucket': self.bucket, 'Name': frame_name}},
                Attributes=['ALL']
            )
            print(f"Analysis for {frame_name}:")
            for face_detail in response['FaceDetails']:
                print("Face Analysis:")
                # print(f"  BoundingBox: {face_detail['BoundingBox']}")
                # print(f"  Landmarks: {json.dumps(face_detail['Landmarks'], indent=4)}")
                print(f"  Pose: {face_detail['Pose']}")
                print(f"  Quality: {face_detail['Quality']}")
                print(f"  Confidence: {face_detail['Confidence']}")
                print(f"  Emotions: {json.dumps(face_detail['Emotions'], indent=4)}")
                print(f"  AgeRange: {face_detail.get('AgeRange', 'N/A')}")
                print(f"  Smile: {face_detail.get('Smile', 'N/A')}")
                print(f"  Eyeglasses: {face_detail.get('Eyeglasses', 'N/A')}")
                print(f"  Sunglasses: {face_detail.get('Sunglasses', 'N/A')}")
                print(f"  Gender: {face_detail.get('Gender', 'N/A')}")
                print(f"  Beard: {face_detail.get('Beard', 'N/A')}")
                print(f"  Mustache: {face_detail.get('Mustache', 'N/A')}")
                print(f"  EyesOpen: {face_detail.get('EyesOpen', 'N/A')}")
                print(f"  MouthOpen: {face_detail.get('MouthOpen', 'N/A')}")
                print()
        except boto3.exceptions.botocore.exceptions.ClientError as e:
            print(f"Error analyzing frame {frame_name}: {e}")

    def compareFaces(self,target_frame):
        print("Starting Comparing of faces....")
        matched_faces = []
        for obj in self.s3.list_objects_v2(Bucket=self.bucket, Prefix = self.facesFolder).get('Contents',[]):
            known_face_name = obj['Key']
            if known_face_name.endswith('.jpg'):
                try:
                    response = self.rek.compare_faces(
                        SourceImage = {'S3Object':{'Bucket':self.bucket, 'Name': known_face_name}},
                        TargetImage = {'S3Object':{'Bucket':self.bucket, 'Name': target_frame}},
                        SimilarityThreshold = 90
                    )
                    for face_match in response['FaceMatches']:
                        matched_faces.append({
                            'Name':os.path.basename(known_face_name).split('.')[0],
                            'BoundingBox': face_match['Face']['BoundingBox']
                        })
                except boto3.exceptions.botocore.exceptions.ClientError as e:
                    print(f"Error comparing faces with {known_face_name}: {e}")
        return matched_faces




bucket_name = 'rekognition-video-console-demo-iad-rckezmes9mgj-1719886164'
local_video_path = 'dia0_utt0_local.mp4'
output_folder = 'output_frames'
known_faces_folder = 'Known_face/'
sns_topic_arn = 'arn:aws:sns:us-east-1:058264446520:detection'
role_arn = 'arn:aws:iam::058264446520:role/RekognitionServiceRole'
region = 'us-east-1'

analyzer = FaceAnalyzer(bucket_name,known_faces_folder,role_arn,sns_topic_arn, region)
analyzer.ExtractFrames(local_video_path, output_folder)