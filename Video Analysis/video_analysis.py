import boto3
import time
import cv2
import os
import json
from PIL import Image, ImageDraw, ImageFont
import io

class FaceAnalyzer:
    def __init__(self, bucket, video, role_arn, sns_topic_arn, region='us-east-1'):
        self.bucket = bucket
        self.video = video
        self.roleArn = role_arn
        self.snsTopicArn = sns_topic_arn
        self.rek = boto3.client('rekognition', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.startJobId = None

    def StartFaceDetection(self):
        response = self.rek.start_face_detection(
            Video={'S3Object': {'Bucket': self.bucket, 'Name': self.video}},
            NotificationChannel={'RoleArn': self.roleArn, 'SNSTopicArn': self.snsTopicArn}
        )
        self.startJobId = response['JobId']
        print('Start Job Id: ' + self.startJobId)

    def CheckJobStatus(self):
        response = self.rek.get_face_detection(JobId=self.startJobId)
        status = response['JobStatus']
        return status

    def GetFaceDetectionResults(self):
        max_results = 10
        pagination_token = ''
        finished = False

        while not finished:
            response = self.rek.get_face_detection(
                JobId=self.startJobId,
                MaxResults=max_results,
                NextToken=pagination_token
            )

            if 'VideoMetadata' in response:
                print('Codec: ' + response['VideoMetadata']['Codec'])
                print('Duration: ' + str(response['VideoMetadata']['DurationMillis']))
                print('Format: ' + response['VideoMetadata']['Format'])
                print('Frame rate: ' + str(response['VideoMetadata']['FrameRate']))
                print()
            else:
                print('VideoMetadata not found in the response')

            for face_detection in response.get('Faces', []):
                print('Face: ' + str(face_detection['Face']))
                print('Confidence: ' + str(face_detection['Face']['Confidence']))
                print('Timestamp: ' + str(face_detection['Timestamp']))
                print()

            if 'NextToken' in response:
                pagination_token = response['NextToken']
            else:
                finished = True

    def DownloadVideoFromS3(self, local_path):
        try:
            self.s3.download_file(self.bucket, self.video, local_path)
            print(f"Downloaded {self.video} from bucket {self.bucket} to {local_path}")
        except Exception as e:
            print(f"Error downloading video: {e}")

    def ExtractFrames(self, video_path, output_folder, interval=1):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("Error: FPS value is zero, which may indicate an unsupported video format or a corrupted file.")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        print(f"Frame per second: {fps}")
        print(f"Frame Count: {frame_count}")

        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % (interval * int(fps)) == 0:
                frame_name = os.path.join(output_folder, f"frame{count:04d}.jpg")

                # Save frame as JPEG image
                cv2.imwrite(frame_name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                count += 1

                # Upload frame to S3
                self.s3.upload_file(frame_name, self.bucket, frame_name)
                print(f"Uploaded {frame_name} to bucket {self.bucket}")

                # Remove frame from local directory
                os.remove(frame_name)
                print(f"Deleted {frame_name} from local directory")

                # Analyze emotions in the frame
                self.AnalyzeFrameForEmotions(frame_name)

                # Remove frame from S3
                self.s3.delete_object(Bucket=self.bucket, Key=frame_name)
                print(f"Deleted {frame_name} from bucket {self.bucket}")

        cap.release()
        print(f"Extracted {count} frames from {video_path}")

    def AnalyzeFrameForEmotions(self, frame_name):
        try:
            response = self.rek.detect_faces(
                Image={'S3Object': {'Bucket': self.bucket, 'Name': frame_name}},
                Attributes=['ALL']
            )
            print(f"Analysis for {frame_name}:")
            for face_detail in response['FaceDetails']:
                print("Face Analysis:")
                print(f"  BoundingBox: {face_detail['BoundingBox']}")
                print(f"  Landmarks: {json.dumps(face_detail['Landmarks'], indent=4)}")
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
            print()
        except boto3.exceptions.botocore.exceptions.ClientError as e:
            print(f"Error analyzing frame {frame_name}: {e}")


    def StartFaceDetectionJob(self):
        # self.DownloadVideoFromS3('local_video.mp4')
        # self.ExtractFrames('local_video.mp4', 'output_frames')
        # self.UploadFramesToS3Folder('output_frames')  # Corrected method name
        self.StartFaceDetection()
        while True:
            status = self.CheckJobStatus()
            print('Job status: ' + status)
            if status in ['SUCCEEDED', 'FAILED']:
                break
            time.sleep(10)  # Wait before checking again

        if status == 'SUCCEEDED':
            self.GetFaceDetectionResults()
        else:
            print('Face detection job failed')

# Replace with your actual values
bucket_name = 'rekognition-video-console-demo-iad-rckezmes9mgj-1719886164'
video_file_name = 'Train/dia1011_utt10.mp4'
sns_topic_arn = 'arn:aws:sns:us-east-1:058264446520:detection'
role_arn = 'arn:aws:iam::058264446520:role/RekognitionServiceRole'
region = 'us-east-1'

analyzer = FaceAnalyzer(bucket_name, video_file_name, role_arn, sns_topic_arn, region)
analyzer.StartFaceDetectionJob()
