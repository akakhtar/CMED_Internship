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

    def StartFaceDetectionJob(self):
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
