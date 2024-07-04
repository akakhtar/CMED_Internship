import boto3
import json

class ImageFaceAnalyzer:
    def __init__(self, bucket, image, role_arn):
        self.bucket = bucket
        self.image = image
        self.roleArn = role_arn
        self.rek = boto3.client('rekognition', region_name='us-east-1')  # Specify the region

    def analyze_face(self,frame_name):
        response = self.rek.detect_faces(
            Image={'S3Object': {'Bucket': self.bucket, 'Name': frame_name}},
            Attributes=['ALL']
        )

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

# Replace with your actual values
bucket_name = 'face4409'  # Your S3 bucket name
image_file_name = 'faceImage/IMG-20210503-WA0044.jpg'  # Path to your image in the S3 bucket
frame_name = image_file_name
role_arn = 'arn:aws:iam::058264446520:role/RekognitionServiceRole'  # Your IAM role ARN

analyzer = ImageFaceAnalyzer(bucket_name, image_file_name, role_arn)
analyzer.analyze_face(frame_name)
