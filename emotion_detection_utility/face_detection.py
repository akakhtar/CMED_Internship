import cv2
import imutils
from facenet_pytorch import MTCNN
from .config import DEVICE

# original definition of mtcnn parameters
# mtcnn = MTCNN(
#     image_size=160,
#     margin=0,
#     min_face_size=200,
#     thresholds=[0.6, 0.7, 0.7],
#     factor=0.709,
#     post_process=True,
#     keep_all=False,
#     device=DEVICE
# )
mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=100,  # Reduced min face size
    thresholds=[0.5, 0.6, 0.6],  # Lower thresholds
    factor=0.709,
    post_process=True,
    keep_all=True,  # Keep all detected faces
    device=DEVICE
)

def detect_faces(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(rgb, width=750)
    boxes, probs = mtcnn.detect(rgb)
    # print(f"Face detected with probability of {probs}.")
    return boxes, rgb
