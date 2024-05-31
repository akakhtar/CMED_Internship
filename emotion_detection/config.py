import torch

ENCODING_PATH = "D:/My stuffs/0Study material/SEMESTER/CMED_Internship/encodings.pickle"
VIDEO_FOLDER_PATH = "D:/My stuffs/0Study material/SEMESTER/CMED_Internship/custom_dataset/test_dia4/"
EXCEl_PATH = "test_dia4.xlsx"

DETECTION_METHOD = "cnn"
DISPLAY = 1

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
