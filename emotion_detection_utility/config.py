import torch

ENCODING_PATH = "D:/My stuffs/0Study material/SEMESTER/CMED_Internship/encodings.pickle"
VIDEO_FOLDER_PATH = "/self_reporting_model/test_dia64/"
EXCEl_PATH = "self_reporting_model/test_dia64.xlsx"

DETECTION_METHOD = "hog"
DISPLAY = 1

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
