import cv2
import argparse

parser = argparse.ArgumentParser(description = "Capture jpeg images from .avi video")
parser.add_argument("-n", "--number", help = "Number to start at for file naming", required = True)
parser.add_argument("-s", "--step", help = "Step between captured frames", required = False, default = 5)
parser.add_argument("file", help = ".avi file to capture frames from")
parser.add_argument("directory", help = "directory to write jpeg frames to")
args = vars(parser.parse_args())

def convert():
    vidcap = cv2.VideoCapture(args["file"])
    success, image = vidcap.read()
    count = int(args["number"])
    frame = 0
    step = int(args["step"])
    directory = args["directory"]
    while success:
        if frame % step == 0:
            cv2.imwrite(f"./{directory}/{count}.jpg", image)  
            print(f"Saved {count}.jpg")
            count += 1
        success, image = vidcap.read()
        frame += 1

if __name__ == "__main__":
    convert()
