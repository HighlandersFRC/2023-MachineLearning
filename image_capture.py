import cv2

def convert(self, video, start_num):
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = start_num
    frame = 0
    while success:
        if frame % 5 == 0:
            cv2.imwrite("./new_images/" + f"{count}" + ".jpg", image)  
            print(f"Saved {count}.jpg")
            count += 1
        success,image = vidcap.read()
        frame += 1

if __name__ == "__main__":
    convert("./842023428.avi", 365)
