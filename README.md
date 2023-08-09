# 2023-MachineLearning

This repository is for the setup and training of FRC-related machine learning models.

# Guide

## Limelight Setup

In order to access the Limelight via ssh, a firmware image with a modified password hash needs to be created.

Download and unzip the desired [Limelight firmware image](https://limelightvision.io/pages/downloads) and navigate to the directory containing the *.img* file in WSL or a Linux shell.

Install and use binwalk to locate the main filesystem inside the image file:
```
sudo apt-get install binwalk
binwalk -B --include="linux ext filesystem" limelight_image.img
```

The command should come up with several hits with the volume name "rootfs" at different memory addresses. Copy the decimal memory address of the first hit.

Mount the located filesystem and locate the */etc/shadow* file that contains the password hashes:
```
mkdir temp
sudo mount -o loop,offset=$(address) limelight_image.img temp/
cd temp/etc/
```

Generate a hash of a known password:
```
openssl passwd -5 -salt xyz new_password
```

Open the */etc/shadow* file and replace the password hashes next to *pi* and *root* with the new hash.

Unmount the filesystem:
```
cd temp/
cd ..
sudo umount temp/
```

Re-zip the *.img* file and flash the Limelight like normal. Upon boot, the Limelight should be accessible via ssh with the known password.

## Data Collection and Labeling

Machine learning models need lots of varied training data in order to perform reliably. When gathering data be sure to vary factors like lighting conditions, camera input settings, backgrounds, angles, distances, and positions in frame.

Record video from the Limelight stream and write it to an *.avi* file:
```
ffmpeg -f mjpeg -r 5 -i "http://limelight.local:5802/stream.mjpeg?fps=10" -r 10 ./capture.avi
```

Capture frames from the *.avi* file to use as training data. Default step size of 5 frames. Use multiple times with offset number arguments (-n) to avoid over-writing other files if capturing from multiple *.avi* files.
```
python image_capture.py -n 0 capture1.avi images/
```
*Creates 0.jpg-92.jpg*
```
python image_capture.py -n 93 capture1
```
*Creates new files starting at 93.jpg*

[MakeSense](https://www.makesense.ai/) is recommended for image labeling as it is efficient, free, and exports to the desired format (VOC XML). After uploading all the training images select the desired model type (further instructions here will cover training an Object Detection model).

After creating the different object classes and labeling the data, under Actions select export labels and select to export in the VOC XML format. Unzip the resulting file and copy the XML files and corresponding images into the same folder. All images must have an XML label, so discard any images without one.

To train a model on the labeled data, go to the [Object Detection Training Colab](https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb#scrollTo=dYVVlv5QUUZF) and follow the instructions there.

After completing the colab, unzip the resulting file and find the model named *"edgetpu.tflite"*. This is the model file that you will upload to the Limelight, along with *"labelmap.txt"*.

## Resources
[Limelight Access](https://www.chiefdelphi.com/t/roslight-ros-on-the-limelight-2/366263)  
[FFMPEG Recording](https://mjpg-streamer.inlab.net/manual/useful-commands/record-mjpg-stream-with-ffmpeg/)  
[Image Labeling](https://www.makesense.ai/)  
[Object Detection Model Training](https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb#scrollTo=dYVVlv5QUUZF)  
