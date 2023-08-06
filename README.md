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

Machine learning models need lots of varied training data in order to perform reliably. When gathering data be sure to vary factors like lighting conditions, camera input settings, angles, distances, and positions in frame.

Record video from the Limelight stream and write it to an *.avi* file:
```
ffmpeg -f mjpeg -r 5 -i "http://limelight.local:5802/stream.mjpeg?fps=10" -r 10 ./capture.avi
```



## Resources
[Limelight Access](https://www.chiefdelphi.com/t/roslight-ros-on-the-limelight-2/366263)  
[FFMPEG Recording](https://mjpg-streamer.inlab.net/manual/useful-commands/record-mjpg-stream-with-ffmpeg/)  
[Image Labeling](https://www.makesense.ai/)  
