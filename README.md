# 2023-MachineLearning

This repository is for the setup and training of FRC-related machine learning models.

# Guide

## Limelight Setup

In order to access the Limelight via ssh, a firmware image with a modified password hash needs to be created.

Download and unzip the desired [Limelight firmware image](https://limelightvision.io/pages/downloads) and navigate to the directory containing the *.img* file in WSL or a Linux shell.

Or use an already made [root access Limelight image](https://drive.google.com/drive/folders/1wUgtF0c072AwEy9fcuHq4qYtK_6V73uf?usp=sharing).

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
ffmpeg -f mjpeg -r 5 -i "http://limelight.local:5802/stream.mjpeg?fps=10" -r 10 -q:v 1 ./capture.avi
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

After creating the different object classes and labeling the data, under "Actions" select "Export Labels" and select to export in the "VOC XML" format. Unzip the resulting file and copy the XML files and corresponding images into the same folder. All images must have an XML label, so discard any images without one.

### Tips for Image Labeling

Only label an object in frame when it is reasonably distinguishable. Labeling objects that are partially obstructed but still recognizable is fine, and can actually really help the final network detect partially obstructed objects. However, do not label objects that are obstructed to the point where they could reasonably be confused for something else.

Make sure that each edge of every label rectangle is close to edge of the object. This helps improve the consistency of training data and results in better detection models.

## Model Creation

To train a model on Google Colab, go to the [Object Detection Training Colab](https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb#scrollTo=dYVVlv5QUUZF) and follow the instructions there. After completing the colab, unzip the resulting file and find the model named *"edgetpu.tflite"*. This is the model file that you will upload to the Limelight, along with *"labelmap.txt"*.

To train a model on NVIDIA Jetson device (been tested on Jetson AGX Orin only), follow the below instructions.

Flash Jetson device with [JetPack](https://developer.nvidia.com/embedded/jetpack) 6.0DP. After finding the Jetson devices's IP address, connect via SSH (VSCode can connect via SSH too, which is very useful for moving and editing files on the Jetson device).

Verify that python and pip are both using the same version, 3.10.2.
```
python --version
pip --version
```

Install python virtual environment.
```
sudo apt-get install python3-venv
```

Create the workspace directory and activate the virtual environment.
```
sudo mkdir /content
sudo chmod 777 /content
python -m venv /content/object-detection-env
cd /content
source /object-detection-env/bin/activate
```

Clone the Tensorflow models repo.
```
git clone --depth 1 https://github.com/tensorflow/models
```

Install protobuf-compiler of right version.
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install protobuf-compiler<3.20.*
pip uninstall protobuf
pip install protobuf<3.20.*
protoc --version
pip freeze | grep protobuf
```

Use protobuf to compile .protos.
```
cd /content/models/research
protoc object_detection/protos/*.proto --python_out=.
```

Install major dependencies.
```
pip install pyyaml
cp /content/models/research/* /content/object-detection-env/lib/python3.10/site-packages/
cp /content/models/official /content/object-detection-env/lib/python3.10/site-packages/
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60dp tensorflow==2.14.0+nv23.11
pip install tensorflow_io
```

Install CUDA and cuDNN.

Add CUDA to path.
```
export PATH=${PATH}:/usr/local/cuda/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```

If there is a previous installation (check via `nvcc -V`), uninstall.
```
sudo apt-get --purge remove cuda
sudo apt-get autoremove
sudo rm -r /usr/local/cuda
sudo rm -r /usr/local/cuda-*
```

Install CUDA using the directions found [here](https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=20.04&target_type=deb_local).

Install cuDNN.
```
sudo apt-get remove libcudnn8
sudo apt-get remove libcudnn8-dev
sudo apt-get remove libcudnn8-samples
sudo apt-get install libcudnn8=8.9.0.84-1+cuda12.2
sudo apt-get install libcudnn8-dev=8.9.0.84-1+cuda12.2
sudo apt-get install libcudnn8-samples=8.9.0.84-1+cuda12.2
```

Run the model builder test. Use `pip install *` to install any modules not being found.

Verify that Tensorflow is configured correctly and recognizes the GPU.
```
python
import tensorflow as tf
print(f'Logical Devices: {tf.config.list_logical_devices()}')
print(f'Physical Devices: {tf.config.list_physical_devices()}')
print(f'CUDA Support: {tf.test.is_built_with_cuda()}')
print(f'GPU Support: {tf.test.is_built_with_gpu_support()}')
print(f'TF Version: {tf.version.VERSION}')
```
Should output something like this (ignore ):
```
Logical Devices: [LogicalDevice(name='/device:CPU:0', device_type='CPU'), LogicalDevice(name='/device:GPU:0', device_type='GPU')]
Physical Devices: [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), LogicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
CUDA Support: true
GPU Support: true
TF Version: 2.14.0
```

Upload `images.zip` via SSH to the Jetson device in /content.

Set up images.
```
cd /content
mkdir /content/images
unzip -q images.zip -d /content/images/all
mkdir /content/images/train; mkdir /content/images/validation; mkdir /content/images/test
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/train_val_test_split.py
python train_val_test_split.py
```

Create file label map.
```
cat <<EOF >> /content/labelmap.txt
${class_1}
${class_2}
EOF
```

Create .csv and .tfrecord files.
```
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/create_csv.py
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/create_tfrecord.py
python create_csv.py
python create_tfrecord.py --csv_input=images/train_labels.csv --labelmap=labelmap.txt --image_dir=images/train --output_path=train.tfrecord
python create_tfrecord.py --csv_input=images/validation_labels.csv --labelmap=labelmap.txt --image_dir=images/validation --output_path=val.tfrecord
```

Copy get_config_info.py to /content, configure chosen_model ('ssd-mobilenet-v2-fpnlite-320' for limelight use), and get download urls.
```
python get_config_info.py
```

Create mymodel directory and download tar and config files.
```
mkdir /content/models/mymodel/
cd /content/models/mymodel/
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/${pretrained_checkpoints}
wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/${base_pipeline_file}
tar -xvzf ${pretrained_checkpoint}
```

To edit the pipeline config file, run the code in the [Colab](https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb#scrollTo=dYVVlv5QUUZF) until "(Optional) If you're curious, you can display the configuration file's contents here in the browser by running the line of code below." Run the cell and copy the printed output to the /content/models/mymodels/${base_pipeline_file} and rename it to `pipeline_file.config`.

Set Tensorflow to use GPU for training by editing /content/models/research/object_detection/model_main_tf2.py and replacing lines 99-102 with `strategy = tf.distribute.OneDeviceStrategy(device = "/gpu:0")`.

Run the training (40000 steps is recommended. More steps increase training time and produce diminishing returns, less steps decrease training time but might not fine tune model as well. On the Jetson AGX Orin 32GB training ran at a speed of ~0.4 sec/step and took ~4.5 hours to complete). 
If running into OOM (Out Of Memory) errors, half the value of `batch_size` in /content/models/mymodels/pipeline_file.config until the error goes away.
```
python /content/models/research/object_detection/model_main_tf2.py --pipeline_config_path=/content/models/mymodel/pipeline_file.config --model_dir=/content/training/ --alsologtostderr --num_train_steps=40000 --sample_1_of_n_eval_examples=1
```

Create export directory and export model.
```
mkdir /content/custom_model_lite
python /content/models/research/object_detection/export_tflite_graph_tf2.py --trained_checkpoint_dir /content/training --output_directory /content/custom_model_lite --pipeline_config_path /content/models/mymodel/pipeline_file.config
```

Copy over lite_converter.py to /content and run to convert model to tflite.
```
python lite_converter.py
```

Copy over quantize.py to /content and run to quantize the tflite model.
```
python quantize.py
```

## EdgeTPU Compilation
(Necessary for Limelight w/ Google Coral)
Download /content/custom_model_lite/detect_quant.tflite to your local system as the EdgeTPU compiler is not available on Jetson devices.

Locally, either in native Linux or WSL, install edgetpu-compiler.
```
sudo apt-get install edgetpu-compiler
```

In the same directory as detect_quant.tflite, compile using the EdgeTPU compiler.
```
edgetpu_compiler detect_quant.tflite
```

Optionally remove the compiler log created.
```
rm detect_quant_edgetpu.log
```

`detect_quant_edgetpu.tflite` is ready to be deployed.

## Resources

### Guides and Services

[Limelight Access](https://www.chiefdelphi.com/t/roslight-ros-on-the-limelight-2/366263)  
[FFMPEG Recording](https://mjpg-streamer.inlab.net/manual/useful-commands/record-mjpg-stream-with-ffmpeg/)  
[Image Labeling](https://www.makesense.ai/)  
[Object Detection Model Training](https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb#scrollTo=dYVVlv5QUUZF)  

### Files

[Machine_Learning_Files](https://drive.google.com/drive/folders/1LbipsDv0HDSO0DMsFyCSFfEf_PhDZ5Yk?usp=sharing)
