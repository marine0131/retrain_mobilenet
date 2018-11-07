used for retraining mobilenet, save to frozen graph, and transfer the frozen graph to tflite modele

#1. download pretrained mobilenet v1 model
this can downloaded from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md

we have downloaded mobilenet_v1_1.0_224 model and unzip to folder

#2. retrain mobilenet
see retrain_mobilenet.py
we eliminate last tensors in pretrained model, add reshape layer and a self defined fc layer, then only train the fc layer params.

frozen graph is saved

#3. turn to tflite
use the code in pbtolite.txt transfer the frozen graph to tflite model
