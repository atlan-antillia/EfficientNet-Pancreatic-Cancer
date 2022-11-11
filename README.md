<h2>EfficientNet-Pancreatic-Cancer: 2022/09/27)</h2>
<a href="#1">1 EfficientNetV2 Pancreatic Cancer Classification </a><br>
<a href="#1.1">1.1 Clone repository</a><br>
<a href="#1.2">1.2 Prepare Pancreatic Cancer dataset</a><br>
<a href="#1.3">1.3 Install Python packages</a><br>
<a href="#2">2 Python classes for Pancreatic Cancer Classification</a><br>
<a href="#3">3 Pretrained model</a><br>
<a href="#4">4 Train</a><br>
<a href="#4.1">4.1 Train script</a><br>
<a href="#4.2">4.2 Training result</a><br>
<a href="#5">5 Inference</a><br>
<a href="#5.1">5.1 Inference script</a><br>
<a href="#5.2">5.2 Sample test images</a><br>
<a href="#5.3">5.3 Inference result</a><br>
<a href="#6">6 Evaluation</a><br>
<a href="#6.1">6.1 Evaluation script</a><br>
<a href="#6.2">6.2 Evaluation result</a><br>

<h2>
<a id="1">1 EfficientNetV2 Pancreatic Cancer Classification</a>
</h2>

 This is an experimental EfficientNetV2 Pancreatic Cancer Classification project based on <b>efficientnetv2</b> in <a href="https://github.com/google/automl">Brain AutoML</a>.
<br>
<br>
The original Dataset PCGIPI-sliced has been taken from the following website:<br>
<a href="https://osf.io/wc4u9/files/osfstorage">Dataset for Pancreatic Cancer Grading in Pathological Images using Deep Learning Convolutional Neural Networks
</a>

<br>We use python 3.8 and tensorflow 2.8.0 environment on Windows 11.<br>
<br>

<h3>
<a id="1.1">1.1 Clone repository</a>
</h3>
 Please run the following command in your working directory:<br>
<pre>
git clone https://github.com/atlan-antillia/Pancreatic-Cancer.git
</pre>
You will have the following directory tree:<br>
<pre>
.
├─asset
└─projects
    └─Pancreatic-Cancer
        ├─eval
        ├─evaluation
        ├─inference
        ├─models
        ├─Pancreatic_Cancer_Images
        └─test
</pre>
<h3>
<a id="1.2">1.2 Prepare Pancreatic Cancer dataset</a>
</h3>
Please download the  Pancreatic Cancer dataset from the following website:<br>
<a href="https://osf.io/wc4u9/files/osfstorage">Dataset for Pancreatic Cancer Grading in Pathological Images using Deep Learning Convolutional Neural Networks
</a>

<br>

As a working master dataset, we have created <b>Dataset PCGIPI-sliced</b> ]
dataset from the original <b>Dataset PCGIPI-sliced</b> above.
<pre>
Dataset PCGIPI-sliced
  ├─Grade_I
  ├─Grade_II
  ├─Grade_III  
  └─Normal
</pre>

Futhermore, we have created a <b>Osteosarcoma_Images</b> from <b>Osteosarcoma-master</b>.
 by using <a href="./projects/Osteosarcoma/split_master.py">split_master.py</a>
<br>
<pre>
Pancreatic_Cancer_Images
├─test
│  ├─Grade_I
│  ├─Grade_II
│  ├─Grade_III
│  └─Normal
└─train
    ├─Grade_I
    ├─Grade_II
    ├─Grade_III    
    └─Normal
</pre>
The number of images in this Pancreatic_Cancer_Images is the following:<br>
<img src="./projects/Pancreatic-Cancer/_Pancreatic_Cancer_Images_.png" width="740" height="auto"><br>
<br>
<br>
Pancreatic_Cancer_Images /train/Grade_I:<br>
<img src="./asset/Pancreatic_Cancer_train_Grade-I.png" width="840" height="auto">
<br>
<br>
Pancreatic_Cancer_Images /train/Grade_II:<br>
<img src="./asset/Pancreatic_Cancer_train_Grade-II.png" width="840" height="auto">
<br>
<br>
Pancreatic_Cancer_Images /train/Grade_III:<br>
<img src="./asset/Pancreatic_Cancer_train_Grade-III.png" width="840" height="auto">
<br>
<br>
Pancreatic_Cancer_Images /train/Normal:<br>
<img src="./asset/Pancreatic_Cancer_train_Normal.png" width="840" height="auto">
<br>
<br>

<h3>
<a id="#1.3">1.3 Install Python packages</a>
</h3>
Please run the following commnad to install Python packages for this project.<br>
<pre>
pip install -r requirements.txt
</pre>
<br>

<h2>
<a id="2">2 Python classes for Pancreatic Cancer Classification</a>
</h2>
We have defined the following python classes to implement our Pancreatic Cancer Classification.<br>
<li>
<a href="./ClassificationReportWriter.py">ClassificationReportWriter</a>
</li>
<li>
<a href="./ConfusionMatrix.py">ConfusionMatrix</a>
</li>
<li>
<a href="./CustomDataset.py">CustomDataset</a>
</li>
<li>
<a href="./EpochChangeCallback.py">EpochChangeCallback</a>
</li>
<li>
<a href="./EfficientNetV2Evaluator.py">EfficientNetV2Evaluator</a>
</li>
<li>
<a href="./EfficientNetV2Inferencer.py">EfficientNetV2Inferencer</a>
</li>
<li>
<a href="./EfficientNetV2ModelTrainer.py">EfficientNetV2ModelTrainer</a>
</li>
<li>
<a href="./FineTuningModel.py">FineTuningModel</a>
</li>

<li>
<a href="./TestDataset.py">TestDataset</a>
</li>

<h2>
<a id="3">3 Pretrained model</a>
</h2>
 We have used pretrained <b>efficientnetv2-m</b> to train Pancreatic Cancer Classification FineTuning Model.
Please download the pretrained checkpoint file from <a href="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m.tgz">efficientnetv2-m.tgz</a>, expand it, and place the model under our top repository.

<pre>
.
├─asset
├─efficientnetv2-m
└─projects
    └─Pancreatic-Cancer
  ...
</pre>

<h2>
<a id="4">4 Train</a>

</h2>
<h3>
<a id="4.1">4.1 Train script</a>
</h3>
Please run the following bat file to train our Pancreatic Cancer Classification efficientnetv2 model by using
<b>Pancreatic_Cancer_Images/train</b>.
<pre>
./1_train.bat
</pre>
<pre>
rem 1_train.bat
python ../../EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --eval_dir=./eval ^
  --model_name=efficientnetv2-m  ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../efficientnetv2-m/model ^
  --optimizer=rmsprop ^
  --image_size=384 ^
  --eval_image_size=480 ^
  --data_dir=./Pancreatic_Cancer_Images/train ^
  --data_augmentation=True ^
  --valid_data_augmentation=True ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.0001 ^
  --trainable_layers_ratio=0.3 ^
  --dropout_rate=0.3 ^
  --num_epochs=50 ^
  --batch_size=4 ^
  --patience=10 ^
  --debug=True  
</pre>
, where data_generator.config is the following:<br>
<pre>
; data_generation.config

[training]
validation_split   = 0.2
featurewise_center = True
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 180
horizontal_flip    = True
vertical_flip      = True 
width_shift_range  = 0.2
height_shift_range = 0.2
shear_range        = 0.01
zoom_range         = [0.1, 2.0]
data_format        = "channels_last"

[validation]8
validation_split   = 0.2
featurewise_center = True
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 180
horizontal_flip    = True
vertical_flip      = True
width_shift_range  = 0.2
height_shift_range = 0.2
shear_range        = 0.01
zoom_range         = [0.1, 2.0]
data_format        = "channels_last"
</pre>

<h3>
<a id="4.2">4.2 Training result</a>
</h3>

This will generate a <b>best_model.h5</b> in the models folder specified by --model_dir parameter.<br>
Furthermore, it will generate a <a href="./projects/Pancreatic-Cancer/eval/train_accuracies.csv">train_accuracies</a>
and <a href="./projects/Pancreatic-Cancer/eval/train_losses.csv">train_losses</a> files
<br>
Training console output:<br>
<img src="./asset/Pancreatic_Cancer_train_console_at_epoch_21_0927.png" width="740" height="auto"><br>
<br>
Train_accuracies:<br>
<img src="./projects/Pancreatic-Cancer/eval/train_accuracies.png" width="640" height="auto"><br>

<br>
Train_losses:<br>
<img src="./projects/Pancreatic-Cancer/eval/train_losses.png" width="640" height="auto"><br>

<br>
<h2>
<a id="5">5 Inference</a>
</h2>
<h3>
<a id="5.1">5.1 Inference script</a>
</h3>
Please run the following bat file to infer the Pancreatic-Cancer in test images by the model generated by the above train command.<br>
<pre>
./2_inference.bat
</pre>
<pre>
rem 2_inference.bat
python ../../EfficientNetV2Inferencer.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.3 ^
  --dropout_rate=0.3 ^
  --image_path=./test/*.JPG ^
  --eval_image_size=480 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --infer_dir=./inference ^
  --debug=False 
</pre>
<br>
label_map.txt:
<pre>
Grade-I
Grade-II
Grade-III
Normal
</pre>
<br>
<h3>
<a id="5.2">5.2 Sample test images</a>
</h3>

Sample test images generated by <a href="./projects/Pancreatic-Cancer/create_test_dataset.py">create_test_dataset.py</a> 
from <a href="./projects/Pancreatic-Cancer/Pancreatic_Cancer_Images/test">Pancreatic_Cancer_Images/test</a>.
<br>
<img src="./asset/Pancreatic_Cancer_test.png" width="840" height="auto"><br>

<h3>
<a id="5.3">5.3 Inference result</a>
</h3>
This inference command will generate <a href="./projects/Osteosarcoma/inference/inference.csv">inference result file</a>.
<br>
<br>
Inference console output:<br>
<img src="./asset/Pancreatic_Cancer_infer_console_at_epoch_21_0927.png" width="740" height="auto"><br>
<br>

Inference result (inference.csv):<br>
<img src="./asset/Pancreatic_Cancer_inference_at_epoch_21_0927.png" width="740" height="auto"><br>
<br>
<h2>
<a id="6">6 Evaluation</a>
</h2>
<h3>
<a id="6.1">6.1 Evaluation script</a>
</h3>
Please run the following bat file to evaluate <a href="./projects/Osteosarcoma/Osteosarcoma_Images/test">
Osteosarcoma_Images/test</a> by the trained model.<br>
<pre>
./3_evaluate.bat
</pre>
<pre>
rem 3_evaluate.bat
python ../../EfficientNetV2Evaluator.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --data_dir=./Pancreatic_Cancer_Images/test ^
  --evaluation_dir=./evaluation ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.3 ^
  --dropout_rate=0.3 ^
  --eval_image_size=480 ^
  --mixed_precision=True ^
  --debug=False 
</pre>
<br>

<h3>
<a id="6.2">6.2 Evaluation result</a>
</h3>

This evaluation command will generate <a href="./projects/Pancreatic-Cancer/evaluation/classification_report.csv">a classification report</a>
 and <a href="./projects/Pancreatic-Cancer/evaluation/confusion_matrix.png">a confusion_matrix</a>.
<br>
<br>
Evaluation console output:<br>
<img src="./asset/Pancreatic_Cancer_evaluate_console_at_epoch_21_0927.png" width="740" height="auto"><br>
<br>

<br>
Classification report:<br>
<img src="./asset/Pancreatic_Cancer_classificaiton_report_at_epoch_21_0927.png" width="740" height="auto"><br>
<br>
Confusion matrix:<br>
<img src="./projects/Pancreatic-Cancer/evaluation/confusion_matrix.png" width="740" height="auto"><br>


<br>
<h3>
References
</h3>
<b>1. Dataset for Pancreatic Cancer Grading in Pathological Images using Deep Learning Convolutional Neural Networks</b><br>
<pre>
https://osf.io/wc4u9/files/osfstorage
</pre>

<b>2.PancreaSys: An Automated Cloud-Based Pancreatic Cancer Grading System</b><br>
Muhammad Nurmahir Mohamad Sehmi<br>
Mohammad Faizal Ahmad Fauzi<br>
Wan Siti Halimatul Munirah Wan Ahmad<br>
Elaine Wan Ling Chan<br>
<pre>
https://www.frontiersin.org/articles/10.3389/frsip.2022.833640/full
</pre>
<b>3. Leveraging Uncertainty in Deep Learning for Pancreatic Adenocarcinoma Grading</b><br>
Biraja Ghoshal,Bhargab Ghoshal and Allan Tucker<br>
<pre>
https://arxiv.org/pdf/2206.08787.pdf
</pre>

