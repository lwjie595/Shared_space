This is the code for ICLR2026 submission

Our gating-driven mechanism is applied into MME, CDAC and ECB, which named as MME-G, CDAC-G and ECB-G here.


########
In MME-G,  run "main_DA1.py" to train the model, and the "download_data.sh" can download the DomainNet dataset.
After download the dataset, please modify the path of dataset in "return_dataset.py"
The gate network is placed in “model/basenet.py”and ResNet34 is in “model/resnet.py”
The dependent environment is included in "requirements.txt"


########
In CDAC-G, run "main.py" to train the model. Also, it needs modify the path of dataset  in "return_dataset.py"
The gate network is placed in“model/basenet.py”and ResNet34 is in “model/resnet.py”
The dependent environment is included in "requirements.txt"


########
In ECB-G, run "train.py" to train the model. It needs modify the path of dataset  in "DomainNet/Gate/xxxx.yaml"
The gate network is placed in“model/basenet.py”and ResNet34 is in “model/resnet.py”
The dependent environment is included in "environment.yaml"


########
GPU：4090
cuda_version: release 12.1, V12.1.105
Pytorch: 2.3.0
