# CRCFound: A Colorectal Cancer CT Image Foundation Model Based on Self-Supervised Learning

Some code is borrowed from [MAE](https://github.com/facebookresearch/mae), and [huggingface](https://huggingface.co/).


## 1 Environmental preparation and quick start
Environmental requirements
* Ubuntu 18.04.5 LTS
* Python 3.8.11
  
## 2 Preparing and loading the model
Download the [pre-trained weight](https://drive.google.com/file/d/1zT6FsCh7RubL_k0LMGQwjhRMV90sprch/view?usp=sharing) first!

First create the ==checkpoints directory== inside the root of the repo:
```
mkdir -p checkpoints/
```
## 3 Fine-tuning in your datasets
### 3.1 Data preparation
"img_idx":
```
{"train":
 {"img_name1",
  "img_name2,"
  ...
}}
```
"label_path":
```
 {
"img_name1":{"task1":label,"task2":label,},
"img_name2":{"task1":label."task1":label,},
...
}
```
### 3.2 Start fine-tuning
Download the pre-trained weight from Google Drive(https://drive.google.com/file/d/1zT6FsCh7RubL_k0LMGQwjhRMV90sprch/view?usp=sharing) and specify pretrained_path in finetuning_1percent.sh.
Start training by running
```
python ./src/train_class.py
```  
