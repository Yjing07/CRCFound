# CRCFound: A Colorectal Cancer CT Image Foundation Model Based on Self-Supervised Learning

This repository includes an official implementation of paper:

Some code is borrowed from [MAE](https://github.com/facebookresearch/mae), and [huggingface](https://huggingface.co/).


## 1 Environmental preparation and quick start
Environmental requirements
* Ubuntu 18.04.5 LTS
* Python 3.8.11
  
## 2 How to load the pre-trained model
Download the pre-trained weight first!

## 3 Fine-tuning 
### Data preparation
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
### Start fine-tuning
Download the pre-trained weight from Google Drive and specify pretrained_path in finetuning_1percent.sh.
Start training by running
```
python ./src/train_class.py
```  
