# CRCFound: A Colorectal Cancer CT Image Foundation Model Based on Self-Supervised Learning

Some code is borrowed from [MAE](https://github.com/facebookresearch/mae), and [huggingface](https://huggingface.co/).

## 1 Environmental preparation and quick start
Environmental requirements
* Ubuntu 18.04.5 LTS
* Python 3.8.11
  
First clone the repo and cd into the directory:
```
git clone https://github.com/Yjing07/CRCFound.git
cd CRCFound
```
Then create a conda env and install the dependencies:
```
conda create -n CRCFound python=3.8.11 -y
conda activate CRCFound
pip install --upgrade pip
pip install -e .
```
## 2 Preparing and loading the model
Download the [pre-trained weight](https://drive.google.com/file/d/1zT6FsCh7RubL_k0LMGQwjhRMV90sprch/view?usp=sharing) first!

First create the _checkpoints directory_ inside the root of the repo:
```
mkdir -p checkpoints/
```
## 3 Data preparation
Before inputting the model, CT images need to be preprocessed. 

> We set all image Spaceing to _(1.0, 1.0, 3.0)_, and all image sizes to _(32, 256, 256)_.

The data organization for the classification task is (take MSI classification as an example)：

```
{
"training":
   {"img_name1":
      {label:"mss", "image_findings":"直-乙交界处肠壁环周增厚，最厚约10mm，..."}
    "img_name2":
      {label:"msi", "image_findings":"降结肠中段肠壁不规则环周增厚，最厚处约1.4cm，..."}
  ...}
"validation":
{"img_name1":
      {"label":"xx", "image_findings":"..."}
 "img_name2":
      {"label":"xx", "image_findings":"..."}
  ...}
}
```
The data organization for the prognosis task is:
```
 {
"training":
   {"img_name1":
      {"os.event": 1.0,"os.delay": 1.51, "dfs.event": 1.0, "dfs.delay": 1.08, "image_findings":"直肠中段左侧壁增厚，累及约1/2肠周，..."}
    "img_name2":
      {"os.event": 0.0, "os.delay": 64.5, "dfs.event": 0.0, "dfs.delay": 64.5, "image_findings":"升结肠中段肠壁不规则增厚，呈肿块样突向腔内生长，..."}
  ...}
"validation":
  {"img_name1":
        {"os.event": xx ,"os.delay": xx, "dfs.event": xx, "dfs.delay":xx, "image_findings":"..."}
   "img_name2":
        {"os.event": xx,"os.delay": xx, "dfs.event": xx, "dfs.delay": xx, "image_findings":"..."}
    ...}
}
```
## 4 Fine-tuning in your datasets
Download the pre-trained weight from [Google Drive](https://drive.google.com/file/d/1zT6FsCh7RubL_k0LMGQwjhRMV90sprch/view?usp=sharing) and specify _weight directory_ during training..
### 4.1 Fine-tuning using CT images

Start training by running：
```
python ./src/train_class.py --img_idx your datas -- data_path 
```  
### 4.1 Fine-tuning using CT images and text reports
