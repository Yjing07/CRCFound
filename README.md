# CRCFound: A Colorectal Cancer CT Image Foundation Model Based on Self-Supervised Learning

Some code is borrowed from [MAE](https://github.com/facebookresearch/mae), [ChatGLM6B](https://github.com/THUDM/ChatGLM-6B?tab=readme-ov-file), [huggingface](https://huggingface.co/).

## 1 Environmental preparation and quick start
  First clone the repo and cd into the directory:
   ```
   git clone https://github.com/Yjing07/CRCFound.git
   cd CRCFound
   ```
  
  Then create a conda env and install the dependencies:
  ```
  conda create -n CRCFound python=3.8.11 -y
  conda activate CRCFound
  pip install -r requirements.txt
  ```
## 2 Preparing and loading the model
  Download the [pre-trained weight](https://drive.google.com/file/d/1zT6FsCh7RubL_k0LMGQwjhRMV90sprch/view?usp=sharing) first!

  First create the ```checkpoints directory``` inside the root of the repo:
  ```
  mkdir -r checkpoints
  ```
## 3 Data preparation
  Before inputting the model, CT images need to be preprocessed. 

> We set all image Spaceing to ```(1.0, 1.0, 3.0)```, and all image sizes to```(32, 256, 256)```.

  The labels organization for the classification task is (take MSI classification as an example)：
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
  The labels organization for the prognosis task is:
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
  Download the pre-trained weight from [Google Drive](https://drive.google.com/file/d/1zT6FsCh7RubL_k0LMGQwjhRMV90sprch/view?usp=sharing) and specify _weight directory_ during training.
  
  ### 4.1 Fine-tuning using CT images
  
  After data preprocessing, prepare the label file according to the above structure and set the label path to start training.
  
  ```
  python ./src/train_class.py --model_name mae --checkpoint checkpoints/checkpoint-999.pth --epochs 200 --batch_size 8 --lr 0.0001 --shape (32,256,256) -- data_path ${IMAGE_DIR}$ --label_path $LABELS DIR$ --log_path ./logs
  ```  
### 4.2 Fine-tuning using CT images and text reports

* Prepare text features
  
  download [ChatGLM6B](https://github.com/THUDM/ChatGLM-6B?tab=readme-ov-file) pretrained [weight](https://huggingface.co/THUDM/chatglm-6b) the placement path is "./GLMcheckpoints"
  
  ```
  python ./src/text_embedding.py --images_dir ${LABELS_DIR}$
  ```
* Start training
  ```
  python ./src/train_class_fusion.py --model_name mae --checkpoint checkpoints/checkpoint-999.pth --epochs 200 --batch_size 8 --lr 0.0001 --shape (32,256,256) -- data_path ${IMAGE_DIR}$ --label_path $LABELS DIR$ --log_path ./logs 
  ```

## Basic Usage: CRCFound as a Vision Encoder
 * 1.Load the CRCFound model
   ```
   import torch
   import os
   from lib.vit_pretrain import ViT

   model = ViT("CRCFound_large_patch16_256")
   pretrained_path = 'checkpoints/checkpoint-999.pth'  # TODO
   if os.path.isfile(pretrained_path):
     print(f"Loading pretrained weights from: {pretrained_path}")
     checkpoint = torch.load(pretrained_path, map_location='cpu')
     pretrained_weights = checkpoint.get('model', checkpoint) 
     cleaned_weights = {k.replace("module.", ""): v for k, v in pretrained_weights.items()}
     compatible_weights = {k: v for k, v in cleaned_weights.items() if k in model.state_dict()}
     model.load_state_dict(
          {**model.state_dict(), **compatible_weights}
      )
     print("Pretrained weights loaded successfully.")
   else:
     raise FileNotFoundError(f"Pretrained model not found at: {pretrained_path}")
    model.to(device="cuda", dtype=torch.float16)
    model.eval()
   ```
 * 2.Encode images with MUSK (refer to demo.ipynb for complete implementation)
   ```
  conda create -n CRCFound python=3.8.11 -y
  conda activate CRCFound
  pip install -r requirements.txt
  ```

