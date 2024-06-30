from transformers import AutoTokenizer, AutoModel
import json
import os 
import joblib
# os.environ["CUDA_VISBLE_DEVICES"]="0"
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse

def main(args):
    images_dir = args.images_dir
    tokenizer = AutoTokenizer.from_pretrained("./GLMcheckpoints", trust_remote_code=True)
    model = AutoModel.from_pretrained("./GLMcheckpoints", trust_remote_code=True, device='cuda:0')
    model = model.eval()
    p_captions = json.load(open(images_dir, 'r'))
    p_embeddings = {}
    for k in tqdm(list(p_captions.keys())):
        response, history = model.chat(tokenizer, "假设你是一个专业的影像科的医生,请帮我提取不重复的ct影像关键词，并且关键词不要出现数字,关键词尽量短且重要: " + p_captions[k]['image_findings'], history=[])
        p_embeddings[k]={}
        p_embeddings[k]['response']=response
        p_embeddings[k]['embeddings']=model.hd_embeddings.detach().cpu()

    joblib.dump(p_embeddings, './report_features.pkl')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='mae')
    opt = parser.parse_args()

    main(opt)