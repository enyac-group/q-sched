import clip
import torch
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import argparse
import os

# Global variables for model caching
_device = 'cuda'
_model = None
_preprocessor = None
_nlp = None

def initialize_models():
    # Initialize CLIP model
    _model, _preprocessor = clip.load("CS-ViT-B/16", device=_device, jit=False)
    _model.eval()

    # Load model weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    state_dict = torch.load(os.path.join(script_dir, "embeding", "PQ-overall.pth"))
    state_dict_1 = torch.load(os.path.join(script_dir, "embeding", "PQ-technical.pth"))
    state_dict_2 = torch.load(os.path.join(script_dir, "embeding", "PQ-rational.pth"))
    state_dict_3 = torch.load(os.path.join(script_dir, "embeding", "PQ-natural.pth"))

    # Initialize embeddings
    pos_embed = _model.positional_embedding.type(_model.dtype)

    def load_embed(state_dict):
        ctx = state_dict['params']['prompt_learner.ctx']
        token_prefix = state_dict['params']['prompt_learner.token_prefix']
        token_suffix = state_dict['params']['prompt_learner.token_suffix']
        pos = torch.cat((state_dict['params']['prompt_learner.token_prefix'], 
                        state_dict['params']['prompt_learner.ctx'], 
                        state_dict['params']['prompt_learner.token_suffix']), 1).to(_device)
        name_len = 3
        half_n_ctx = 8
        prefix_i = token_prefix[:, :, :]
        class_i = token_suffix[:, :name_len, :]
        suffix_i = token_suffix[:, name_len:, :]
        ctx_i_half1 = ctx[:, :half_n_ctx, :]
        ctx_i_half2 = ctx[:, half_n_ctx:, :]
        text_embed = torch.cat(
            [
                prefix_i,     # (1, 1, dim)
                ctx_i_half1,  # (1, n_ctx//2, dim)
                class_i,      # (1, name_len, dim)
                ctx_i_half2,  # (1, n_ctx//2, dim)
                suffix_i,     # (1, *, dim)
            ],
            dim=1,
        ).to(_device).type(_model.dtype)
        return text_embed

    # Load text embeddings
    text_embed = load_embed(state_dict)
    text_embed_1 = load_embed(state_dict_1)
    text_embed_2 = load_embed(state_dict_2)
    text_embed_3 = load_embed(state_dict_3)

def encode_text(text_embed, pos):
    x = text_embed
    x = x + pos
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = _model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = _model.ln_final(x).type(_model.dtype)
    x = x[torch.arange(x.shape[0]), torch.tensor([20, 20])] @ _model.text_projection
    return x

def single_quality_map(image, text_embed, img_size):
    with torch.no_grad():
        image_features = _model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = encode_text(text_embed, pos_embed)
        features = image_features @ text_features.t()
        similarity_map = clip.get_similarity_map(features[:, 1:, :], img_size)
        feature_per = features[0][0].cpu().numpy()
        perceptual_score = np.exp(feature_per)[0]/np.sum(np.exp(feature_per))
        perceptual_map = similarity_map[0,:,:,0]
        return perceptual_map, perceptual_score

def perceptual_quality_map(pil_img, preprocess, multi=False, draw=True, bound1=1.0, bound2=0.6, bound3=0.8):
    image = preprocess(pil_img).unsqueeze(0).to(_device)
    pm0, ps0 = single_quality_map(image, text_embed, pil_img.size)
    pm0 = pm0.cpu().numpy()
    
    if multi:
        ps1 = single_quality_map(image, text_embed_1, pil_img.size)[1]
        ps2 = single_quality_map(image, text_embed_2, pil_img.size)[1]
        ps3 = single_quality_map(image, text_embed_3, pil_img.size)[1]
        
        bound1 = 0.935*bound1
        bound2 = 0.892*bound2
        bound3 = 0.826*bound3
        
        t = min(ps1/bound1,1) * min(ps2/bound1,1) * min(ps3/bound1,1)
        return pm0, ps0*t
    else:
        return pm0, ps0

# Only run the main block if this file is executed directly
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="perceptual-example.jpg", help="input image path")
    parser.add_argument("-d", "--draw", type=bool, default=True, help="draw quality map or not")
    parser.add_argument("-m", "--multi", type=bool, default=False, help="quality multiple dimension")
    args = parser.parse_args()
    
    pil_img = Image.open(args.path)
    pm, ps = perceptual_quality_map(pil_img=pil_img, preprocess=_preprocessor, draw=args.draw, multi=args.multi)
    
    if isinstance(ps, (list, np.ndarray)):
        print('The quality score is: ' + str(ps))
    else:
        print('The quality score is: ' + str(float(ps)))

    if args.draw:
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        vis = (pm * 255).astype('uint8')
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        vis = cv2_img * 0.4 + vis * 0.6
        vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(vis)
        plt.savefig('PQ-Map.png', bbox_inches='tight')