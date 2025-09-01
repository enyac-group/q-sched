# Adapted from example code in https://github.com/yuvalkirstain/PickScore

# import
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import os
import json
from typing import Dict, List, Tuple
import pickle as pkl
import numpy as np
# from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
import hpsv2

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

def load_images_and_captions(image_folder: str, captions_file: str) -> Tuple[List[Image.Image], List[str]]:
    """
    Load images from a folder and their corresponding captions from a JSON file.
    
    Args:
        image_folder: Path to folder containing images
        captions_file: Path to JSON file containing captions
        
    Returns:
        Tuple of (list of PIL Images, list of corresponding captions)
    """
    # Read captions
    with open(captions_file, 'r') as f:
        captions_dict = json.load(f)
    
    images = []
    captions = []
    
    # Load each image and its caption
    for image_name, caption in captions_dict.items():
        image_path = os.path.join(image_folder, image_name)
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                images.append(img)
                captions.append(caption)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                
    return images, captions

def calc_probs(prompt, images):
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
    
    return scores.cpu().tolist()

def evaluate_images_with_captions(image_folder: str, captions_file: str) -> Dict[str, float]:
    """
    Evaluate images using their actual captions as prompts.
    
    Args:
        image_folder: Path to folder containing images
        captions_file: Path to JSON file containing captions
        
    Returns:
        Dictionary mapping image names to their PickScore
    """
    images, captions = load_images_and_captions(image_folder, captions_file)
    
    results = {}
    for i, (image, caption) in enumerate(zip(images, captions)):
        score = calc_probs(caption, [image])[0]
        image_name = list(json.load(open(captions_file)).keys())[i]
        results[image_name] = score
        
    return results

if __name__ == "__main__":
    # Example usage:

    folder="schedule_ablations/sd15_FLUX.1_svdquant_int4_eta0.0"
    f = open(folder + "/dict.pkl", "rb")
    imgs_dict = pkl.load(f)
    f.close()


    imgs = imgs_dict["m1.0_s1.0"]
    imgs =imgs_dict["m0.867_s1.0"]

    num_imgs_per_row = 1
    captions = ["a car and a bus on a french highway" for i in range(0,num_imgs_per_row)]
    captions.extend(["An Eniac computer balanced on top of a stack of rocks over a river" for i in range(0,num_imgs_per_row)])
    captions.extend(["A russian dance with men balancing vodka glasses on their heads" for i in range(0,num_imgs_per_row)])
    captions.extend(["Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" for i in range(0,num_imgs_per_row)])

    results = []
    # metric = PeakSignalNoiseRatio()
    for i, (image, caption) in enumerate(zip(imgs, captions)):
        # score = calc_probs(caption, [image])[0]
        score = hpsv2.score(image, caption, hps_version="v2.1")
        # score = metric(image)
        results.append(score)
    
    print(results)
    print(np.mean(results))
    pass