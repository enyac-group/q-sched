import os
import sys

# Add the Q-Refine directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import clip
import torch
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import spacy
from nltk import Tree
import string
from nltk.corpus import stopwords
import nltk

# Download NLTK data if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Global variables for model caching
_device = 'cuda'
_model = None
_preprocessor = None
_nlp = None

def initialize_models():
    """Initialize and cache the models for reuse"""
    global _model, _preprocessor, _nlp
    
    if _model is None:
        _model, _preprocessor = clip.load("CS-ViT-B/16", device=_device, jit=False)
        _model.eval()
    
    if _nlp is None:
        try:
            _nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load('en_core_web_sm')
    
    return _model, _preprocessor, _nlp

def partition(sentence, noun_phrases):
    if len(noun_phrases) == 1 or len(noun_phrases) == 0:
        return [sentence] 
    parts = []
    cur_part_index = 0
    cur_find_begin_index = 0
    for phrase_idx, cur_noun_phrase in enumerate(noun_phrases):
        find_index = sentence.find(cur_noun_phrase, cur_find_begin_index)
        assert find_index != -1
        if phrase_idx == 0:
            cur_part_index = 0
            cur_find_begin_index = find_index + len(cur_noun_phrase)
        elif phrase_idx == len(noun_phrases) - 1:
            parts.append(sentence[cur_part_index:find_index].strip())
            cur_part_index = find_index
            parts.append(sentence[cur_part_index:].strip())
        else:
            parts.append(sentence[cur_part_index:find_index].strip())
            cur_part_index = find_index
            cur_find_begin_index = find_index + len(cur_noun_phrase)
    return parts

def remove_stopwords_and_punctuation(phrases):
    stop = set(stopwords.words('english')) 
    new_phrases = []
    for phrase in phrases:
        new_phrase= ' '.join([w for w in phrase.split(' ') if w not in stopwords.words('english') and w not in string.punctuation])
        new_phrases.append(new_phrase)
    return new_phrases

def get_token_to_pos_dictionary(doc):
    token_to_pos_dictionary = {}
    for token in doc:
        token_to_pos_dictionary[token.text] = token.pos_
    return token_to_pos_dictionary

def get_phrase_parent(sentence, phrases, filtered_phrases, nltk_tree, token_to_pos_dictionary, spacy_nlp):
    phrase_to_parent = {}
    word_to_phrase = {}
    for phrase, filtered_phrase in zip(phrases, filtered_phrases):
        phrase_to_parent[filtered_phrase] = filtered_phrase
        for token in spacy_nlp(phrase.strip()):
            word_to_phrase[token.text] = filtered_phrase
    
    node_queue = [nltk_tree.root]
    
    while node_queue:
        cur_node = node_queue[0]
        cur_phrase = word_to_phrase[str(cur_node)]
        for child in list(cur_node.children):
            node_queue.append(child)
            if word_to_phrase[str(list(child.ancestors)[0])] != word_to_phrase[str(child)]:
                for cur_ancestor in list(child.ancestors):
                    if token_to_pos_dictionary[str(cur_ancestor)] == "NOUN" or token_to_pos_dictionary[str(cur_ancestor)] == "PROPN":
                        phrase_to_parent[word_to_phrase[str(child)]] = word_to_phrase[str(cur_ancestor)]
                        break
        node_queue = node_queue[1:]
    return phrase_to_parent

def alignment_map(pil_img, preprocess, texts, red=[""], draw=False):
    with torch.no_grad():
        # CLIP architecture surgery acts on the image encoder
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        image = preprocess(pil_img).unsqueeze(0).to(_device)
        image_features = _model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # Prompt ensemble for text features with normalization
        text_features = clip.encode_text_with_prompt_ensemble(_model, texts, _device)

        # Extract redundant features from an empty string
        redundant_features = clip.encode_text_with_prompt_ensemble(_model, red, _device)
        
        features = image_features @ (text_features-redundant_features).t()
        similarity_map = clip.get_similarity_map(features[:, 1:, :], cv2_img.shape[:2])

        feature_ali=features[0][0].cpu().numpy()
        alignment_score=np.exp(10*feature_ali)/(np.exp(10*feature_ali)+1)
        
        similarity_map = clip.get_similarity_map(features[:, 1:, :], cv2_img.shape[:2])
        
        am=[]
        as0=alignment_score
        
        for num in range(len(texts)):
            if draw:
                cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                vis = (similarity_map[0,:,:,num] * 255).cpu().numpy().astype('uint8')
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                vis = cv2_img * 0.4 + vis * 0.6
                vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                print(texts[num])
                plt.axis('off')
                plt.imshow(vis)
                plt.show()
            am.append((similarity_map[0, :, :, num].cpu().numpy() * 255).astype('uint8'))

    return am, as0

def evaluate_aqmap(image_path, query, draw=False):
    """
    Evaluate an image using AQ-Map with the given query
    
    Args:
        image_path: Path to the image file
        query: Text prompt describing the image
        draw: Whether to draw the quality map
        
    Returns:
        alignment_scores: The alignment scores between the image and the query
        The alignment scores are a list of scores, one for each prompt in the query
    """
    # Initialize models if not already done
    model, preprocessor, nlp = initialize_models()
    
    # Process the query
    doc = nlp(query)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    parts = partition(query, noun_phrases)
    filtered_parts = remove_stopwords_and_punctuation(parts)
    token_to_pos_dictionary = get_token_to_pos_dictionary(doc)
    phrase_to_parent_dictionary = get_phrase_parent(query, parts, filtered_parts, list(doc.sents)[0], token_to_pos_dictionary, nlp)

    # Extract noun parts
    noun_parts = []
    for filtered_part in filtered_parts:
        tmp = ''
        for item in filtered_part.split(' '):
            # Strip punctuation from the token before looking it up
            clean_item = item.strip(string.punctuation)
            if clean_item and clean_item in token_to_pos_dictionary and (token_to_pos_dictionary[clean_item]=='NOUN' or token_to_pos_dictionary[clean_item]=='PROPN'):
                tmp = clean_item
        noun_parts.append(tmp)
    
    # Load and process the image
    pil_img = Image.open(image_path)
    
    # Calculate alignment map
    _, alignment_score = alignment_map(pil_img=pil_img, preprocess=preprocessor, texts=noun_parts, red=[""], draw=draw)
    
    return alignment_score

# For backward compatibility with the original script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", type=str, 
        default="alignment-example.jpg", 
        help="input image path",
    )
    parser.add_argument(
        "-d", "--draw", type=bool, 
        default=False, 
        help="draw quality map or not"
    )
    parser.add_argument(
        "-q", "--query", type=str, 
        default="Mr. Beans wearing sun glasses with blue doctor suit and stripe tie", 
        help="prompt"
    )
    args = parser.parse_args()
    
    # Initialize models
    initialize_models()
    
    # Evaluate the image
    alignment_score = evaluate_aqmap(args.path, args.query, args.draw)
    print('The alignment score is: ' + str(alignment_score)) 