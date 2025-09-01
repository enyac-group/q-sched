import os
import tempfile
import shutil
from pathlib import Path

# Set up cache directory before any dataset operations
CACHE_DIR = tempfile.mkdtemp(prefix="svdquant_")
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.environ["FILELOCK_DIR"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# Create necessary subdirectories
os.makedirs(os.path.join(CACHE_DIR, "datasets"), exist_ok=True)

# Import remaining modules after setting up directories
from torch.utils.data import Dataset
import datasets
import yaml
import random
from PIL import Image
import json
import collections
import warnings
import logging

# Suppress HuggingFace warnings about missing repo card
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.repocard")
logging.getLogger("huggingface_hub.repocard").setLevel(logging.ERROR)

IMAGE_URL = "https://huggingface.co/datasets/mit-han-lab/svdquant-datasets/resolve/main/sDCI.gz"
PROMPT_URLS = {"sDCI": "https://huggingface.co/datasets/mit-han-lab/svdquant-datasets/resolve/main/sDCI.yaml"}

class MJHQConfig(datasets.BuilderConfig):
    def __init__(self, max_dataset_size: int = -1, return_gt: bool = False, **kwargs):
        super(MJHQConfig, self).__init__(
            name=kwargs.get("name", "default"),
            version=kwargs.get("version", "0.0.0"),
            data_dir=kwargs.get("data_dir", None),
            data_files=kwargs.get("data_files", None),
            description=kwargs.get("description", None),
        )
        self.max_dataset_size = max_dataset_size
        self.return_gt = return_gt
        
class DCIConfig(datasets.BuilderConfig):
    def __init__(self, max_dataset_size: int = -1, return_gt: bool = False, **kwargs):
        super(DCIConfig, self).__init__(
            name=kwargs.get("name", "default"),
            version=kwargs.get("version", "0.0.0"),
            data_dir=kwargs.get("data_dir", None),
            data_files=kwargs.get("data_files", None),
            description=kwargs.get("description", None),
        )
        self.max_dataset_size = max_dataset_size
        self.return_gt = return_gt

class DCI(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    BUILDER_CONFIG_CLASS = DCIConfig
    BUILDER_CONFIGS = [DCIConfig(name="sDCI", version=VERSION, description="sDCI full prompt set")]
    DEFAULT_CONFIG_NAME = "sDCI"

    def _info(self):
        features = datasets.Features(
            {
                "filename": datasets.Value("string"),
                "image": datasets.Image(),
                "prompt": datasets.Value("string"),
                "meta_path": datasets.Value("string"),
                "image_root": datasets.Value("string"),
                "image_path": datasets.Value("string"),
                "split": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description="SVDQuant dataset", features=features, homepage="", license="", citation=""
        )

    def _split_generators(self, dl_manager: datasets.download.DownloadManager):
        image_url = IMAGE_URL
        meta_url = PROMPT_URLS[self.config.name]

        meta_path = dl_manager.download(meta_url)
        image_root = dl_manager.download_and_extract(image_url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"meta_path": meta_path, "image_root": image_root}
            )
        ]

    def _generate_examples(self, meta_path: str, image_root: str):
        meta = yaml.safe_load(open(meta_path, "r"))
        names = list(meta.keys())
        if self.config.max_dataset_size > 0:
            random.Random(0).shuffle(names)
            names = names[: self.config.max_dataset_size]
            names = sorted(names)

        for i, name in enumerate(names):
            prompt = meta[name]
            image_path = os.path.join(image_root, f"{name}.jpg")
            yield i, {
                "filename": name,
                "image": Image.open(image_path) if self.config.return_gt else None,
                "prompt": prompt,
                "meta_path": meta_path,
                "image_root": image_root,
                "image_path": image_path,
                "split": self.config.name,
            }

def load_svdquant_dataset():
    """Load the SVDQuant dataset using the Hugging Face datasets library."""
    print("Loading SVDQuant dataset...")
    try:
        dataset = datasets.load_dataset("mit-han-lab/svdquant-datasets", trust_remote_code=True)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

class SVDQuantDataset(Dataset):
    def __init__(self, split='train'):
        self.dataset = load_svdquant_dataset()[split]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            'prompt': sample['prompt'],
            'image': sample['image']
        }

    def __del__(self):
        # Cleanup cache directory when the dataset is destroyed
        try:
            shutil.rmtree(CACHE_DIR)
        except:
            pass

class KarpathyDataset(Dataset):
    def __init__(self, coco_dir, split='restval'):
        annotations = json.load(open(f"{coco_dir}/dataset_coco.json"))["images"]
        annotation_maps = collections.defaultdict(dict)
        for a in annotations:
            split_name = a['split']
            image_filename = a["filename"]
            annotation_maps[split_name][image_filename] = a
        self.samples = annotation_maps[split]
        self.image_files = list(self.samples.keys())
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        sample = self.samples[image_file]
        return {
            'prompt': [caption["raw"] for caption in sample["sentences"]],
            'image': None  # Images are loaded separately in the main code
        } 