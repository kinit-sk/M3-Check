import abc
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image, UnidentifiedImageError
from imagehash import phash
import hashlib
import pandas as pd
import numpy as np
import pickle
import json
import torch
import os
import ast
from tqdm import tqdm
from collections import defaultdict

#-------------------------------------------------------------------------------
# Data index
#-------------------------------------------------------------------------------

# id, platform, list of images
InstanceType = tuple[int|None, str, list[str]]

class MultiClaimIndex:
    def __init__(self,
        instances: dict[int, list[InstanceType]],
        filtered_images: list[str],
        mapping_dict: dict[str, str],
    ):
        self.instances = instances
        self.filtered_images = filtered_images
        self.mapping_dict = mapping_dict

    @property
    def ids(self):
        return np.array(list(self.instances.keys()))

    @staticmethod
    def from_file(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

#-------------------------------------------------------------------------------
# Model registry
#-------------------------------------------------------------------------------

embedder_registry = {
    'DINOv2': {
        'image_embedder': lambda device: HFEmbedder(
            AutoModel.from_pretrained('facebook/dinov2-base'),
            device
        ),
        'image_transform': lambda: HFDataTransformImage(
            AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        ),
        'image_emb_dim': 768
    },

    'CLIP': {
        'image_embedder': lambda device: CLIPImageEmbedder(
            AutoModel.from_pretrained('openai/clip-vit-large-patch14'),
            device
        ),
        'image_transform': lambda: HFDataTransformImage(
            AutoImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
        ),
        'text_embedder': lambda device: CLIPTextEmbedder(
            AutoModel.from_pretrained('openai/clip-vit-large-patch14'),
            device
        ),
        'text_transform': lambda: HFDataTransformText(
            AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14'),
            max_length=77
        ),
        'image_emb_dim': 768,
        'text_emb_dim': 768
    },

    'LaionCLIP': {
        'image_embedder': lambda device: CLIPImageEmbedder(
            AutoModel.from_pretrained('laion/CLIP-ViT-L-14-laion2B-s32B-b82K'),
            device
        ),
        'image_transform': lambda: HFDataTransformImage(
            AutoImageProcessor.from_pretrained('laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
        ),
        'text_embedder': lambda device: CLIPTextEmbedder(
            AutoModel.from_pretrained('laion/CLIP-ViT-L-14-laion2B-s32B-b82K'),
            device
        ),
        'text_transform': lambda: HFDataTransformText(
            AutoTokenizer.from_pretrained('laion/CLIP-ViT-L-14-laion2B-s32B-b82K'),
            max_length=77
        ),
        'image_emb_dim': 768,
        'text_emb_dim': 768
    },

    'E5': {
        'text_embedder': lambda device: HFEmbedder(
            AutoModel.from_pretrained('intfloat/e5-base'),
            device
        ),
        'text_transform': lambda: HFDataTransformText(
            AutoTokenizer.from_pretrained('intfloat/e5-base')
        ),
        'text_emb_dim': 768
    },

    'E5-large': {
        'text_embedder': lambda device: HFEmbedder(
            AutoModel.from_pretrained('intfloat/e5-large'),
            device
        ),
        'text_transform': lambda: HFDataTransformText(
            AutoTokenizer.from_pretrained('intfloat/e5-large')
        ),
        'text_emb_dim': 1024
    },

    'multilingual-E5': {
        'text_embedder': lambda device: HFEmbedder(
            AutoModel.from_pretrained('intfloat/multilingual-e5-base'),
            device
        ),
        'text_transform': lambda: HFDataTransformText(
            AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
        ),
        'text_emb_dim': 768
    },

    'multilingual-E5-large': {
        'text_embedder': lambda device: HFEmbedder(
            AutoModel.from_pretrained('intfloat/multilingual-e5-large'),
            device
        ),
        'text_transform': lambda: HFDataTransformText(
            AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        ),
        'text_emb_dim': 1024
    },

}

#-------------------------------------------------------------------------------
# Embedding accessor classes
#-------------------------------------------------------------------------------

class PerItemEmbeds:
    def __init__(self, embeddings, missing=None, allow_missing=False, default_value=None, all_dummy=False):
        self.embeddings = embeddings
        self.missing = missing
        self.allow_missing = allow_missing
        self.default_value = default_value
        self.all_dummy = all_dummy

        if isinstance(self.embeddings, str):
            self.embeddings, missing = self.load_from_file(self.embeddings)

            if self.missing is None:
                self.missing = missing

        if self.all_dummy:
            self.embeddings = {}

    def keys(self):
        return self.embeddings.keys()

    def __len__(self):
        return len(self.embeddings)
    
    def get(self, idx, default=None):
        return self.embeddings.get(idx, default)
        
    def __getitem__(self, idx):
        emb = self.embeddings.get(idx, None)

        if emb is None:
            if not self.allow_missing or self.all_dummy:
                raise ValueError(f"Missing embedding for {idx}")
            else:
                return self.default_value

        return emb
    
    @staticmethod
    def load_from_file(file):
        contents = torch.load(file, weights_only=True)
        
        data = contents['data']
        key_list = contents['key_list']
        missing = contents['missing']

        embeddings = {key: data[i] for i, key in enumerate(key_list)}

        return embeddings, missing
    
    @staticmethod
    def from_file(file, allow_missing=False, default_value=None, all_dummy=False):
        embeddings, missing = PerItemEmbeds.load_from_file(file)
        return PerItemEmbeds(embeddings, missing, allow_missing, default_value, all_dummy=all_dummy)
    
    def to_file(self, file):
        if self.all_dummy:
            raise ValueError("Cannot save dummy embeddings")

        shallow_copy = self.embeddings.copy()
        missing = self.missing
        key_list = list(shallow_copy.keys())
        data = torch.stack(list(shallow_copy.values()))

        torch.save({
            'data': data,
            'key_list': key_list,
            'missing': missing
        }, file)

class PerItemListEmbeds(PerItemEmbeds):
    @staticmethod
    def load_from_file(file):
        contents = torch.load(file, weights_only=True)
        
        data = contents['data']
        key_dict = contents['key_dict']
        missing = contents['missing']

        embeddings = {key: data[i:j] for key, (i, j) in key_dict.items()}

        return embeddings, missing

    @staticmethod
    def from_file(file, allow_missing=False, default_value=None, all_dummy=False):
        embeddings, missing = PerItemListEmbeds.load_from_file(file)
        return PerItemListEmbeds(embeddings, missing, allow_missing, default_value, all_dummy=all_dummy)
    
    def to_file(self, file):
        if self.all_dummy:
            raise ValueError("Cannot save dummy embeddings")
        
        missing = self.missing

        data = []
        key_dict = {}
        rolling_index = 0

        for key, value in self.embeddings.items():
            key_dict[key] = (rolling_index, rolling_index + value.shape[0])
            rolling_index += value.shape[0]
            data.append(value)
            
        data = torch.cat(data, dim=0)

        torch.save({
            'data': data,
            'key_dict': key_dict,
            'missing': missing
        }, file)

class PerPathInstanceEmbeds:
    def __init__(self,
        instances,
        embeddings,
        missing=None,
        allow_missing=False,
        all_dummy=False
    ):
        self.instances = instances
        self.embeddings = embeddings
        self.missing = missing
        self.allow_missing = allow_missing
        self.all_dummy = all_dummy

        if isinstance(self.embeddings, str):
            self.embeddings, missing = self.load_from_file(self.embeddings)

            if self.missing is None:
                self.missing = missing

        if self.all_dummy:
            self.embeddings = {}

    def keys(self):
        return self.instances.keys()

    def __len__(self):
        return len(self.instances)
    
    def _gather_instances(self, item_instances):
        instances = []
        
        for inst in item_instances:
            inst_emb = []

            for img_path in inst[2]:
                emb = self.embeddings.get(img_path, None)

                if emb is None:
                    if not self.allow_missing:
                        raise ValueError(f"Missing embedding for {img_path}")
                    else:
                        continue

                inst_emb.append(emb)
                
            if len(inst_emb):
                inst_emb = torch.stack(inst_emb)
            
            instances.append(inst_emb)
            
        return instances

    def get(self, idx, default=None):
        if self.all_dummy: return []
        item_instances = self.instances.get(idx, None)

        if item_instances is None:
            return default
        
        return self._gather_instances(item_instances)

    def __getitem__(self, idx):
        if self.all_dummy: return []
        item_instances = self.instances[idx]
        return self._gather_instances(item_instances)

    @staticmethod
    def load_from_file(file):
        contents = torch.load(file, weights_only=True)
        
        data = contents['data']
        key_list = contents['key_list']
        missing = contents['missing']

        embeddings = {key: data[i] for i, key in enumerate(key_list)}

        return embeddings, missing

    @staticmethod
    def from_file(file, instances, allow_missing=False):
        embeddings, missing = PerPathInstanceEmbeds.load_from_file(file)
        return PerPathInstanceEmbeds(instances, embeddings, missing, allow_missing)
    
    def to_file(self, file):
        if self.all_dummy:
            raise ValueError("Cannot save dummy embeddings")
        
        missing = self.missing
        key_list = list(self.embeddings.keys())
        data = torch.stack(list(self.embeddings.values()))

        torch.save({
            'data': data,
            'key_list': key_list,
            'missing': missing
        }, file)

class EmbedDefaultSelector:
    def __init__(self, embeddings, default_embedding):
        self.embeddings = embeddings
        self.default_embedding = default_embedding

        if isinstance(self.default_embedding, str):
            self.default_embedding = self.embeddings.missing[self.default_embedding]

    def keys(self):
        return self.embeddings.keys()

    def __len__(self):
        return len(self.embeddings)
    
    def get(self, idx, default=None):
        return self.embeddings.get(idx, default)
    
    def __getitem__(self, idx):
        emb = self.embeddings.get(idx, None)

        if emb is None:
            return self.default_embedding
        else:
            return emb

class EmbedInListSelector:
    def __init__(self, embeddings, default_embedding):
        self.embeddings = embeddings
        self.default_embedding = default_embedding

        if isinstance(self.default_embedding, str):
            self.default_embedding = self.embeddings.missing[self.default_embedding]

    def keys(self):
        return self.embeddings.keys()

    def __len__(self):
        return len(self.embeddings)
    
    def get(self, idx, default=None):
        sel = self.embeddings.get(idx, None)

        if sel is None:
            return default

        if len(sel) == 0:
            return self.default_embedding
        else:
            return sel[0]

    def __getitem__(self, idx):
        sel = self.embeddings[idx]

        if len(sel) == 0:
            return self.default_embedding
        else:
            return sel[0]

class EmbedInInstanceSelector:
    def __init__(self, embeddings, default_embedding):
        """
        Given an embeddings object, with per-instance embeddings, this
        returns the first embedding of the first instance; if the first instance has no embeddings, a default embedding is returned instead.
        """

        self.embeddings = embeddings
        self.default_embedding = default_embedding

        if isinstance(self.default_embedding, str):
            self.default_embedding = self.embeddings.missing[self.default_embedding]

    def keys(self):
        return self.embeddings.keys()

    def __len__(self):
        return len(self.embeddings)
    
    def get(self, idx, default=None):
        sel = self.embeddings.get(idx, None)

        if sel is None:
            return default

        if len(sel) == 0 or len(sel[0]) == 0:
            return self.default_embedding
        else:
            return sel[0][0]

    def __getitem__(self, idx):
        sel = self.embeddings[idx]

        if len(sel) == 0 or len(sel[0]) == 0:
            return self.default_embedding
        else:
            return sel[0][0]

#-------------------------------------------------------------------------------
# Modality Registry
#-------------------------------------------------------------------------------

modality_registry = {
    'image': {
        'embed_class': lambda embeddings, data_index, default_embed, all_dummy: PerPathInstanceEmbeds(
            data_index.instances, embeddings, all_dummy=all_dummy
        ),
        'single_embed_class': lambda embeddings, data_index, default_embed, all_dummy: EmbedInInstanceSelector(
            PerPathInstanceEmbeds(
                data_index.instances, embeddings, all_dummy=all_dummy
            ),
            default_embed
        ),
        'shape': lambda embedder: embedder_registry[embedder]['image_emb_dim'],
        'single_shape': lambda embedder: embedder_registry[embedder]['image_emb_dim']
    },

    'text': {
        'embed_class': lambda embeddings, data_index, default_embed, all_dummy: EmbedDefaultSelector(
            PerItemEmbeds(embeddings, all_dummy=all_dummy), default_embed
        ),
        'single_embed_class': lambda embeddings, data_index, default_embed, all_dummy: EmbedDefaultSelector(
            PerItemEmbeds(embeddings, all_dummy=all_dummy), default_embed
        ),
        'shape': lambda embedder: embedder_registry[embedder]['text_emb_dim'],
        'single_shape': lambda embedder: embedder_registry[embedder]['text_emb_dim']
    },

    'ocr': {
        'embed_class': lambda embeddings, data_index, default_embed, all_dummy: PerPathInstanceEmbeds(
            data_index.instances, embeddings, allow_missing=True, all_dummy=all_dummy
        ),
        'single_embed_class': lambda embeddings, data_index, default_embed, all_dummy: EmbedInInstanceSelector(
            PerPathInstanceEmbeds(
                data_index.instances, embeddings, allow_missing=True, all_dummy=all_dummy
            ),
            default_embed
        ),
        'shape': lambda embedder: (None, embedder_registry[embedder]['text_emb_dim']),
        'single_shape': lambda embedder: embedder_registry[embedder]['text_emb_dim']
    },

    'ocr_combined': {
        'embed_class': lambda embeddings, data_index, default_embed, all_dummy: PerItemListEmbeds(
            embeddings, allow_missing=True, default_value=[], all_dummy=all_dummy
        ),
        'single_embed_class': lambda embeddings, data_index, default_embed, all_dummy: EmbedInListSelector(
            PerItemListEmbeds(embeddings, allow_missing=True, default_value=[], all_dummy=all_dummy), default_embed
        ),
        'shape': lambda embedder: (None, embedder_registry[embedder]['text_emb_dim']),
        'single_shape': lambda embedder: embedder_registry[embedder]['text_emb_dim']
    },

    'ocr_all_combined': {
        'embed_class': lambda embeddings, data_index, default_embed, all_dummy: EmbedDefaultSelector(
            PerItemEmbeds(embeddings, all_dummy=all_dummy), default_embed
        ),
        'single_embed_class': lambda embeddings, data_index, default_embed, all_dummy: EmbedDefaultSelector(
            PerItemEmbeds(embeddings, all_dummy=all_dummy), default_embed
        ),
        'shape': lambda embedder: embedder_registry[embedder]['text_emb_dim'],
        'single_shape': lambda embedder: embedder_registry[embedder]['text_emb_dim']
    },
}

modality_registry['oldocr_all_combined'] = modality_registry['ocr_all_combined']

def get_embeds(
    embeddings_dir, data_index, data, embedder, modality, language=None,
    default_embed='zeros', single_embed=True, all_dummy=False
):
    """
    Loads the embeddings for the specified data, embedding model, modality,
    and language from the embeddings directory and returns an object that can be used to access them conveniently by post/article ID.

    Args:
        all_dummy (bool, optional): If True, all embeddings are set to default_embed instead of their actual value.
    """

    if single_embed:
        make_embeds_class = modality_registry[modality]['single_embed_class']
    else:
        make_embeds_class = modality_registry[modality]['embed_class']

    embeds_path = get_embeds_path(embeddings_dir, data, embedder, modality, language)
    embeds = make_embeds_class(embeds_path, data_index, default_embed, all_dummy=all_dummy)

    return embeds

#-------------------------------------------------------------------------------
# Auxiliary functions
#-------------------------------------------------------------------------------

def extract_language(text, language):
    if not len(text):
        return ""
    
    if language == 'translated':
        return text[1]
    elif language == 'original':
        return text[0]
    else:
        raise ValueError(f'Language not supported {language}')

def split_by_lang(instances):
    original = []
    translated = []
    
    for inst in instances:
        if len(inst) < 3:
            continue
        
        original.append(inst[0])
        translated.append(inst[1])
    
    return original, translated

def parse_str_list(str_list):
    return json.loads(str_list.replace("'", '"'))

def parse_instances(instances):
    return [
        (
            int(inst[0]) if not inst[0] is None else None,
            inst[1],
            json.loads(inst[2].replace("'", '"'))
        )
            for inst in instances
    ]

def combine_texts(series):
    # Concatenate all non-empty texts into a single string
    return '\n\n'.join(text for text in series if text.strip() != '')

def count_post_images(post):
    num_images = 0
    
    for inst in post['instances']:
        num_images += len(inst[2])
    
    return num_images

def get_images(instances, flatten=True):
    images = []
    
    if flatten:
        for inst in instances:
            images.extend(inst[2])
    else:
        for inst in instances:
            images.append(inst[2])
    
    return images

def list_files(folder, non_existing_ok=False):
    try:
        return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    except FileNotFoundError as e:
        if non_existing_ok:
            return []
        else:
            raise e

def sha_hash(full_path):
    with open(full_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def phash_hash(full_path):
    img = Image.open(full_path)
    img = img.convert('RGBA')
    return phash(img)

def detect_duplicates(image_paths, image_dir, hash_func=sha_hash):
    hash_dict = {}
    mapping_dict = {}

    for img_path in image_paths:
        full_path = os.path.join(image_dir, img_path)
        try:
            img_hash = hash_func(full_path)

            if img_hash in hash_dict:
                mapping_dict[img_path] = hash_dict[img_hash]
            else:
                hash_dict[img_hash] = img_path
                mapping_dict[img_path] = img_path
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    return list(hash_dict.values()), mapping_dict

def gather_images(df_instances):
    image_set = set()

    for instances in df_instances:
        for inst in instances:
            image_set.update(inst[2])

    return image_set

def filter_images(image_paths, abs_image_dir):
    filtered_images = []

    for img_path in image_paths:
        # check that the image can be loaded
        full_path = os.path.join(abs_image_dir, img_path)

        try:
            img = Image.open(full_path)

        except UnidentifiedImageError:
            continue

        # check that the image is not uselessly tiny
        if min(img.size) < 8:
            continue

        filtered_images.append(img_path)

    return filtered_images

def filter_instance_images(
    instances: list[tuple],
    image_mapping: dict,
    skip_empty: bool = True
):
    """
    Filters the images in the supplied post instances to only contain the 
    ones that are present in image_mapping and remap them using
    image_mapping, which is either an identity mapping if no deduplication
    is performed or a mapping from the original image paths to the deduplicated
    versions.
    
    Args:
        instances (list[tuple]): The list of instances to filter; each instance
            is a tuple of the form (id, source, list of images).
        image_mapping (dict): The images to keep; either an identity mapping
            or a mapping from the original image paths to the deduplicated
            versions.
        skip_empty (bool, optional): Whether to skip instances with no images.
            Defaults to True.

    Returns:
        list[tuple]: The filtered instances.
    """
    assert isinstance(image_mapping, dict)
    filtered_instances = []

    for inst in instances:
        inst_images = []

        for img in inst[2]:
            mapped = image_mapping.get(img, None)

            if mapped is not None:
                inst_images.append(mapped)
        
        if skip_empty and len(inst_images) == 0:
            continue

        # if the same image is repeated in the same instance
        #  after deduplication, keep only one
        inst_images = list(set(inst_images))

        filtered_instances.append(
            (inst[0], inst[1], inst_images)
        )

    return filtered_instances

def filter_and_deduplicate_images(
    dataset_path, image_dir,
    image_set, deduplication_hash='sha', # sha, phash, or None
):
    if deduplication_hash == 'sha':
        hash_func = sha_hash
    elif deduplication_hash == 'phash':
        hash_func = phash_hash
    elif deduplication_hash is None:
        hash_func = None
    else:
        raise ValueError('Invalid deduplication hash')

    filtered_images = filter_images(image_set, os.path.join(dataset_path, image_dir))

    if deduplication_hash is None:
        mapping_dict = {k: k for k in filtered_images}
    else:
        filtered_images, mapping_dict = detect_duplicates(
            filtered_images,
            os.path.join(dataset_path, image_dir),
            hash_func=hash_func
        )

    return filtered_images, mapping_dict

def gather_instance_ocrs(instances, ocr_index):
    ocr_instances = []

    for instance in instances:
        ocrs = []

        for image in instance[2]:
            ocr = ocr_index.get(image, None)
            if not ocr is None: ocrs.append(ocr)
    
        ocr_instances.append(ocrs)

    return ocr_instances

def combine_instance_ocrs(instances):
    return [combine_texts(ocrs) for ocrs in instances]


#-------------------------------------------------------------------------------
# Preprocessors and Embedders
#-------------------------------------------------------------------------------

class HFDataTransformImage:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img):
        proc = self.processor(img, return_tensors='pt')
        proc['pixel_values'] = proc['pixel_values'].squeeze(0)
        return proc
    
class HFDataTransformText:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer

        if max_length is None:
            if hasattr(self.tokenizer, 'model_max_length'):
                max_length = self.tokenizer.model_max_length
            elif hasattr(self.tokenizer, 'max_length'):
                max_length = self.tokenizer.max_length

        self.max_length = max_length

    def __call__(self, text):
        proc = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length)
        
        for k, v in proc.items():
            proc[k] = v.squeeze(0)

        return proc

class Embedder(torch.nn.Module):
    def __init__(self, model, device):
        super().__init__()
        model.eval()
        self.model = model.to(device)
        self.device = device

    @abc.abstractmethod
    def __call__(self, inputs):
        raise NotImplementedError       

class HFEmbedder(Embedder):
    def __call__(self, inputs):
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(**inputs)
            # we only use the CLS token, not the patch tokens
            return outputs['last_hidden_state'][:, 0, :]
        
class CLIPImageEmbedder(Embedder):
    def __call__(self, inputs):
        with torch.no_grad():
            inputs = inputs.to(self.device)
            return self.model.get_image_features(**inputs)

class CLIPTextEmbedder(Embedder):
    def __call__(self, inputs):
        with torch.no_grad():
            inputs = inputs.to(self.device)
            return self.model.get_text_features(**inputs)

#-------------------------------------------------------------------------------
# Datasets
#-------------------------------------------------------------------------------

class DFImageDataset(Dataset):
    def __init__(self, image_path_list, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_path_list = image_path_list

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        img_path = self.image_path_list[idx]
        full_path = os.path.join(self.image_dir, img_path)

        img = Image.open(full_path)
        img = img.convert('RGBA')

        if self.transform:
            img = self.transform(img)

        return img_path, img

class DFImageListDataset(Dataset):
    def __init__(self, df, image_dir, image_col='images_dedup', transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_idx_pairs = self._create_image_idx_pairs(df, image_col)

    def _create_image_idx_pairs(self, df, image_col):
        image_idx_pairs = []

        for idx, post in df.iterrows():
            post_images = post[image_col]

            for img_path in post_images:
                image_idx_pairs.append((idx, img_path))

        return image_idx_pairs

    def __len__(self):
        return len(self.image_idx_pairs)

    def __getitem__(self, idx):
        idx, img_path = self.image_idx_pairs[idx]
        full_path = os.path.join(self.image_dir, img_path)

        img = Image.open(full_path)
        img = img.convert('RGBA')

        if self.transform:
            img = self.transform(img)

        return idx, img

class DFTextColumnDataset(Dataset):
    def __init__(self, df, text_col, transform=None, skip_empty=True):
        if skip_empty:
            df = df[df[text_col] != '']

        self.df = df
        self.text_col = text_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the post text
        post = self.df.iloc[idx]
        text = post[self.text_col]

        # Apply the transform
        if self.transform:
            processed_text = self.transform(text)
        else:
            processed_text = text

        return post.name, processed_text

class DFTextListDataset(Dataset):
    def __init__(self, df, text_col, transform=None, skip_empty=True):
        self.transform = transform
        self.text_idx_pairs = self._create_text_idx_pairs(df, text_col, skip_empty=skip_empty)

    def _create_text_idx_pairs(self, df, text_col, skip_empty):
        text_idx_pairs = []

        for idx, post in df.iterrows():
            post_texts = post[text_col]

            for text in post_texts:
                if skip_empty and len(text.strip()) == 0: continue
                text_idx_pairs.append((idx, text))

        return text_idx_pairs
    
    def __len__(self):
        return len(self.text_idx_pairs)

    def __getitem__(self, idx):
        idx, text = self.text_idx_pairs[idx]

        # Apply the transform
        if self.transform:
            processed_text = self.transform(text)
        else:
            processed_text = text

        return idx, processed_text
    
#-------------------------------------------------------------------------------
# Data loading
#-------------------------------------------------------------------------------

def parse_col(s):
    return ast.literal_eval(s.replace('\n', '\\n')) if s else s

def load_multiclaim_posts_csv(dataset_path):
    posts_path = os.path.join(dataset_path, 'MultiClaim', 'posts.csv')
    assert os.path.isfile(posts_path)

    df_posts = pd.read_csv(posts_path).fillna('').set_index('post_id')
    for col in ['instances', 'ocr', 'verdicts', 'text']:
        df_posts[col] = df_posts[col].apply(parse_col)

    return df_posts

def load_multiclaim_fact_checks_csv(dataset_path):
    fact_checks_path = os.path.join(dataset_path, 'MultiClaim', 'fact_checks.csv')
    assert os.path.isfile(fact_checks_path)

    df_fact_checks = pd.read_csv(fact_checks_path).fillna('').set_index('fact_check_id')
    for col in ['claim', 'instances', 'title']:
        df_fact_checks[col] = df_fact_checks[col].apply(parse_col)

    return df_fact_checks

def load_multiclaim_fact_check_post_mapping_csv(dataset_path):
    fact_check_post_mapping_path = os.path.join(dataset_path, 'MultiClaim', 'fact_check_post_mapping.csv')
    assert os.path.isfile(fact_check_post_mapping_path)
    df_fact_check_post_mapping = pd.read_csv(fact_check_post_mapping_path)
    return df_fact_check_post_mapping

def load_posts(
    dataset_path, image_dir='images',
    deduplication_hash='sha', # sha, phash, or None
    data_index: MultiClaimIndex|str = None
):
    df_posts = load_multiclaim_posts_csv(dataset_path)
    df_posts.rename(columns={'ocr': 'oldocr'}, inplace=True)

    if data_index is None or (
        isinstance(data_index, str)
        and not os.path.exists(data_index)
    ):
        compute_index = True
    else:
        compute_index = False

    if compute_index:
        df_posts['instances'] = df_posts['instances'].apply(parse_instances)
        image_set = gather_images(df_posts['instances'])

        filtered_images, mapping_dict = filter_and_deduplicate_images(
            dataset_path, image_dir, image_set, deduplication_hash
        )

        df_posts['instances'] = df_posts['instances'].apply(lambda x: filter_instance_images(x, mapping_dict))

        if isinstance(data_index, str):
            index = MultiClaimIndex(
                instances=df_posts['instances'].to_dict(),
                filtered_images=filtered_images,
                mapping_dict=mapping_dict
            )
            
            index.save(data_index)
        
    else:
        if isinstance(data_index, str):
            data_index = MultiClaimIndex.from_file(data_index)

        df_posts['instances'] = pd.Series(data_index.instances)
        filtered_images = data_index.filtered_images
        mapping_dict = data_index.mapping_dict
    
    df_posts['images'] = df_posts['instances']

    df_posts["text_original"] = df_posts["text"].apply(lambda x: extract_language(x, 'original'))
    df_posts["text_translated"] = df_posts["text"].apply(lambda x: extract_language(x, 'translated'))

    df_posts[["oldocr_original", "oldocr_translated"]] = df_posts["oldocr"].apply(split_by_lang).apply(pd.Series)
    df_posts['oldocr_original_all_combined'] = df_posts['oldocr_original'].apply(combine_texts)
    df_posts['oldocr_translated_all_combined'] = df_posts['oldocr_translated'].apply(combine_texts)

    df_posts_ocr = pd.read_csv(os.path.join(dataset_path, 'MultiClaim', 'POST_TRANS_OCR_gpt4o.csv'))
    df_posts_ocr['path_image'] = df_posts_ocr['path_image'].apply(lambda x: "/".join(x.split('/')[3:]))
    df_posts_ocr['path_image'] = df_posts_ocr['path_image'].apply(lambda x: mapping_dict[x])
    df_posts_ocr.drop_duplicates(subset='path_image', inplace=True)

    df_posts_ocr.rename(columns={
        'OCR_text': 'ocr_original',
        'OCR_text_TRANSLATED': 'ocr_translated'
    }, inplace=True)

    df_posts_ocr_grouped = df_posts_ocr.groupby('post_id').agg(list)[["ocr_original", "ocr_translated"]]

    df_posts_ocr_grouped['ocr_original_all_combined'] = df_posts_ocr_grouped['ocr_original'].apply(combine_texts)
    df_posts_ocr_grouped['ocr_translated_all_combined'] = df_posts_ocr_grouped['ocr_translated'].apply(combine_texts)

    df_posts = df_posts.merge(df_posts_ocr_grouped[['ocr_original_all_combined', 'ocr_translated_all_combined']], on='post_id', how='left').fillna('')

    # ocrs by instance
    ocr_index = df_posts_ocr.set_index('path_image')['ocr_original']
    ocrs_original = df_posts["images"].apply(lambda x: gather_instance_ocrs(x, ocr_index))
    df_posts["ocr_original_combined"] = ocrs_original.apply(combine_instance_ocrs)

    ocr_index = df_posts_ocr.set_index('path_image')['ocr_translated']
    ocrs_translated = df_posts["images"].apply(lambda x: gather_instance_ocrs(x, ocr_index))
    df_posts["ocr_translated_combined"] = ocrs_translated.apply(combine_instance_ocrs)

    # sanity check
    diff = set(df_posts_ocr['path_image']) - set(filtered_images)

    if len(diff):
        print(f"Warning: missing {len(diff)} images that are in ocr_index")
        print(diff)

    return df_posts, df_posts_ocr, filtered_images, mapping_dict

def load_fact_checks(
    dataset_path, image_dir='images',
    deduplication_hash='sha', # sha, phash, or None
    data_index: MultiClaimIndex|str = None
):
    df_fact_checks = load_multiclaim_fact_checks_csv(dataset_path)

    if data_index is None or (
        isinstance(data_index, str)
        and not os.path.exists(data_index)
    ):
        compute_index = True
    else:
        compute_index = False

    if compute_index:
        image_dirs = df_fact_checks.index.to_series().astype(str).apply(
            lambda x: os.path.join('drive_images', f'image_{x}')
        )

        df_fact_checks['images'] = image_dirs.apply(
            lambda imdir: [(
                None, None,
                [os.path.join(imdir, f)
                for f in list_files(os.path.join(dataset_path, image_dir, imdir), non_existing_ok=True)]
            )]
        )

        assert df_fact_checks['images'].apply(len).sum() > 0, "No fact-checks have images"
        image_set = gather_images(df_fact_checks['images'])

        filtered_images, mapping_dict = filter_and_deduplicate_images(
            dataset_path, image_dir, image_set, deduplication_hash
        )

        df_fact_checks['images'] = df_fact_checks['images'].apply(
            lambda x: filter_instance_images(x, mapping_dict)
        )

        if isinstance(data_index, str):
            index = MultiClaimIndex(
                instances=df_fact_checks['images'].to_dict(),
                filtered_images=filtered_images,
                mapping_dict=mapping_dict
            )
            
            index.save(data_index)
            
    else:
        if isinstance(data_index, str):
            data_index = MultiClaimIndex.from_file(data_index)

        df_fact_checks['images'] = pd.Series(data_index.instances)
        filtered_images = data_index.filtered_images
        mapping_dict = data_index.mapping_dict

    df_fact_checks["text_original"] = df_fact_checks["claim"].apply(lambda x: extract_language(x, 'original'))
    df_fact_checks["text_translated"] = df_fact_checks["claim"].apply(lambda x: extract_language(x, 'translated'))

    df_fact_checks_ocr = pd.read_csv(
        os.path.join(dataset_path, 'MultiClaim', 'TRANS_OCR_gpt4o_fc.csv')
    ).fillna('')

    df_fact_checks_ocr['path_image'] = df_fact_checks_ocr['path_image'].apply(lambda x: x.replace('\\', '/')[1:])
    df_fact_checks_ocr['path_image'] = df_fact_checks_ocr['path_image'].apply(lambda x: mapping_dict[x])
    df_fact_checks_ocr.drop_duplicates(subset='path_image', inplace=True)

    df_fact_checks_ocr.rename(columns={
        'OCR_text': 'ocr_original',
        'OCR_text_TRANSLATED': 'ocr_translated'
    }, inplace=True)

    df_fact_checks_ocr_grouped = df_fact_checks_ocr.groupby('fact_check_id').agg(list)[["ocr_original", "ocr_translated"]]

    df_fact_checks_ocr_grouped['ocr_original_all_combined'] = df_fact_checks_ocr_grouped['ocr_original'].apply(combine_texts)
    df_fact_checks_ocr_grouped['ocr_translated_all_combined'] = df_fact_checks_ocr_grouped['ocr_translated'].apply(combine_texts)

    df_fact_checks = df_fact_checks.merge(df_fact_checks_ocr_grouped[['ocr_original_all_combined', 'ocr_translated_all_combined']], on='fact_check_id', how='left').fillna('')

    # ocrs by instance
    ocr_index = df_fact_checks_ocr.set_index('path_image')['ocr_original']
    ocrs_original = df_fact_checks["images"].apply(lambda x: gather_instance_ocrs(x, ocr_index))
    df_fact_checks["ocr_original_combined"] = ocrs_original.apply(combine_instance_ocrs)

    ocr_index = df_fact_checks_ocr.set_index('path_image')['ocr_translated']
    ocrs_translated = df_fact_checks["images"].apply(lambda x: gather_instance_ocrs(x, ocr_index))
    df_fact_checks["ocr_translated_combined"] = ocrs_translated.apply(combine_instance_ocrs)

    # sanity check
    diff = set(df_fact_checks_ocr['path_image']) - set(filtered_images)

    if len(diff):
        print(f"Warning: missing {len(diff)} images that are in ocr_index")
        print(diff)

    return df_fact_checks, df_fact_checks_ocr, filtered_images, mapping_dict

#-------------------------------------------------------------------------------
# Embedding functions
#-------------------------------------------------------------------------------

def make_missing_text_embeds(embedder, transform, device):
    """
    Returns a dictionary of embeddings that can be used to replace missing
    text embeddings. Currently returns:
        {
            'zeros': an all-zero embedding tensor,
            'empty_string': the embedding of an empty string
        }
    """
    x = torch.utils.data.default_collate([transform("")]).to(device)
    y = embedder(x).squeeze(0).cpu()
    
    return {
        'zeros': torch.zeros_like(y),
        'empty_string': y
    }

def make_missing_image_embeds(embedder, transform, device):
    """
    Returns a dictionary of embeddings that can be used to replace missing
    image embeddings. Currently returns:
        {
            'zeros': an all-zero embedding tensor,
            'black_image': an embedding of an all-black image,
            'white_image': an embedding of an all-white image,
            'gray_image': an embedding of an all-gray image
        }.
    """

    black_image = Image.new('RGB', (512, 512), (0, 0, 0))
    white_image = Image.new('RGB', (512, 512), (255, 255, 255))
    gray_image = Image.new('RGB', (512, 512), (128, 128, 128))

    x = torch.utils.data.default_collate([
        transform(black_image),
        transform(white_image),
        transform(gray_image)
    ]).to(device)

    y = embedder(x).squeeze(0).cpu()
    
    return {
        'zeros': torch.zeros_like(y[0]),
        'black_image': y[0],
        'white_image': y[1],
        'gray_image': y[2]
    }

def get_embeds_path(embeddings_dir, data, embedder, modality, language=None):
    if modality == 'image':
        fname =  f'{data}_{modality}_{embedder}.pt'
    else:
        if language is None:
            raise ValueError('Language must be specified for text embeddings')

        fname = f'{data}_{modality}_{language}_{embedder}.pt'

    return os.path.join(embeddings_dir, fname)

#-------------------------------------------------------------------------------
# The main make_embeds function
#-------------------------------------------------------------------------------

def make_embeds(args):
    embeddings_dir = args.embeddings_dir
    if args.embeddings_dir is None:
        embeddings_dir = os.path.join(args.dataset_path, 'embeddings')

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.deduplication_hash == 'None':
        args.deduplication_hash = None

    if args.deduplication_hash not in ['sha', 'phash', None]:
        raise ValueError(f'Invalid deduplication hash "{args.deduplication_hash}"')
    
    if isinstance(args.index, str) and (args.index == 'None' or args.index == 'False'):
        args.index = None
        
    model_spec = embedder_registry.get(args.model, None)

    if model_spec is None:
        raise ValueError(f'Model not found: {args.model}')

    if args.data == 'posts':
        if args.index == 'auto':
            hash_type = args.deduplication_hash if args.deduplication_hash is not None else "nohash"

            args.index = os.path.join(args.dataset_path, f"posts_index_{hash_type}.pkl")

        df, df_ocr, filtered_images, mapping_dict = load_posts(
            dataset_path=args.dataset_path,
            image_dir=args.image_dir,
            deduplication_hash=args.deduplication_hash,
            data_index=args.index
        )
    elif args.data == 'articles':
        if args.index == 'auto':
            hash_type = args.deduplication_hash if args.deduplication_hash is not None else "nohash"

            args.index = os.path.join(args.dataset_path, f"articles_index_{hash_type}.pkl")

        df, df_ocr, filtered_images, mapping_dict = load_fact_checks(
            dataset_path=args.dataset_path,
            image_dir=args.image_dir,
            deduplication_hash=args.deduplication_hash,
            data_index=args.index
        )
    else:
        raise ValueError(f'Data not supported: {args.data}')

    if args.type == 'image':
        # images stored in a flat dict, indexed by image paths

        # load model and transform
        transform = model_spec['image_transform']()
        embedder = model_spec['image_embedder'](device)

        if not args.no_compile:
            embedder = torch.compile(embedder)

        # create dataset and dataloader
        dataset = DFImageDataset(
            filtered_images,
            os.path.join(args.dataset_path, args.image_dir),
            transform=transform
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        # compute embeddings
        embeddings = {}

        for ids, tensors in tqdm(dataloader, desc="Processing batches"):
            embeds = embedder(tensors)
            for idx, embed in enumerate(embeds):
                embeddings[ids[idx]] = embed.cpu()

        missing = make_missing_image_embeds(embedder, transform, device)

        # save embeddings
        if args.output is not None:
            output_path = args.output
        else:
            output_path = get_embeds_path(embeddings_dir, args.data, args.model, args.type)
    
        print(f'Saving embeddings to {output_path}')
        PerPathInstanceEmbeds(df['images'].to_dict(), embeddings, missing).to_file(output_path)

    elif args.type == 'text' or args.type == 'oldocr_all_combined' or args.type == 'ocr_all_combined':
        # texts and all_combined ocrs are indexed by post/article ids, with a single embedding for each

        # pick the text col
        if args.type == 'text':
            if args.language == 'translated':
                text_col = "text_translated"
            elif args.language == 'original':
                text_col = "text_original"
            else:
                raise ValueError(f'Language not supported: {args.language}')
        elif args.type == 'oldocr_all_combined':
            if args.language == 'translated':
                text_col = "oldocr_translated_all_combined"
            elif args.language == 'original':
                text_col = "oldocr_original_all_combined"
            else:
                raise ValueError(f'Language not supported: {args.language}')
        else: 
            if args.language == 'translated':
                text_col = "ocr_translated_all_combined"
            elif args.language == 'original':
                text_col = "ocr_original_all_combined"
            else:
                raise ValueError(f'Language not supported: {args.language}')
            
        # load model and transform
        transform = model_spec['text_transform']()
        embedder = model_spec['text_embedder'](device)

        if not args.no_compile:
            embedder = torch.compile(embedder)

        # create dataset and dataloader
        dataset = DFTextColumnDataset(
            df=df,
            text_col=text_col,
            transform=transform
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        # compute embeddings
        embeddings = defaultdict(list)

        for post_id, tensors in tqdm(dataloader, desc="Processing batches"):
            embeds = embedder(tensors)
            for idx, embed in enumerate(embeds):
                embeddings[post_id[idx].item()].append(embed.cpu())

        tmp_embeddings = embeddings
        embeddings = {}

        for k, v in tmp_embeddings.items():
            assert len(v) == 1, f'Expected 1 embedding for post {k}, got {len(v)}'
            embeddings[k] = v[0]

        missing = make_missing_text_embeds(embedder, transform, device)

        # save embeddings
        if args.output is not None:
            output_path = args.output
        else:
            output_path = get_embeds_path(embeddings_dir, args.data, args.model, args.type, args.language)
            
        print(f'Saving embeddings to {output_path}')
        PerItemEmbeds(embeddings, missing).to_file(output_path)

    elif args.type == 'ocr_combined':
        # ocr combined are indexed by post/article ids, with a combined embedding for each instance

        if args.language == 'translated':
            text_col = "ocr_translated_combined"
        elif args.language == 'original':
            text_col = "ocr_original_combined"
        else:
            raise ValueError(f'Language not supported: {args.language}')

        # load model and transform
        transform = model_spec['text_transform']()
        embedder = model_spec['text_embedder'](device)

        if not args.no_compile:
            embedder = torch.compile(embedder)

        # create dataset and dataloader
        dataset = DFTextListDataset(
            df,
            text_col,
            transform=transform
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        # compute embeddings
        embeddings = defaultdict(list)

        for ids, tensors in tqdm(dataloader, desc="Processing batches"):
            embeds = embedder(tensors)
            for idx, embed in enumerate(embeds):
                embeddings[ids[idx].item()].append(embed.cpu())

        embeddings = {k: torch.stack(v) for k, v in embeddings.items()}
        missing = make_missing_text_embeds(embedder, transform, device)

        # save embeddings
        if args.output is not None:
            output_path = args.output
        else:
            output_path = get_embeds_path(embeddings_dir, args.data, args.model, args.type, args.language)
    
        print(f'Saving embeddings to {output_path}')
        PerItemListEmbeds(embeddings, missing).to_file(output_path)

    elif args.type == 'ocr':
        # ocrs are save in a flat dictionary, keyed by image path
        if args.language == 'translated':
            text_col = "ocr_translated"
        elif args.language == 'original':
            text_col = "ocr_original"
        else:
            raise ValueError(f'Language not supported: {args.language}')

        # load model and transform
        transform = model_spec['text_transform']()
        embedder = model_spec['text_embedder'](device)

        if not args.no_compile:
            embedder = torch.compile(embedder)

        df_ocr = df_ocr.set_index('path_image')[[text_col]]
        dataset = DFTextColumnDataset(
            df_ocr, text_col,
            transform=transform
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        # compute embeddings
        embeddings = {}

        for ids, tensors in tqdm(dataloader, desc="Processing batches"):
            embeds = embedder(tensors)
            for idx, embed in enumerate(embeds):
                embeddings[ids[idx]] = embed.cpu()

        missing = make_missing_text_embeds(embedder, transform, device)

        # save embeddings
        if args.output is not None:
            output_path = args.output
        else:
            output_path = get_embeds_path(embeddings_dir, args.data, args.model, args.type, args.language)
    
        print(f'Saving embeddings to {output_path}')
        PerPathInstanceEmbeds(df['images'].to_dict(), embeddings, missing).to_file(output_path)

    elif args.type == 'none':
        pass

    else:
        raise ValueError(f'Type not supported: {args.type}')