import os
import ast
import numpy as np
import pandas as pd
import torch
from embeds import get_embeds, MultiClaimIndex
from torch.utils.data import DataLoader

def combine_texts(series):
    # Concatenate all non-empty texts into a single string
    return ' '.join(text for text in series if text.strip() != '')

def index_remove_duplicates(index, *arrays):
    """
    Remove duplicate indices from the index and corresponding arrays.
    """
    unique_index, indices = np.unique(index, return_index=True)
    return unique_index, [array[indices] for array in arrays]

def load_multiclaim_files(dataset_path):
    posts_path = os.path.join(dataset_path, 'posts.csv')
    fact_checks_path = os.path.join(dataset_path, 'fact_checks.csv')
    fact_check_post_mapping_path = os.path.join(dataset_path, 'fact_check_post_mapping.csv')

    for path in [posts_path, fact_checks_path, fact_check_post_mapping_path]:
        assert os.path.isfile(path)

    # We need to apply t = t.replace('\n', '\\n') for text fields before using `ast.literal_eval`.
    # `ast.literal_eval` has problems when there are new lines in the text, e.g.:
    # `ast.literal_eval('("\n")')` effectively tries to interpret the following code:

    # ```
    # ("
    # ")
    # ```

    # This raises a SyntaxError exception. By escaping new lines we are able to force it to interpret it properly. There might
    # be some other way to do this more systematically, but it is a workable fix for now.

    parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s

    df_fact_checks = pd.read_csv(fact_checks_path).fillna('').set_index('fact_check_id')
    for col in ['claim', 'instances', 'title']:
        df_fact_checks[col] = df_fact_checks[col].apply(parse_col)

    df_posts = pd.read_csv(posts_path).fillna('').set_index('post_id')
    for col in ['instances', 'ocr', 'verdicts', 'text']:
        df_posts[col] = df_posts[col].apply(parse_col)

    df_fact_check_post_mapping = pd.read_csv(fact_check_post_mapping_path)

    return df_posts, df_fact_checks, df_fact_check_post_mapping

def load_multiclaim(
    dataset_path,
    our_dataset_folder="MultiClaim"
):
    """
    Load datasets, embeddings, and perform preprocessing for multi-claim analysis.
    """

    # Load train, validation, and test datasets
    train_data = pd.read_csv(os.path.join(dataset_path, "train_split.csv"))
    train_data[['post_id', 'fact_check_id']] = train_data[['post_id', 'fact_check_id']].astype(int)

    valid_data = pd.read_csv(os.path.join(dataset_path, "valid_split.csv"))
    valid_data[['post_id', 'fact_check_id']] = valid_data[['post_id', 'fact_check_id']].astype(int)

    test_data = pd.read_csv(os.path.join(dataset_path, "test_split.csv"))
    test_data[['post_id', 'fact_check_id']] = test_data[['post_id', 'fact_check_id']].astype(int)

    # Load fact-check post mapping
    our_dataset_path = os.path.join(dataset_path, our_dataset_folder)
    df_fact_check_post_mapping = pd.read_csv(os.path.join(our_dataset_path, 'fact_check_post_mapping.csv'))

    return (
        train_data,
        valid_data,
        test_data,
        df_fact_check_post_mapping
    )

class IndexedDictDataset(torch.utils.data.Dataset):
    def __init__(self,
        index,
        arg_dicts: dict[str, dict],
        include_indices=True
    ):
        """
        Args:
            arg_dicts (dict[str, dict]): A dictionary mapping from the names
                of arguments to tensor dictionaries keyed by indices present
                in index.
        """

        self.index = np.asarray(index)
        self.arg_dicts = arg_dicts
        self.include_indices = include_indices

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        idx = self.index[idx]

        if self.include_indices:
            return dict(idx=idx, **{k: v[idx] for k, v in self.arg_dicts.items()})
        
        else:
            return {k: v[idx] for k, v in self.arg_dicts.items()}

class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return tuple(dataset[idx] for dataset in self.datasets)

def prepare_dataloader(
    post_features,
    article_features,
    df_index,
    batch_size,
    num_workers,
    shuffle,
    include_indices=True
):
    post_index = df_index.get("post_id")
    article_index = df_index.get("fact_check_id")

    if not post_index is None and not article_index is None:
        if len(post_index) != len(article_index):
            raise BaseException("Index lengths do not match!")

        posts_dataset = IndexedDictDataset(
            post_index,
            post_features,
            include_indices=include_indices
        )

        articles_dataset = IndexedDictDataset(
            article_index,
            article_features,
            include_indices=include_indices
        )

        dataset = ZipDataset(posts_dataset, articles_dataset)
        
    elif not post_index is None:
        posts_dataset = IndexedDictDataset(
            post_index,
            post_features,
            include_indices=include_indices
        )

        dataset = posts_dataset

    elif not article_index is None:
        articles_dataset = IndexedDictDataset(
            article_index,
            article_features,
            include_indices=include_indices
        )

        dataset = articles_dataset

    else:
        raise BaseException("No index provided!")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # this is here to make sure that the generator does not change
        # the state of the global random generator when shuffling is off anyway
        generator=torch.Generator() if not shuffle else None,
        num_workers=num_workers,
        pin_memory=False
    )

    return dataloader

class Dataset:
    def __init__(self,
        dataset_path,
        args,
        post_embeddings,
        article_embeddings,
        posts_index,
        articles_index,
    ):
        embeddings_path = os.path.join(dataset_path, args["embeddings_folder"])

        (train_data, valid_data, test_data, fact_check_post_mapping) = \
            load_multiclaim(dataset_path)
        
        self.posts_index = MultiClaimIndex.from_file(os.path.join(dataset_path, posts_index))
        self.articles_index = MultiClaimIndex.from_file(os.path.join(dataset_path, articles_index))

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.fact_check_post_mapping = fact_check_post_mapping
        self.post_to_fact = self.fact_check_post_mapping.groupby('post_id')['fact_check_id'].apply(list).to_dict()

        self.article_ids_all = self.articles_index.ids
        self.article_ids_in_splits = pd.concat([
            self.train_data["fact_check_id"],
            self.valid_data["fact_check_id"],
            self.test_data["fact_check_id"]
        ]).drop_duplicates().values

        self.post_ids_in_splits = pd.concat([
            self.train_data["post_id"],
            self.valid_data["post_id"],
            self.test_data["post_id"]
        ]).drop_duplicates().values

        self.post_features = self._make_features(
            'posts', post_embeddings, embeddings_path
        )
        
        self.article_features = self._make_features(
            'articles', article_embeddings, embeddings_path
        )

        self.train_dataloader = prepare_dataloader(
            post_features=self.post_features,
            article_features=self.article_features,
            df_index=self.train_data,
            batch_size=args["batch_size"],
            num_workers=args["num_workers"],
            shuffle=True
        )

    def _make_features(self, data, embeddings_cfg, embeddings_path):
        if data == 'posts':
            data_index = self.posts_index
        elif data == 'articles':
            data_index = self.articles_index
        else:
            raise ValueError(f"Invalid data: {data}")

        features = {}

        for k, cfg in embeddings_cfg.items():
            features[k] = get_embeds(
                embeddings_dir=embeddings_path,
                data_index=data_index,
                data=data,
                **cfg
            )

        return features