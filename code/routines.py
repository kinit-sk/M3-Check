import os
import ast
import abc
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import wandb
import tqdm
import hydra
from omegaconf import OmegaConf
from dataset import prepare_dataloader
from models import ModalitySpec
from embeds import modality_registry
import tempfile

# !!! these imports need to be here: we refer to them from configs !!!
from hydra.utils import get_method 
from dataset import Dataset
# !!! these imports need to be here: we refer to them from configs !!!

# allow the use of eval: and method: in the configs
# eval: allows for the evaluation of python code
# method: allows for the use of functions from the configs
#
# Example:
# emb_dim: ${args.emb_dim}
# emb_dim2: ${eval:'${args.emb_dim}*3'}
# eval_method: ${method:routines.eval_method}
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("method", get_method)

#-------------------------------------------------------------------------------
# Loggers
#-------------------------------------------------------------------------------

from abc import ABC, ABCMeta, abstractmethod

class RandomState:
    def __init__(self):
        self.save_state()

    def save_state(self):
        self.random_state = random.getstate()
        self.np_random_state = np.random.get_state()
        self.torch_random_state = torch.get_rng_state()
        
        if torch.cuda.is_available():
            self.torch_cuda_random_state = torch.cuda.get_rng_state()

    def restore_state(self):
        random.setstate(self.random_state)
        np.random.set_state(self.np_random_state)
        torch.set_rng_state(self.torch_random_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(self.torch_cuda_random_state)

class RandomStatePreserver:
    def __init__(self):
        self.my_state = RandomState()
        self.global_state = RandomState()
        self.nested = False

    def __enter__(self):
        if not self.nested:
            self.global_state.save_state()
            self.my_state.restore_state()
            self.nested = True

    def __exit__(self, exc_type, exc_value, traceback):
        if self.nested:
            self.my_state.save_state()
            self.global_state.restore_state()
            self.nested = False

def preserve_random_state(func):
    def wrapper(self, *args, **kwargs):
        with self._random_state_preserver:
            return func(self, *args, **kwargs)
    return wrapper

class ExperimentTrackerMeta(ABCMeta):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)

        methods = ['initialize', 'log_dir', 'log_metrics', 'log_summary', 'log_artifact', 'finish']

        for method in methods:
            value = dct.get(method)
            if isinstance(value, property):
                setattr(cls, method, property(preserve_random_state(value.fget)))
            else:
                setattr(cls, method, preserve_random_state(value))

class ExperimentTracker(metaclass=ExperimentTrackerMeta):
    def __init__(self, project_name, experiment_name, description, config):
        self._random_state_preserver = RandomStatePreserver()
        self.initialize(project_name, experiment_name, description, config)

    def initialize(self, project_name, experiment_name, description, config):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def log_dir(self):
        raise NotImplementedError
    
    @abstractmethod
    def log_metrics(self, metrics, step=None, commit=None):
        raise NotImplementedError

    @abstractmethod
    def log_summary(self, metrics):
        raise NotImplementedError

    @abstractmethod
    def log_artifact(self, path, name=None):
        raise NotImplementedError

    @abstractmethod
    def finish(self):
        raise NotImplementedError

class DummyTracker(ExperimentTracker):
    def __init__(self,
        project_name,
        experiment_name,
        description,
        config,
        checkpoint_path='outputs'
    ):
        super().__init__(project_name, experiment_name, description, config)
        os.makedirs(checkpoint_path, exist_ok=True)
        self._log_dir = tempfile.TemporaryDirectory(dir=checkpoint_path)

    def initialize(self, project_name, experiment_name, description, config):
        pass

    @property
    def log_dir(self):
        return self._log_dir.name

    def log_metrics(self, metrics, step=None, commit=None):
        pass

    def log_summary(self, metrics):
        pass

    def log_artifact(self, path, name=None):
        pass

    def finish(self):
        self._log_dir.cleanup()

class WandbTracker(ExperimentTracker):
    def initialize(self, project_name, experiment_name, description, config):
        import wandb

        self.wandb_run = wandb.init(
            project=project_name,
            name=experiment_name,
            notes=description,
            config=flatten_dict(config)
        )

    @property
    def log_dir(self):
        return self.wandb_run.dir
    
    def log_metrics(self, metrics, step=None, commit=None):
        return self.wandb_run.log(metrics, step=step, commit=commit)

    def log_summary(self, metrics):
        for name, value in metrics.items():
            self.wandb_run.summary[name] = value

    def log_artifact(self, path, name=None):
        return self.wandb_run.log_artifact(path, name=name)

    def finish(self):
        self.wandb_run.finish()

class MLFlowTracker(ExperimentTracker):
    def __init__(self,
        project_name,
        experiment_name,
        description,
        config,
        checkpoint_path='outputs',
        tracking_uri='http://localhost:5000' # 'sqlite:///mlruns/mlruns.db'
    ):
        self.tracking_uri = tracking_uri
        super().__init__(project_name, experiment_name, description, config)
        os.makedirs(checkpoint_path, exist_ok=True)
        self._log_dir = tempfile.TemporaryDirectory(dir=checkpoint_path)

    def initialize(self, project_name, experiment_name, description, config):
        from mlflow.tracking import MlflowClient

        if self.tracking_uri.startswith('sqlite'):
            base_path = os.path.dirname(self.tracking_uri.split('///')[-1])
            os.makedirs(base_path, exist_ok=True)

        self.step = 0
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        project = self.client.get_experiment_by_name(project_name)

        if project is None:
            self.project_id = self.client.create_experiment(project_name)
        else:
            self.project_id = project.experiment_id

        run = self.client.create_run(self.project_id, run_name=experiment_name)
        self.run_id = run.info.run_id

        # incorporate description and config
        self.client.set_tag(self.run_id, "experiment_name", experiment_name)
        self.client.set_tag(self.run_id, "description", description)

        config = flatten_dict(config)

        # TODO: we need to be able to restore the ~ to - in the keys
        # when reconstructing the nested version of the config
        for k, v in config.items():
            k = k.replace("~", "-") # mlflow does not allow ~ in keys
            self.client.log_param(self.run_id, k, v)

    @property
    def log_dir(self):
        return self._log_dir.name

    def log_metrics(self, metrics, step=None, commit=None):
        if step is None:
            if commit is None:
                commit = True

            step = self.step
        else:
            if commit is None:
                commit = False
            
        for k, v in metrics.items():
            self.client.log_metric(self.run_id, k, v, step=step)

        if commit is True:
            self.step += 1

    def log_summary(self, metrics):
        for k, v in metrics.items():
            self.client.log_metric(self.run_id, "summary/" + k, v)
    
    def log_artifact(self, path, name=None):
        return self.client.log_artifact(self.run_id, path, artifact_path=name)
            
    def finish(self):
        self.client.set_terminated(self.run_id)
        self._log_dir.cleanup()

#-------------------------------------------------------------------------------
# Modality specs
#-------------------------------------------------------------------------------

def make_modality_specs(embeddings):
    emb_specs = {}

    for name, emb in embeddings.items():
        modality = emb['modality']

        if emb['single_embed']:
            shape = modality_registry[modality]['single_shape'](emb['embedder'])
        else:
            shape = modality_registry[modality]['shape'](emb['embedder'])

        emb_specs[name] = ModalitySpec(modality, shape)

    return emb_specs

#-------------------------------------------------------------------------------
# The Experiment class
#-------------------------------------------------------------------------------

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

_base_config = """
build_order:
    - run_args
    - args
    - seed
    - dataset
    - model
    - optimizer
    - scheduler
    - trainer

run_args:
  dataset_path: ../disai_dataset/
  experiment_name: ???
  description: null
  main: trainer

args:
  _target_: routines.make_args
  batch_size: 512
  batch_size_eval: 1024
  num_workers: 8
  epochs: 500
  embedder_text: CLIP
  embedder_image: CLIP
  embedder_ocr: CLIP
  embeddings_folder: embeddings
  checkpoint_metric: recall_1
  checkpoint_metric_min: false
  normalize_features: false
  patience: 150
  seed: 0
  lr: 5e-5
  choose_gpu: 0
  emb_dim: 768
  null_tensor: zeros
  eval_device: null
  logger: ${method:routines.MLFlowTracker}

seed:
    _target_: routines.set_seed
    seed: ${args.seed}

modalities:
  posts:
    image:
      modality: image
      embedder: ${args.embedder_image}
      single_embed: true
      default_embed: zeros
      all_dummy: false

    text:
      modality: text
      embedder: ${args.embedder_text}
      single_embed: true
      language: original
      default_embed: zeros
      all_dummy: false

    ocr:
      modality: ocr
      embedder: ${args.embedder_ocr}
      single_embed: true
      language: original
      default_embed: zeros
      all_dummy: false

  articles: ${modalities.posts}

dataset:
  _target_: routines.Dataset
  args: ${args}
  dataset_path: ${run_args.dataset_path}

  posts_index: posts_index_sha.pkl
  articles_index: articles_index_sha.pkl
  
  post_embeddings: ${modalities.posts}
  article_embeddings: ${modalities.articles}

model:
  _target_: ???

  post_modality_specs:
    _target_: routines.make_modality_specs
    embeddings: ${modalities.posts}
    
  article_modality_specs:
    _target_: routines.make_modality_specs
    embeddings: ${modalities.articles}

optimizer:
    _target_: torch.optim.AdamW
    params: ${eval:'config.model.parameters()'}
    lr: ${args.lr}

scheduler: null

trainer:
  _target_: routines.Trainer
  args: ${args}
  model: ${model}
  dataset: ${dataset}

  _per_target_:
    routines.Trainer:
      optimizer: ${optimizer}
      scheduler: ${scheduler}
"""

def make_args(base_config=_base_config, args_key="args", **args):
    """       
    args:
        batch_size: int
            The batch size to use during training.

        num_workers: int
            The number of workers to use for the DataLoader.

        epochs: int
            The number of epochs to train the model for.

        embedder_text: str
            The embedding model for text.

        embedder_image: str
            The embedding model for images.

        embedder_ocr: str
            The embedding model for OCRs.

        embeddings_folder: str
            The folder the embeddings are in under dataset_path.

        checkpoint_metric: str
            The metric to use for saving the best model. Options:
                "recall_1", "recall_5", "recall_10".

        checkpoint_metric_min: bool
            Whether the checkpoint metric should be minimized.

        normalize_features: bool
            Whether to normalize the features before inference.
        
        patience: int, default
            The number of epochs to wait before early stopping
        
        seed: int, default
            The seed to use for reproducibility.

        lr: float, default
            The learning rate to use during training.

        choose_gpu: int, default
            The GPU to use for training.
    """
    args = OmegaConf.create(args, flags={"allow_objects": True})

    if not base_config is None:
        if isinstance(base_config, str):
            base_config = OmegaConf.create(base_config)

        if args_key:
            args = OmegaConf.merge(base_config[args_key], args)
        else:
            args = OmegaConf.merge(base_config, args)

    if args.batch_size_eval is None:
        args.batch_size_eval = args.batch_size

    return args

class ProcessStep(abc.ABC):
    @abc.abstractmethod
    def __call__(self, config) -> None:
        """
        Processes the config in-place in some way.
        """
        raise NotImplementedError

class RemoveNegKeysStep(ProcessStep):
    def __init__(self, missing_ok=False):
        self.missing_ok = missing_ok
    
    def __call__(self, config):
        for key in list(config.keys()):
            if key.startswith("~"):
                config.pop(key, None)

                if self.missing_ok:
                    config.pop(key[1:], None)
                else:
                    del config[key[1:]]

class MakePerTargetArgsStep(ProcessStep):
    def __call__(self, config):
        per_target = config.get("_per_target_", None)        
        if per_target is None: return

        target = config.get("_target_", None)

        if not target is None:
            per_target_args = per_target.get(target, {})

            for k, v in per_target_args.items():
                config[k] = v

        del config["_per_target_"]

class ConfigProcessor:
    def __init__(self, steps):
        self.steps = steps

    def _proc_config(self, config):
        for step in self.steps:
            step(config)

        for val in config.values():
            if hasattr(val, 'keys'):
                self._proc_config(val)

        return config

    def __call__(self, config):
        config = OmegaConf.to_container(config, resolve=False)
        config = self._proc_config(config)
        return OmegaConf.create(config)

class Experiment:
    __doc__ = make_args.__doc__

    @staticmethod
    def make_base_config(flattened=False):
        base_config = OmegaConf.create(_base_config)

        if flattened:
            base_config = flatten_dict(OmegaConf.to_container(base_config))

        return base_config

    def __init__(
        self,
        config,
        dataset_path=None,
        experiment_name=None,
        description=None,
        main=None,
        config_processor='default'
    ):
        if config_processor == 'default':
            config_processor = ConfigProcessor([
                RemoveNegKeysStep(),
                MakePerTargetArgsStep()
            ])

        base_config = self.__class__.make_base_config()

        if isinstance(config, str):
            config = OmegaConf.create(config)

        config = OmegaConf.merge(base_config, config)
        self.config = config.copy()
        
        if not dataset_path is None:
            config.run_args.dataset_path = dataset_path

        if not experiment_name is None:
            experiment_name = experiment_name.strip()
            config.run_args.experiment_name = experiment_name

        if not description is None:
            description = description.strip()
            config.run_args.description = description

        if not main is None:
            config.run_args.main = main

        # preprocess the config
        if not config_processor is None:
            config = config_processor(config)

        build_order = config.build_order
        config._set_flag(flags=["allow_objects"], values=[True])

        # instantiate everything in the config to set up the trainer
        for k in build_order:
            val = config.get(k, None)
            if not val is None:
                config[k] = hydra.utils.instantiate(val, _convert_="object")

        self.config_built = config

        missing_keys = OmegaConf.missing_keys(self.config_built)
        if missing_keys:
            raise ValueError(f"Missing keys in config: {missing_keys}")

    def __call__(self):
        run_args = self.config_built.run_args
        main = self.config_built[run_args.main]
        config = OmegaConf.to_container(self.config)

        logger = self.config_built.args.logger(
            project_name="disai-multimodal",
            experiment_name=run_args.experiment_name,
            description=run_args.description,
            config=config
        )

        ret = main(logger)

        logger.finish()
        return ret

#-------------------------------------------------------------------------------
# training
#-------------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def make_labels(post_ids, article_ids, post_to_fact):
    labels = torch.zeros(post_ids.shape[0], article_ids.shape[0], device=post_ids.device)

    for i, post_id in enumerate(post_ids):
        post_fact_checks = post_to_fact[post_id.item()]
        l = torch.isin(article_ids, torch.tensor(post_fact_checks, device=post_ids.device))
        l = l / l.sum()
        labels[i, :] = l

    return labels

class ContrastiveLoss:
    @staticmethod
    def from_logits(logits, labels=None):
        if labels is None:
            labels = torch.arange(logits.shape[0]).to(logits.device)

        return F.cross_entropy(logits, labels)  

    @staticmethod
    def from_outputs(post_output, article_output, labels=None):
        logits = post_output @ article_output.T
        return ContrastiveLoss.from_logits(logits, labels)

class DummyTrainer:
    def __init__(
        self, args, model, dataset,
        device=None,
        use_post_to_fact_eval=True
    ):
        self.args = args
        self.model = model
        self.dataset = dataset
        self.use_post_to_fact_eval = use_post_to_fact_eval

        if device is None:
            choose_gpu = args["choose_gpu"]

            if choose_gpu is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                device = torch.device("cuda:" + str(choose_gpu) if torch.cuda.is_available() else "cpu")

        self.device = device

    def __call__(self, logger):
        training_start_time = time.time()

        # run final evaluation
        scores = eval_retrieval(
            self.args, self.model, self.dataset,
            ['train', 'valid', 'test'],
            use_post_to_fact=self.use_post_to_fact_eval,
            use_faiss=True
        )

        train_score = scores['train']
        valid_score = scores['valid']
        test_score = scores['test']

        print(f"Train: {train_score}")
        for key, val in train_score.items():
            logger.log_summary({"train/" + key: val})

        print(f"Validation: {valid_score}")
        for key, val in valid_score.items():
            logger.log_summary({"val/" + key: val})
    
        print(f"Test: {test_score}")
        for key, val in test_score.items():
            logger.log_summary({"test/" + key: val})

        # run final evaluation with all articles as candidates
        scores = eval_retrieval(
            self.args, self.model, self.dataset,
            ['train', 'valid', 'test'],
            use_post_to_fact=self.use_post_to_fact_eval,
            use_all_articles=True, use_faiss=True,
        )

        train_score = scores['train']
        valid_score = scores['valid']
        test_score = scores['test']

        print(f"Train, all ids: {train_score}")
        for key, val in train_score.items():
            logger.log_summary({"all_ids/train/" + key: val})

        print(f"Validation, all ids: {valid_score}")
        for key, val in valid_score.items():
            logger.log_summary({"all_ids/val/" + key: val})
    
        print(f"Test, all ids: {test_score}")
        for key, val in test_score.items():
            logger.log_summary({"all_ids/test/" + key: val})

        # log the total time, etc.
        end_time = time.time()
        total_time = end_time - training_start_time

        logger.log_summary({"total_time": total_time})
        print(f"Total time: {total_time/60:.3g} minutes")

class Trainer:
    def __init__(
        self, args, model, dataset,
        optimizer=None, device=None,
        scheduler=None,
        use_post_to_fact=True,
        use_post_to_fact_eval=None
    ):
        self.args = args
        self.model = model
        self.dataset = dataset
        self.use_post_to_fact = use_post_to_fact

        if use_post_to_fact_eval is None:
            use_post_to_fact_eval = use_post_to_fact

        self.use_post_to_fact_eval = use_post_to_fact_eval

        self.optimizer = optimizer
        self.scheduler = scheduler

        if device is None:
            choose_gpu = args["choose_gpu"]

            if choose_gpu is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                device = torch.device("cuda:" + str(choose_gpu) if torch.cuda.is_available() else "cpu")

        self.device = device

    def __call__(self, logger):
        training_start_time = time.time()

        train_dataloader = self.dataset.train_dataloader
        epochs = self.args["epochs"]
        checkpoint_metric = self.args["checkpoint_metric"]
        patience = self.args["patience"]
        
        patience_counter = 0
        opti_sign = 1 if self.args["checkpoint_metric_min"] else -1
        best_val = float('inf')

        # make sure the checkpoint path exists
        self.best_model_path = os.path.join(logger.log_dir, "best.pth")
        self.last_model_path = os.path.join(logger.log_dir, "last.pth")

        with tqdm.tqdm(total=epochs*len(train_dataloader)) as pbar:
            try:
                for epoch in range(epochs):
                    epoch_start_time = time.time()
                    self.model.to(self.device)

                    metrics = self._run_epoch(
                        model=self.model,
                        input_dataloader=train_dataloader,
                        current_epoch=epoch,
                        optimizer=self.optimizer,
                        device=self.device,
                        pbar=pbar,
                        pbar_postfix={
                            'patience': patience_counter,
                            'best': best_val
                        },
                        logger=logger
                    )                

                    if not self.scheduler is None:
                        self.scheduler.step()

                    val_score = eval_retrieval(
                        self.args, self.model, self.dataset,
                        'valid', use_post_to_fact=self.use_post_to_fact_eval
                    )

                    epoch_time = time.time() - epoch_start_time
                    
                    val_metrics = {
                        "val/" + str(key.lower()): val for key, val in val_score.items()
                    }

                    logger.log_metrics({**metrics, **val_metrics, "epoch_time": epoch_time})
                    current_val = opti_sign * val_score[checkpoint_metric]

                    if current_val < best_val:
                        best_val = current_val
                        patience_counter = 0

                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                            },
                            self.best_model_path
                        )

                        logger.log_artifact(self.best_model_path)
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f"Performance has not improved for {patience} epochs. Stop training at epoch {epoch}!")
                        break

                print(f"Finished Training. Saving the last model and loading the best one from checkpoints.")

                # torch.save(
                #     {
                #         "epoch": epoch,
                #         "model_state_dict": self.model.state_dict(),
                #         "optimizer_state_dict": self.optimizer.state_dict(),
                #     },
                #     self.last_model_path
                # )

                # logger.log_artifact(self.last_model_path)

            except KeyboardInterrupt:
                print("Training interrupted. Running the final evaluation and logging...")
          
            checkpoint = torch.load(self.best_model_path, map_location=torch.device('cpu'), weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # run final evaluation
            scores = eval_retrieval(
                self.args, self.model, self.dataset,
                ['train', 'valid', 'test'],
                use_post_to_fact=self.use_post_to_fact_eval,
                use_faiss=True
            )

            train_score = scores['train']
            valid_score = scores['valid']
            test_score = scores['test']

            print(f"Train: {train_score}")
            for key, val in train_score.items():
                logger.log_summary({"train/" + key: val})

            print(f"Validation: {valid_score}")
            for key, val in valid_score.items():
                logger.log_summary({"val/" + key: val})
        
            print(f"Test: {test_score}")
            for key, val in test_score.items():
                logger.log_summary({"test/" + key: val})

            # run final evaluation with all articles as candidates
            scores = eval_retrieval(
                self.args, self.model, self.dataset,
                ['train', 'valid', 'test'],
                use_post_to_fact=self.use_post_to_fact_eval,
                use_all_articles=True, use_faiss=True,
            )

            train_score = scores['train']
            valid_score = scores['valid']
            test_score = scores['test']

            print(f"Train, all ids: {train_score}")
            for key, val in train_score.items():
                logger.log_summary({"all_ids/train/" + key: val})

            print(f"Validation, all ids: {valid_score}")
            for key, val in valid_score.items():
                logger.log_summary({"all_ids/val/" + key: val})
        
            print(f"Test, all ids: {test_score}")
            for key, val in test_score.items():
                logger.log_summary({"all_ids/test/" + key: val})

            # log the total time, etc.
            end_time = time.time()
            total_time = end_time - training_start_time

            logger.log_summary({"epoch_last": epoch})
            logger.log_summary({"epoch_best": checkpoint["epoch"]})
            logger.log_summary({"total_time": total_time})
            
            print(f"Total time: {total_time/60:.3g} minutes")

    def _run_epoch(
        self,
        model,
        input_dataloader,
        current_epoch,
        optimizer,
        device,
        pbar,
        pbar_postfix,
        logger
    ):
        running_loss = 0.0
        num_samples = 0
        model.train()
        pbar.set_description(f"epoch {current_epoch + 1}/{self.args['epochs']}")

        for i, (data_posts, data_articles) in enumerate(input_dataloader):
            post_ids = data_posts.pop('idx')
            posts_shape = post_ids.shape

            post_tensors = {
                k: v.to(device, non_blocking=True).to(torch.float32)
                    for k, v in data_posts.items()
            }
                
            article_ids = data_articles.pop('idx')

            article_tensors = {
                k: v.to(device, non_blocking=True).to(torch.float32)
                    for k, v in data_articles.items()
            }

            optimizer.zero_grad()
            post_output, article_output = model(post_tensors, article_tensors)

            if self.use_post_to_fact:
                labels = make_labels(post_ids, article_ids, self.dataset.post_to_fact).to(device, non_blocking=True)
                loss = ContrastiveLoss.from_outputs(post_output, article_output, labels=labels)
            else:
                loss = ContrastiveLoss.from_outputs(post_output, article_output)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_samples += posts_shape[0]

            metrics = {
                'train/loss': loss.item() / posts_shape[0],
                'lr': optimizer.param_groups[0]['lr']
            }

            if i + 1 < len(input_dataloader):
                logger.log_metrics(metrics)

            pbar.update()
            pbar.set_postfix(dict(
                pbar_postfix,
                loss=running_loss / num_samples,
                lr=metrics['lr']
            ))

        return metrics

#-------------------------------------------------------------------------------
# Evaluation functions
#-------------------------------------------------------------------------------

@torch.no_grad()
def model_embed(
    embed_func, dataset, df_index,
    batch_size=512, num_workers=8,
    normalize_features=False, device=None,
    target_device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if target_device is None:
        target_device = device
  
    dataloader = prepare_dataloader(
        post_features=dataset.post_features,
        article_features=dataset.article_features,
        df_index=df_index,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    outputs = []

    for data in dataloader:
        _ = data.pop('idx')

        tensors = {
            k: v.to(device, non_blocking=True).to(torch.float32)
                for k, v in data.items()
        }

        with torch.no_grad():
            out = embed_func(tensors)

            if normalize_features:
                out = F.normalize(out, dim=-1).to(target_device, non_blocking=True)

            outputs.append(out)

    outputs = torch.cat(outputs)

    return outputs

@torch.no_grad()
def run_inference(
    args, model, dataset,
    query_data: str|list[str]|pd.DataFrame|dict[str, pd.DataFrame] = None,
    normalize_features=None,
    use_all_articles=False,
    use_faiss=False,
    return_loss=False,
    device=None,
    top_k=10,
    return_dict=None
):
    """
    Args:
        query_data (str|list|pd.DataFrame|dict[str, pd.DataFrame], optional):
            Posts to evaluate on. If a string, it must be one of 'train', 'valid', 'test'.
            If a list, it may contain any of 'train', 'valid', 'test'. If a DataFrame,
            it must contain the column 'post_id'.
            
            If a dictionary, keys represent the name of the fold (they can be arbitrary)
            and the values must be DataFrames with the column 'post_id'.
            
            Defaults to None, which is equivalent to evaluating on the test set.
    """
    
    # query data
    if query_data is None:
        query_data = 'test'

    if isinstance(query_data, list):
        query_data = {qd: getattr(dataset, f"{qd}_data") for qd in query_data}

    if isinstance(query_data, str):
        query_data = getattr(dataset, f"{query_data}_data")

    if isinstance(query_data, dict):
        return_dict = True
    else:
        if return_dict is None:
            return_dict = False
        query_data = {'default': query_data}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # normalize features
    if normalize_features is None:
        normalize_features = args["normalize_features"]

    # embed the articles
    if use_all_articles:
        candidates = dataset.article_ids_all
    else:
        candidates = dataset.article_ids_in_splits

    candidates_t = torch.as_tensor(candidates).to(device)
    fact_check_df_index = pd.DataFrame(candidates, columns=["fact_check_id"])

    # prepare the model
    model.to(device)
    model.eval()

    article_outputs = model_embed(
        embed_func=model.embed_article, dataset=dataset, df_index=fact_check_df_index,
        batch_size=args["batch_size"], num_workers=args["num_workers"],
        normalize_features=normalize_features, device=device,
        target_device='cpu' if use_faiss else None
    )

    if use_faiss:
        if return_loss:
            raise ValueError("Computing the loss (return_loss=True) is not supported when using faiss for retrieval.")

        import faiss
        
        index = faiss.IndexFlatIP(article_outputs.shape[1])
        index.add(article_outputs.cpu().numpy())

    results = {}

    for fold, qd in query_data.items():
        post_ids = qd.post_id.values
        post_outputs = model_embed(
            embed_func=model.embed_post, dataset=dataset, df_index=qd[['post_id']],
            batch_size=args["batch_size"], num_workers=args["num_workers"],
            normalize_features=normalize_features, device=device,
            target_device='cpu' if use_faiss else None
        )

        if use_faiss:
            _, top_k_indices = index.search(post_outputs.cpu().numpy(), top_k)
        else:
            sim = post_outputs @ article_outputs.T
            _, top_k_indices = torch.topk(sim, top_k, dim=1)
            top_k_indices = top_k_indices.to("cpu").numpy()

            if return_loss:
                loss_labels = make_labels(
                    torch.as_tensor(post_ids).to(device),
                    candidates_t,
                    dataset.post_to_fact
                )
                loss = ContrastiveLoss.from_logits(sim, labels=loss_labels)

        top_k_predictions = candidates[top_k_indices]

        results[fold] = {
            'post_ids': post_ids,
            'top_k_predictions': top_k_predictions
        }

        if return_loss:
            results[fold]['loss'] = loss.item()

    if return_dict:
        return results
    else:
        assert len(results) == 1
        return results['default']

def compute_recalls(
    post_ids, top_k_predictions,
    post_to_fact=None, query_data=None
):
    r_1 = 0
    r_5 = 0
    r_10 = 0

    if post_to_fact is None and query_data is None:
        raise ValueError("Either post_to_fact or query_data must be provided.")

    if not post_to_fact is None:
        for post_id, pred in zip(post_ids, top_k_predictions):
            post_fact_checks = post_to_fact[post_id]

            if pred[0] in post_fact_checks:
                r_1 += 1

            if len(set(pred[:5]).intersection(post_fact_checks)) > 0:
                r_5 += 1

            if len(set(pred[:10]).intersection(post_fact_checks)) > 0:
                r_10 += 1
    else:
        labels = query_data.fact_check_id.tolist()

        for pred, label in zip(top_k_predictions, labels):
            if label == pred[0]:
                r_1 += 1

            if label in pred[:5]:
                r_5 += 1

            if label in pred[:10]:
                r_10 += 1

    r_1 = (r_1 / len(top_k_predictions)) * 100
    r_5 = (r_5 / len(top_k_predictions)) * 100
    r_10 = (r_10 / len(top_k_predictions)) * 100

    return {
        "recall_1": r_1,
        "recall_5": r_5,
        "recall_10": r_10
    }

@torch.no_grad()
def eval_retrieval(
    args, model, dataset,
    query_data: str|list[str]|pd.DataFrame|dict[str, pd.DataFrame] = None,
    normalize_features=None,
    use_post_to_fact=True,
    use_all_articles=False,
    use_faiss=False,
    return_loss=False,
    device=None,
    deduplicate_post_ids=None
):
    """
    Args:
        query_data (str|list|pd.DataFrame|dict[str, pd.DataFrame], optional):
            Posts to evaluate on. If a string, it must be one of 'train', 'valid', 'test'.
            If a list, it may contain any of 'train', 'valid', 'test'. If a DataFrame,
            it must contain the column 'post_id'.
            
            If a dictionary, keys represent the name of the fold (they can be arbitrary)
            and the values must be DataFrames with the column 'post_id'.
            
            Defaults to None, which is equivalent to evaluating on the test set.

        use_all_articles (bool, optional): Retrieval candidates include
            fact-check articles that are in no part of the split
            (train, valid, test). If false, candidates are the combined articles
            from the train and valid splits. Defaults to False.

        deduplicate_post_ids (bool, optional): Whether to deduplicate the post ids
            from query_data before running inference. Defaults to None, which
            means True if use_post_to_fact = False (when the matching article
            from query_data is taken as the ground truth match) and False
            otherwise (when all linked articles are taken as ground truth).
    """
    
    # query data
    if query_data is None:
        query_data = 'test'

    if isinstance(query_data, list):
        query_data = {qd: getattr(dataset, f"{qd}_data") for qd in query_data}

    if isinstance(query_data, str):
        query_data = getattr(dataset, f"{query_data}_data")

    if isinstance(query_data, dict):
        return_dict = True
    else:
        return_dict = False
        query_data = {'default': query_data}

    if deduplicate_post_ids is None:
        deduplicate_post_ids = use_post_to_fact

    if deduplicate_post_ids:
        if not use_post_to_fact:
            raise ValueError("Deduplicating post ids is only supported when use_post_to_fact is True.")

        for qd in query_data.values():
            qd.drop_duplicates(subset=['post_id'], inplace=True)

    fold_preds = run_inference(
        args, model, dataset,
        query_data=query_data,
        normalize_features=normalize_features,
        use_all_articles=use_all_articles,
        use_faiss=use_faiss,
        return_loss=return_loss,
        device=device
    )

    results = {}

    for fold, qd in query_data.items():
        preds = fold_preds[fold]
        post_ids = preds['post_ids']
        top_k_predictions = preds['top_k_predictions']

        results[fold] = compute_recalls(
            post_ids=post_ids,
            top_k_predictions=top_k_predictions,
            post_to_fact=dataset.post_to_fact if use_post_to_fact else None,
            query_data=qd
        )

        if return_loss:
            results[fold]['loss'] = preds['loss']

    if return_dict:
        return results
    else:
        assert len(results) == 1
        return results['default']

#-------------------------------------------------------------------------------
# Custom Optimizers
#-------------------------------------------------------------------------------

import torch
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Tuple, Union
from torch import Tensor
import math

class CustomAdam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        maximize: bool = False,       
        accumulate=False,
        m_correction=True,
        v_correction=True,
        use_v=True,
        sqrt_first=False,
        weight_decay: float = 0.01,
        decay_type=None # "adamw" or "adam", None
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
    
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            maximize=maximize,
            accumulate=accumulate,
            m_correction=m_correction,
            v_correction=v_correction,
            use_v=use_v,
            sqrt_first=sqrt_first,
            weight_decay=weight_decay,
            decay_type=decay_type
        )
    
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            maximize = group.get("maximize", False)
            lr = group["lr"]
            eps = group["eps"]

            accumulate = group["accumulate"]
            m_correction = group["m_correction"]
            v_correction = group["v_correction"]
            use_v = group["use_v"]
            sqrt_first = group["sqrt_first"]

            weight_decay = group["weight_decay"]
            decay_type = group["decay_type"]
            
            for p in group["params"]:
                if p.grad is not None:
                    grad = p.grad if not maximize else -p.grad

                    if weight_decay > 0:
                        if decay_type == "adamw":
                            p.mul_(1 - lr * weight_decay)
                        elif decay_type == "adam":
                            grad = grad.add(p, alpha=weight_decay)
                        elif decay_type is None:
                            pass
                        else:
                            raise ValueError(f"Invalid decay type: {decay_type}")

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    step = state["step"]

                    if accumulate:
                        exp_avg.mul_(beta1).add_(grad)
                    else:
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                    if sqrt_first:
                        exp_avg_sq.mul_(beta2).add_(torch.sqrt(grad * grad), alpha=1 - beta2)
                    else:
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    if m_correction:
                        bias_correction_m = 1 - beta1**step
                    else:
                        bias_correction_m = 1

                    if v_correction and use_v:
                        bias_correction_v = 1 - beta2**step
                    else:
                        bias_correction_v = 1

                    step_size = lr * math.sqrt(bias_correction_v) / bias_correction_m

                    if use_v:
                        if sqrt_first:
                            p.add_(-step_size * exp_avg / (exp_avg_sq + eps))
                        else:
                            p.add_(-step_size * exp_avg / (exp_avg_sq.sqrt() + eps))
                    else:
                        p.add_(-step_size * exp_avg)

        return loss