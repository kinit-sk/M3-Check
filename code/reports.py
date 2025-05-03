import markdown
from IPython.display import display, HTML
from omegaconf import OmegaConf
from pygments import highlight
from pygments.lexers import YamlLexer
from pygments.formatters import HtmlFormatter
import pandas as pd
import numpy as np
from routines import Experiment
from abc import ABC, abstractmethod
import os

def unflatten_dict(d):
    result = {}
    for key, value in d.items():
        parts = key.split('.')
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]

        d[parts[-1]] = value
    return result

def config_diff(config1, config2):
    """Computes the difference between two configs. Note: this requires
    the configs to be flat dicts; you can use flatten_dict() to achieve this."""
    diff_config = {}

    for key in config1:
        if key not in config2 or config1[key] != config2[key]:
            diff_config[key] = config1[key]

    return diff_config

#-------------------------------------------------------------------------------
# Log inspectors and the reporter
#-------------------------------------------------------------------------------

class LogInspector(ABC):
    @abstractmethod
    def set_project(self, project_name):
        raise NotImplementedError

    @abstractmethod
    def get_project_names(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_runs(self, experiment_name=None):
        raise NotImplementedError
    
    @abstractmethod
    def get_experiment_names(self, runs=None):
        raise NotImplementedError
    
    @abstractmethod
    def get_metrics(self, experiment_name, metrics="test/recall_1"):
        raise NotImplementedError

    @abstractmethod
    def list_artifacts(self, run, return_raw=False):
        raise NotImplementedError
    
    @abstractmethod
    def download_artifact(self, run, name, path, replace=False, exist_ok=False):
        raise NotImplementedError

class WandbLogInspector(LogInspector):
    def __init__(self, entity, project_name: str|None = None, wandb_api=None):
        super().__init__()
        
        if wandb_api is None:
            import wandb
            wandb_api = wandb.Api()

        self.api = wandb_api
        self.entity = entity
        self.project_name = project_name

    def set_project(self, project_name):
        self.project_name = project_name

    def get_project_names(self):
        return [p.name for p in self.api.projects(self.entity)]

    def get_runs(self, experiment_name=None):
        return self.api.runs(
            path=f"{self.entity}/{self.project_name}",
            filters={"display_name": experiment_name} if experiment_name else None
        )
    
    def get_experiment_names(self, runs=None):
        if runs is None:
            runs = self.get_runs()

        return sorted(set(run.name for run in runs))

    def get_metrics(self, experiment_name, metrics="test/recall_1"):
        if isinstance(metrics, str):
            metrics = [metrics]
            return_dict = False
        else:
            return_dict = True

        runs = self.get_runs(experiment_name)
        metrics = {metric: [] for metric in metrics}

        for run in runs:
            if run.state != "finished":
                continue

            for metric in metrics:
                metrics[metric].append(run.summary[metric])

        for metric in metrics:
            metrics[metric] = np.array(metrics[metric])

        if return_dict:
            return metrics
        else:
            return next(iter(metrics.values()))
        
    def get_history(self, run, metrics="val/recall_1"):
        if isinstance(metrics, str):
            metrics = [metrics]
            return_array = True
        else:
            return_array = False

        history = {}

        for metric in metrics:
            history_scan = run.scan_history(keys=["_step", metric])

            steps = []
            values = []
        
            for batch in history_scan:
                steps.append(batch["_step"])
                values.append(batch[metric])

            history[metric] = {'step': np.array(steps), 'value': np.array(values)}

        if return_array:
            return history[metrics[0]]
        else:
            return history

    def make_config(self, run, diff_only=False):
        config = run.config

        if diff_only:
            base_config = Experiment.make_base_config(flattened=True)
  
            config = config_diff(
                OmegaConf.to_container(config, resolve=False) if isinstance(config, OmegaConf) else config,
                OmegaConf.to_container(base_config, resolve=False) if isinstance(base_config, OmegaConf) else base_config
            )           

        return OmegaConf.create(unflatten_dict(config))

    def list_artifacts(self, run, return_raw=False):
        artifacts = run.files()

        if return_raw:
            return artifacts

        return [a.name for a in artifacts]
    
    def download_artifact(self, run, name, path, replace=False, exist_ok=False):
        return run.file(name).download(path, replace=replace, exist_ok=exist_ok)

class MLFlowLogInspector(LogInspector):
    def __init__(self, client, project_name: str|None = None):
        super().__init__()
        self.client = client
        self.project_name = project_name

        if project_name is not None:
            self.set_project(project_name)  

    def set_project(self, project_name: str):
        self.project_name = project_name
        projects = self.client.search_experiments(filter_string=f"name='{project_name}'")
        self.project_ids = [p.experiment_id for p in projects]

    def get_project_names(self):
        projects = self.client.search_experiments()
        return [p.name for p in projects]

    def get_runs(self, experiment_name=None):
        if experiment_name is None:
            runs = self.client.search_runs(self.project_ids)
        else:
            runs = self.client.search_runs(
                self.project_ids,
                filter_string=f"run_name='{experiment_name}'"
            )
            
        runs = sorted(runs, key=lambda run: run.info.start_time)
        return runs

    def get_experiment_names(self, runs=None):
        if runs is None:
            runs = self.get_runs()

        return sorted(set(run.info.run_name for run in runs))

    def get_metrics(self, experiment_name, metrics="test/recall_1"):
        if isinstance(metrics, str):
            metrics = [metrics]
            return_dict = False
        else:
            return_dict = True

        runs = self.get_runs(experiment_name)
        metrics = {metric: [] for metric in metrics}

        for run in runs:
            if run.info.end_time is None:
                continue

            for metric in metrics:
                metrics[metric].append(
                    run.data.metrics['summary/' + metric])

        for metric in metrics:
            metrics[metric] = np.array(metrics[metric])

        if return_dict:
            return metrics
        else:
            return next(iter(metrics.values()))
        
    def get_history(self, run, metrics="val/recall_1"):        
        if isinstance(metrics, str):
            metrics = [metrics]
            return_array = True
        else:
            return_array = False

        history = {}

        for metric in metrics:
            h = self.client.get_metric_history(run.info.run_id, metric)

            steps = []
            values = []

            for x in h:
                steps.append(x.step)
                values.append(x.value)

            history[metric] = {'step': np.array(steps), 'value': np.array(values)}

        if return_array:
            return history[metrics[0]]
        else:
            return history

    def _fix_config_key(self, k):
        parts = k.split('.')
        if parts[-1][0] == '-':
            parts[-1] = '~' + parts[-1][1:]
        
        return '.'.join(parts)

    def make_config(self, run, diff_only=False):
        config = run.data.params
        config = {self._fix_config_key(k): v for k, v in config.items()}

        # fix string representation of values
        for key, value in config.items():
            if isinstance(value, str):
                if value[0] == '[':
                    value = OmegaConf.create(value)
                    value = OmegaConf.to_container(value, resolve=False)
                    config[key] = value
                elif value == 'None':
                    config[key] = None
                elif value in ['True', 'False']:
                    config[key] = value == 'True'
                elif value in ['true', 'false']:
                    config[key] = value == 'true'
                else:
                    try:
                        config[key] = int(value)
                    except (ValueError, TypeError):
                        try:
                            config[key] = float(value)
                        except ValueError:
                            pass
            
        if diff_only:
            base_config = Experiment.make_base_config(flattened=True)
  
            config = config_diff(
                OmegaConf.to_container(config, resolve=False) if isinstance(config, OmegaConf) else config,
                OmegaConf.to_container(base_config, resolve=False) if isinstance(base_config, OmegaConf) else base_config
            )           

        return OmegaConf.create(unflatten_dict(config))
    
    def list_artifacts(self, run, return_raw=False):
        artifacts = self.client.list_artifacts(run.info.run_id)

        if return_raw:
            return artifacts

        return [a.path for a in artifacts]
    
    def download_artifact(self, run, name, path, replace=False, exist_ok=False):
        if os.path.exists(path):
            if replace:
                pass
            elif exist_ok:
                return
            else:
                raise ValueError(f"File '{path}' already exists pass replace=True to overwrite or exist_ok=True to leave it as is without raising.")
        
        return self.client.download_artifacts(run.info.run_id, name, path)

class WandbReporter:
    def __init__(self, entity, project):
        import wandb

        self.api = wandb.Api()
        self.inspector = WandbLogInspector(entity, project, self.api)
        self.entity = entity
        self.project = project

    def make_summary(self, run):
        summary = run.summary
        timestamp = pd.Timestamp(summary['_timestamp'], unit='s').strftime('%Y-%m-%d %H:%M:%S')

        summary_html = f"""
        <style>
            .dashboard {{
                display: flex;
                flex-wrap: wrap;
            }}
            .dashboard .column {{
                flex: 1;
                padding: 10px;
                box-sizing: border-box;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .equal-width-table th, .equal-width-table td {{
                width: 25%;
            }}
            .training-summary th {{
                width: 40%;
            }}
        </style>
        <div class="dashboard">
            <div class="column">
                <h3><b>Metrics</b></h3>
                <table class="equal-width-table">
                    <tr>
                        <th><b>Metric</b></th>
                        <th><b>Train</b></th>
                        <th><b>Validation</b></th>
                        <th><b>Test</b></th>
                    </tr>
                    <tr>
                        <th><b>Loss</b></th>
                        <td>{summary.get('train/loss', float('nan')):.6f}</td>
                        <td>{summary.get('val/loss', float('nan')):.6f}</td>
                        <td>{summary.get('test/loss', float('nan')):.6f}</td>
                    </tr>
                    <tr>
                        <th><b>Recall@1</b></th>
                        <td>{summary.get('train/recall_1', float('nan')):.2f}</td>
                        <td>{summary.get('val/recall_1', float('nan')):.2f}</td>
                        <td>{summary.get('test/recall_1', float('nan')):.2f}</td>
                    </tr>
                    <tr>
                        <th><b>Recall@5</b></th>
                        <td>{summary.get('train/recall_5', float('nan')):.2f}</td>
                        <td>{summary.get('val/recall_5', float('nan')):.2f}</td>
                        <td>{summary.get('test/recall_5', float('nan')):.2f}</td>
                    </tr>
                    <tr>
                        <th><b>Recall@10</b></th>
                        <td>{summary.get('train/recall_10', float('nan')):.2f}</td>
                        <td>{summary.get('val/recall_10', float('nan')):.2f}</td>
                        <td>{summary.get('test/recall_10', float('nan')):.2f}</td>
                    </tr>
                </table>
            </div>
            <div class="column">
                <h3><b>Training Summary</b></h3>
                <table class="training-summary">
                    <tr><th><b>Runtime</b></th><td>{summary['_runtime']/60:.2f} min</td></tr>
                    <tr><th><b>Best Epoch</b></th><td>{summary['epoch_best']}</td></tr>
                    <tr><th><b>Last Epoch</b></th><td>{summary['epoch_last']}</td></tr>
                    <tr><th><b>Total Training Time</b></th><td>{summary['total_time']:.2f} seconds</td></tr>
                    <tr><th><b>Timestamp</b></th><td>{timestamp}</td></tr>
                </table>
            </div>
        </div>
        """

        return summary_html

    def make_config(self, run, diff_only=True, format=True):
        if diff_only:
            config = self.inspector.make_config(run, diff_only=True)
            full_config = None
        else:
            config = self.inspector.make_config(run, diff_only=True)
            full_config = run.config = self.inspector.make_config(run, diff_only=False)

        if not format: return config

        config_yaml = OmegaConf.to_yaml(config)
        # Add newlines before top-level keys for better readability
        config_yaml = "\n".join([f"\n{line}" if not line.startswith(" ") else line for line in config_yaml.splitlines()])

        formatter = HtmlFormatter(style="colorful")
        highlighted_config_yaml = highlight(config_yaml, YamlLexer(), formatter)

        # HTML formatting with code highlighting
        html = f"""
        <style>
            {formatter.get_style_defs('.highlight')}
            pre {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            padding: 10px;
            overflow: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Courier New', Courier, monospace; /* Change font for better readability */
            }}
            .highlight .l-Scalar-Plain {{
                color: #800000; /* Red color for literals */
            }}
        </style>
        <h3><b>Config</b></h3>
        <pre class="highlight">{highlighted_config_yaml}</pre>
        """

        if full_config is not None:
            full_config_yaml = OmegaConf.to_yaml(full_config)
            full_config_yaml = "\n".join([f"\n{line}" if not line.startswith(" ") else line for line in full_config_yaml.splitlines()])
            highlighted_full_config_yaml = highlight(full_config_yaml, YamlLexer(), formatter)

            html += f"""
            <details>
                <summary><b>Full Config (click to expand/collapse)</b></summary>
                <pre class="highlight">{highlighted_full_config_yaml}</pre>
            </details>"""

        return html

    def make_header(self, run):
        name = run.name
        description = run.description

        # remove the first line of the description as it is the title
        description = description[description.find("\n") + 1:]

        header_html = f"""
        <h2>{name}</h2>
        <div>{markdown.markdown(description)}</div>
        """

        return header_html

    def get_run(self, run_id):
        return self.api.run(f"{self.entity}/{self.project}/{run_id}")

    def make_report(self, run_id):
        run = self.get_run(run_id)

        html = ""
        html += self.make_header(run)
        html += self.make_summary(run)
        html += self.make_config(run)

        return html

    def display_report(self, run_id):
        html = self.make_report(run_id)
        display(HTML(html))

        # print(run.config)

#-------------------------------------------------------------------------------
# Tables
#-------------------------------------------------------------------------------

import inspect
from models import (
    Retrieval_EmbedModel, MultiProjectModel,
    RetrievalLammaTransformer, Retrieval_Project
)

def get_default_values(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}

def get_init_default(default_value):
    if not default_value is None and not isinstance(default_value, str):
        default_value = default_value.__name__

    return default_value

def is_init_orthogonal(init):
    if init is None:
        return False
    
    if not isinstance(init, str):
        init = init.__name__

    if isinstance(init, str) and "orthogonal_" in init:
        return True

    raise ValueError(f"Unknown linear init '{init}'")

def get_tie_info(config):
    model = config['model']
    info = {}

    if model['_target_'] == "models.Retrieval_EmbedModel":
        default_values = get_default_values(Retrieval_EmbedModel)

        info['tied'] = config['model'].get('tied', default_values['tied'])
        info['tied_init'] = config['model'].get('tied_init', default_values['tied_init'])

        make_post_embed_model = config['model']['make_post_embed_model']
        make_article_embed_model = config['model'].get('make_article_embed_model', default_values['make_article_embed_model'])

        if make_article_embed_model is not None:
            if make_post_embed_model != make_article_embed_model:
                raise ValueError("Model mismatch")
            
        if not make_post_embed_model['_target_'] == 'models.MultiProjectModel':
            raise ValueError("Unknown model")
        
        default_values = get_default_values(MultiProjectModel)

        modality_tied = make_post_embed_model.get('modality_tied', default_values['modality_tied'])
        modality_tied_init = make_post_embed_model.get('modality_tied_init', default_values['modality_tied_init'])       
        linear_init = make_post_embed_model.get('modality_tied_init', default_values['linear_init'])
        
        info['modality_tied'] = modality_tied
        info['modality_tied_init'] = modality_tied_init
        info['orthogonal_init'] = is_init_orthogonal(linear_init)

    elif model['_target_'] == "models.RetrievalLammaTransformer":
        default_values = get_default_values(RetrievalLammaTransformer)

        tied = config['model'].get('tied', default_values['tied'])
        tied_init = config['model'].get('tied_init', default_values['tied_init'])
        tied_projections = config['model'].get('tied_projections', default_values['tied_projections'])
        tied_init_projections = config['model'].get('tied_init_projections', default_values['tied_init_projections'])
        modality_tied_init_projections = config['model'].get('modality_tied_init_projections', default_values['modality_tied_init_projections'])
        linear_init_projections = config['model'].get('linear_init_projections', default_values['linear_init_projections'])
        linear_init_transformer = config['model'].get('linear_init_transformer', default_values['linear_init_transformer'])
        
        info['tied'] = tied
        info['tied_init'] = tied_init

        info['tied_projections'] = tied_projections
        info['tied_init_projections'] = tied_init_projections

        info['modality_tied'] = False
        info['modality_tied_init'] = modality_tied_init_projections

        if linear_init_projections != linear_init_transformer:
            info['orthogonal_init'] = None
        else:
            info['orthogonal_init'] = is_init_orthogonal(linear_init_projections)
        
    elif model['_target_'] == "models.Retrieval_Project":
        default_values = get_default_values(Retrieval_Project)

        info['tied'] = config['model'].get('tied', default_values['tied'])
        info['tied_init'] = config['model'].get('tied_init', default_values['tied_init'])

        linear_init = config['model'].get('linear_init', default_values['linear_init'])
        info['orthogonal_init'] = is_init_orthogonal(linear_init)

    elif model['_target_'] == "models.Retrieval_Concat":
        pass

    else:
        raise ValueError(f"Unknown model {model['_target_']}")

    return info

def get_config_essentials(config):
    essentials = get_tie_info(config)
    essentials["language"] = "–"  
    
    assert config.modalities.articles == config.modalities.posts
    
    if hasattr(config.modalities.posts, "image"):
        essentials['image_embedder'] = config.modalities.posts.image.embedder

    else:
        essentials['image_embedder'] = "–"

    if hasattr(config.modalities.posts, "text"):
        essentials['text_embedder'] = config.modalities.posts.text.embedder
        language = config.modalities.posts.text.language
        if not essentials["language"] == "–" and language != essentials["language"]:
            raise ValueError("Language mismatch")
        essentials["language"] = language
    
    else:
        essentials['text_embedder'] = "–"

    if hasattr(config.modalities.posts, "ocr"):
        essentials['ocr_embedder'] = config.modalities.posts.ocr.embedder
        language = config.modalities.posts.ocr.language
        if not essentials["language"] == "–" and language != essentials["language"]:
            raise ValueError("Language mismatch")
        essentials["language"] = language

    else:
        essentials['ocr_embedder'] = "–"

    return essentials

def make_essentials_table(
    inspector, experiment_names,
    metrics=["test/recall_1", "test/recall_5", "test/recall_10"]
):
    data = []

    for exp_name in experiment_names:
        runs = list(inspector.get_runs(exp_name))
        config = inspector.make_config(runs[0], diff_only=False)
        config['args'].pop('seed', None)
        config.pop('seed', None)
        config['run_args'].pop('experiment_name', None)

        essentials = get_config_essentials(config)
        metrics = inspector.get_metrics(exp_name, metrics=metrics)

        for run in runs[1:]:
            run_config = inspector.make_config(run, diff_only=False)
            run_config['args'].pop('seed', None)
            run_config.pop('seed', None)
            run_config['run_args'].pop('experiment_name', None)

            if config != run_config:
                print(f"Config mismatch for {exp_name}!")

        data.append({
            "experiment": exp_name,
            **metrics,
            **essentials
        })

    df = pd.DataFrame(data)
    df = df.where(pd.notnull(df), None)

    return df