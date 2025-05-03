
import torch
import torch.utils.checkpoint
from torch import nn
from collections import defaultdict
from itertools import product

#-------------------------------------------------------------------------------
# Modality specs
#-------------------------------------------------------------------------------

class ModalitySpec:
    def __init__(self,
        modality: str, 
        shape: tuple[None|int],
    ):
        """
        modality (str): Type of modality; one of the modalities registered in
            the embeds.modality_registry.
        shape (tuple): Shape of this modality's tensors; the batch dimension
            is not included (a batch-first format is assumed). Some dimensions
            may be None, indicating that the tensor can have variable size in
            that dimension.
        """
        self.modality = modality
        self.shape = shape

    def __repr__(self):
        return f"ModalitySpec({self.modality}, {self.shape})"
    
    def __str__(self):
        return f"{self.modality} {self.shape}"
    
    def __eq__(self, other_spec):
        return self.modality == other_spec.modality and self.shape == other_spec.shape

class modspec_asserts:
    @staticmethod
    def all_equal_shape(specs: dict[str, ModalitySpec]):
        all_shapes = [spec.shape for spec in specs.values()]

        for shape in all_shapes[1:]:
            if shape != all_shapes[0]:
                raise ValueError("All modality shapes must be equal.")
            
        return all_shapes[0]
    
    @staticmethod
    def equal_shape(spec1: ModalitySpec, spec2: ModalitySpec):
        if spec1.shape != spec2.shape:
            raise ValueError("Modality shapes must be equal.")
        
        return spec1.shape

    @staticmethod
    def all_scalar(specs: dict[str, ModalitySpec]):
        for spec in specs.values():
            if not isinstance(spec.shape, int):
                raise ValueError("Only scalar modality shapes are supported.")

    @staticmethod
    def equal(
        specs1: dict[str, ModalitySpec],
        specs2: dict[str, ModalitySpec],
        normalize=False
    ):
        """
        Raises if the two specs are not equal.
        
        If normalize is True, the specs2 dictionary is reordered so that
        its keys come in the same order as the keys in specs1 and both
        specs are returned."""

        if not specs1 == specs2:
            raise ValueError("Modality specs must be equal.")
        
        if normalize:
            specs2 = {key: specs2[key] for key in specs1.keys()}

        return specs1, specs2
        
    @staticmethod
    def auto_emb_dim(
        emb_dim,
        specs1: dict[str, ModalitySpec],
        specs2: dict[str, ModalitySpec] = None
    ):
        if emb_dim is None:
            emb_dim = modspec_asserts.all_equal_shape(specs1)

            if not specs2 is None and emb_dim != modspec_asserts.all_equal_shape(specs2):
                raise ValueError("Post and article embeddings must have the same dimensionality when emb_dim is None.")
            
        return emb_dim

#-------------------------------------------------------------------------------
# Base retrieval model class
#-------------------------------------------------------------------------------

class RetrievalModel(nn.Module):
    """
    When instantiated, models are provided with a specification of the
    modalities that they will be expected to handle; they are provided
    in a dictionary keyed by the modality name, with values being
    ModalitySpec objects describing the modality.

    - post_modality_specs: dict[str, ModalitySpec]
    - article_modality_specs: dict[str, ModalitySpec]

    Each derived class is responsible for setting the architecture so
    that it can handle the modalities or raise a ValueError if some of
    the modality specifications are not supported by the model.
    """

    def embed_post(self, post_tensors: dict[str, torch.Tensor]):
        """
        Takes in a batch of each modality for posts and returns a batch of
        post embedding vectors.
        """
        raise NotImplementedError

    def embed_article(self, article_tensors: dict[str, torch.Tensor]):
        """
        Takes in a batch of each modality for articles and returns a batch of
        article embedding vectors.
        """
        raise NotImplementedError

    def forward(self,
        post_tensors: dict[str, torch.Tensor],
        article_tensors: dict[str, torch.Tensor],
    ):
        post_y = self.embed_post(post_tensors)
        article_y = self.embed_article(article_tensors)
        return post_y, article_y

#-------------------------------------------------------------------------------
# MLP + Projection Retrievers
#-------------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self,
        input_size,
        output_size,
        hidden_sizes=[1024],
        activation=nn.SiLU
    ):
        super().__init__()

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_size),
                nn.Linear(in_size, out_size),
                activation()
            ) for in_size, out_size in zip(layer_sizes[:-2], layer_sizes[1:-1])
        ])

        self.output_layer = nn.Sequential(
            nn.LayerNorm(layer_sizes[-2]),
            nn.Linear(hidden_sizes[-2], layer_sizes[-1])
        )

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        return self.output_layer(x)

class Retrieval_MLP(RetrievalModel):
    def __init__(
        self,
        post_modality_specs: dict[str, ModalitySpec],
        article_modality_specs: dict[str, ModalitySpec],
        emb_dim=None,
        fusion_method="concat",
        make_mlp=MLP,
        tied=False,
        tied_init=False,
        linear_init=None
    ):
        super().__init__()

        self.emb_dim = modspec_asserts.auto_emb_dim(
            emb_dim, post_modality_specs, article_modality_specs
        )

        self.fusion_method = fusion_method
        self.post_keys = list(post_modality_specs.keys())
        self.article_keys = list(article_modality_specs.keys())

        modspec_asserts.all_scalar(post_modality_specs)
        modspec_asserts.all_scalar(article_modality_specs)
        
        if tied or tied_init:
            modspec_asserts.equal(post_modality_specs, article_modality_specs)
            self.article_keys = self.post_keys # ensure that the keys are in the same order

        self.post_emb_dim = None
        self.article_emb_dim = None

        if fusion_method == "addition":
            self.post_emb_dim = modspec_asserts.all_equal_shape(post_modality_specs)
            self.article_emb_dim = modspec_asserts.all_equal_shape(article_modality_specs)

        elif fusion_method == "concat":
            self.post_emb_dim = 0
            for spec in post_modality_specs.values():
                self.post_emb_dim += spec.shape

            self.article_emb_dim = 0
            for spec in article_modality_specs.values():
                self.article_emb_dim += spec.shape

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        if tied:
            self.post_mlp = make_mlp(self.post_emb_dim, self.emb_dim)
            
            if linear_init is not None:
                self._apply_init(self.post_mlp, linear_init)
            
            self.article_mlp = self.post_mlp
        else:
            self.post_mlp = make_mlp(self.post_emb_dim, self.emb_dim)
            self.article_mlp = make_mlp(self.article_emb_dim, self.emb_dim)
            
            if linear_init is not None:
                self._apply_init(self.post_mlp, linear_init)
                self._apply_init(self.article_mlp, linear_init)
            
            if tied_init:
                self.article_mlp.load_state_dict(self.post_mlp.state_dict())

    def _apply_init(self, model, linear_init):
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                linear_init(layer.weight)

    def make_features(self, tensors, keys):
        if self.fusion_method == "concat":
            features = torch.cat([tensors[key] for key in keys], axis=1)
        elif self.fusion_method == "addition":
            features = torch.sum(torch.stack([tensors[key] for key in keys]), axis=0)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return features

    def embed_post(self, tensors):
        features = self.make_features(tensors, self.post_keys)
        return self.post_mlp(features.float())

    def embed_article(self, tensors):
        features = self.make_features(tensors, self.article_keys)
        return self.article_mlp(features.float())

class Retrieval_Project(Retrieval_MLP):
    """
    This is a special case of the MLP model where the model only applies a
    linear projection to the input features.
    """
    def __init__(
        self,
        post_modality_specs: dict[str, ModalitySpec],
        article_modality_specs: dict[str, ModalitySpec],
        emb_dim=None,
        fusion_method="concat",
        tied=False,
        tied_init=False,
        linear_init=torch.nn.init.orthogonal_,
        layer_norm=None,
        layer_norm_post=None
    ):
        def make_mlp(in_dim, out_dim):
            layers = []

            if not layer_norm is None:
                layers.append(layer_norm(in_dim))

            layers.append(nn.Linear(in_dim, out_dim))

            if not layer_norm_post is None:
                layers.append(layer_norm_post(out_dim))

            return nn.Sequential(*layers)
        
        super().__init__(
            post_modality_specs=post_modality_specs,
            article_modality_specs=article_modality_specs,
            emb_dim=emb_dim,
            fusion_method=fusion_method,
            make_mlp=make_mlp,
            tied=tied,
            tied_init=tied_init,
            linear_init=linear_init
        )

#-------------------------------------------------------------------------------
# architectures: LLaMA-inspired transformer with bidirectional attention
#-------------------------------------------------------------------------------

try:
    from llama_transformer import MMModel, MMConfig
    from llama_transformer import RMSNorm # <- do not remove; for use in the config

    class RetrievalLammaTransformer(RetrievalModel):
        def __init__(
            self,
            config: MMConfig,
            post_modality_specs: dict[str, ModalitySpec],
            article_modality_specs: dict[str, ModalitySpec],
            output_token: int|list[int]|tuple[int]|slice = slice(None),
            add_cls_token: bool = False,
            tied: bool = True,
            tied_init: bool = False,
            tied_projections: bool = False,
            tied_init_projections: bool = True,
            modality_tied_init_projections: bool|dict[str, str] = True,
            post_modality_tied_init_projections: bool|dict[str, str]|None = None,
            article_modality_tied_init_projections: bool|dict[str, str]|None = None,
            linear_init_projections=torch.nn.init.orthogonal_,
            linear_init_transformer=torch.nn.init.orthogonal_
        ):
            """
            Args:
                modality_tied_init_projections (bool | dict[str, str], optional):
                    If a bool, this controls whether linear projections are initialized
                    as tied across all modalities. If a dict, it specifies which
                    modality is tied to which; the keys are the modality names and the
                    values are the names of the modality to which they are tied.
                    Defaults to True.
            """

            super().__init__()

            modspec_asserts.all_scalar(post_modality_specs)
            modspec_asserts.all_scalar(article_modality_specs)

            self.post_keys = list(post_modality_specs.keys())
            self.article_keys = list(article_modality_specs.keys())

            if tied_projections or tied_init_projections:
                modspec_asserts.equal(post_modality_specs, article_modality_specs)
                self.article_keys = self.post_keys # ensure that the keys are in the same order

            # if modality tied init is not specified for post/articles separately,
            # we copy the setting from the common modality_tied_init_projections
            if post_modality_tied_init_projections is None:
                post_modality_tied_init_projections = modality_tied_init_projections

            if article_modality_tied_init_projections is None:
                article_modality_tied_init_projections = modality_tied_init_projections

            # normalization of post_modality_tied_init_projections and article_modality_tied_init_projections
            if isinstance(post_modality_tied_init_projections, bool):
                if post_modality_tied_init_projections:
                    # init of all modalities is tied to the first one
                    key_iter = iter(post_modality_specs.keys())
                    first_modality = next(key_iter)
                    post_modality_tied_init_projections = {key: first_modality for key in key_iter}
                else:
                    # no modalities have tied init
                    post_modality_tied_init_projections = {}

            if isinstance(article_modality_tied_init_projections, bool):
                if article_modality_tied_init_projections:
                    # init of all modalities is tied to the first one
                    key_iter = iter(article_modality_specs.keys())
                    first_modality = next(key_iter)
                    article_modality_tied_init_projections = {key: first_modality for key in key_iter}
                else:
                    # no modalities have tied init
                    article_modality_tied_init_projections = {}

            # sanity checks for modality tied init
            for k, v in post_modality_tied_init_projections.items():
                modspec_asserts.equal_shape(post_modality_specs[k], post_modality_specs[v])

            for k, v in article_modality_tied_init_projections.items():
                modspec_asserts.equal_shape(article_modality_specs[k], article_modality_specs[v])             

            # create and initialize the modules
            if tied_projections:
                self.proj_post = nn.ModuleDict({
                    key: nn.Linear(spec.shape, config.hidden_size[0])
                        for key, spec in post_modality_specs.items()
                })

                if linear_init_projections is not None:
                    for key in self.post_keys:
                        linear_init_projections(self.proj_post[key].weight)
            
                self.proj_article = self.proj_post

            else:
                self.proj_post = nn.ModuleDict({
                    key: nn.Linear(spec.shape, config.hidden_size[0])
                        for key, spec in post_modality_specs.items()
                })

                self.proj_article = nn.ModuleDict({
                    key: nn.Linear(spec.shape, config.hidden_size[0])
                        for key, spec in article_modality_specs.items()
                })

                if linear_init_projections is not None:
                    for key in self.post_keys:
                        linear_init_projections(self.proj_post[key].weight)

                    for key in self.article_keys:
                        linear_init_projections(self.proj_article[key].weight)

                if tied_init_projections:
                    for key in self.post_keys:
                        self.proj_article[key].load_state_dict(self.proj_post[key].state_dict())

            # apply modality tied init for projections (if any)
            for key, tied_key in post_modality_tied_init_projections.items():
                self.proj_post[key].load_state_dict(self.proj_post[tied_key].state_dict())

            for key, tied_key in article_modality_tied_init_projections.items():
                self.proj_article[key].load_state_dict(self.proj_article[tied_key].state_dict())

            if tied:
                self.transformer_post = MMModel(config)
                self.transformer_article = self.transformer_post

                if linear_init_transformer is not None:
                    self._apply_transformer_init(
                        self.transformer_post, linear_init_transformer
                    )
            else:
                self.transformer_post = MMModel(config)
                self.transformer_article = MMModel(config)

                if linear_init_transformer is not None:
                    self._apply_transformer_init(
                        self.transformer_post, linear_init_transformer
                    )
                    self._apply_transformer_init(
                        self.transformer_article, linear_init_transformer
                    )

                if tied_init:
                    self.transformer_article.load_state_dict(self.transformer_post.state_dict())
                    
            if isinstance(output_token, tuple) or isinstance(output_token, list):
                output_token = slice(*output_token)
            
            self.output_token = output_token

            if add_cls_token:
                self.cls_token = nn.Parameter(
                    torch.randn(config.hidden_size[0]) * config.initializer_range,
                    requires_grad=True
                )
            else:
                self.cls_token = None

            if config.output_dropout > 0.0:
                self.output_dropout = nn.Dropout(config.output_dropout)
            else:
                self.output_dropout = None

        def _apply_transformer_init(self, transformer, linear_init):
            for layer in transformer.modules():
                if isinstance(layer, nn.Linear):
                    linear_init(layer.weight)

        def embed_post(self, post_tensors: dict[str, torch.Tensor]):
            features = []

            if not self.cls_token is None:
                batch_size = next(iter(post_tensors.values())).shape[0]
                features.append(self.cls_token.expand(batch_size, -1))

            for key in self.post_keys:
                features.append(self.proj_post[key](post_tensors[key]))

            features = torch.stack(features, dim=1)

            y = self.transformer_post(features, return_dict=False)
            assert len(y) == 1

            y = y[0]
            y = y[:, self.output_token, :]

            if not self.output_dropout is None:
                y = self.output_dropout(y)

            return torch.flatten(y, start_dim=1)

        def embed_article(self, article_tensors: dict[str, torch.Tensor]):
            features = []

            if not self.cls_token is None:
                batch_size = next(iter(article_tensors.values())).shape[0]
                features.append(self.cls_token.expand(batch_size, -1))

            for key in self.article_keys:
                features.append(self.proj_article[key](article_tensors[key]))

            features = torch.stack(features, dim=1)
            
            y = self.transformer_article(features, return_dict=False)
            assert len(y) == 1
            
            y = y[0]
            y = y[:, self.output_token, :]

            if not self.output_dropout is None:
                y = self.output_dropout(y)

            return torch.flatten(y, start_dim=1)

    def make_llama_config(
        hidden_size: int|list[int],
        intermediate_size: int|list[int] = 256,
        num_hidden_layers: int|None = 5,
        num_attention_heads: int|list[int] = 8,
        num_key_value_heads: int|list[int]|None = None,
        hidden_act: str|list[str] = "silu",
        attention_bias: bool|list[bool] = False,
        attention_dropout: float|list[float] = 0.0,
        mlp_bias: bool|list[bool] = False,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings=32,
        pretraining_tp=1,
        rope_theta=10000.0,
        rope_scaling=None,
        use_position_embeds=True,
        mlp_type='mlp',
        gating_weight: float|None = None,
        output_dropout: float = 0.0,
        **kwargs,
    ):
        return MMConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            mlp_bias=mlp_bias,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            max_position_embeddings=max_position_embeddings,
            pretraining_tp=pretraining_tp,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_position_embeds=use_position_embeds,
            mlp_type=mlp_type,
            gating_weight=gating_weight,
            output_dropout=output_dropout,
            **kwargs
        )

except ModuleNotFoundError as err:
    print("LLaMA transformer not available:", err)

#-------------------------------------------------------------------------------
# architectures: retrieval concat
#-------------------------------------------------------------------------------

class Retrieval_Concat(RetrievalModel):
    def __init__(self,
        post_modality_specs: dict[str, ModalitySpec],
        article_modality_specs: dict[str, ModalitySpec]
    ):
        super().__init__()

        modspec_asserts.equal(post_modality_specs, article_modality_specs)
        self.keys = list(post_modality_specs.keys())

    def embed_post(self, post_tensors: dict[str, torch.Tensor]):
        return torch.cat([post_tensors[k] for k in self.keys], dim=1)
        
    def embed_article(self, article_tensors: dict[str, torch.Tensor]):
        return torch.cat([article_tensors[k] for k in self.keys], dim=1)

#-------------------------------------------------------------------------------
# architectures: multi-project
#-------------------------------------------------------------------------------

class MultiProjectBlock(nn.Module):
    def __init__(self,
        input_dim,
        output_dim,
        activation=nn.SiLU,
        linear_init=nn.init.orthogonal_,
        residual=False
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        linear_init(self.linear.weight)
        self.activation = activation()
        self.residual = residual if input_dim == output_dim else False

    def forward(self, x):
        if self.residual:
            return x + self.activation(self.linear(x))
        else:
            return self.activation(self.linear(x))

class MultiProjectModel(nn.Module):
    def __init__(
        self,
        modality_specs,
        pre_fusion_layers,
        post_fusion_layers,
        output_dim=None,
        activation=nn.SiLU,
        last_activation=nn.Identity,
        modality_tied_init=True,
        modality_tied=False,
        linear_init=nn.init.orthogonal_,
        fusion='addition', # or 'concat'
        residual=True
    ):
        """
        This class implements a retrieval model that first applies layers
        to each of the three modalities separately, then sums the results
        and applies another set of layers to the resulting representation.

        Args:
            pre_fusion_layers: list of integers specifying the dimensions of
                           the layers before the three modalities are fused.

            post_fusion_layers: list of integers specifying the dimensions of
                            the layers after the three modalities are fused.
        """
        super().__init__()

        modspec_asserts.all_scalar(modality_specs)

        output_dim = modspec_asserts.auto_emb_dim(
            output_dim, modality_specs
        )

        if modality_tied or modality_tied_init:
            spec_shape = modspec_asserts.all_equal_shape(modality_specs)

        # pre-fusion
        self.pre_fusion = {}

        if modality_tied:
            layers = []
            _pre_fusion_layers = [spec_shape] + pre_fusion_layers
            
            for in_dim, out_dim in zip(
                _pre_fusion_layers[:-1],
                _pre_fusion_layers[1:]
            ):
                layers.append(MultiProjectBlock(in_dim, out_dim, activation, linear_init, residual))

            layers = nn.Sequential(*layers)
            self.pre_fusion = {k: layers for k in modality_specs.keys()}

        else:
            for k, spec in modality_specs.items():
                layers = []
                _pre_fusion_layers = [spec.shape] + pre_fusion_layers

                for in_dim, out_dim in zip(
                    _pre_fusion_layers[:-1],
                    _pre_fusion_layers[1:]
                ):
                    layers.append(MultiProjectBlock(in_dim, out_dim, activation, linear_init, residual))

                self.pre_fusion[k] = nn.Sequential(*layers)

            if modality_tied_init:
                spec_keys = list(modality_specs.keys())
                sd = self.pre_fusion[spec_keys[0]].state_dict()

                for k in spec_keys[1:]:
                    self.pre_fusion[k].load_state_dict(sd)

        pre_fusion_layers = _pre_fusion_layers # for the case with just the input layer
        self.pre_fusion = nn.ModuleDict(self.pre_fusion)

        # fusion
        self.fusion = fusion
    
        # post-fusion
        post_fusion_layers = [pre_fusion_layers[-1]] + list(post_fusion_layers) + [output_dim]
        layers = []

        for in_dim, out_dim in zip(
            post_fusion_layers[:-1],
            post_fusion_layers[1:]
        ):
            layers.append(MultiProjectBlock(in_dim, out_dim, activation, linear_init, residual))

        layers[-1] = last_activation()
        self.post_fusion = nn.Sequential(*layers)
        
    def forward(self, tensors: dict[str, torch.Tensor]):
        modalities = [
            pf(tensors[k]) for k, pf in self.pre_fusion.items()
        ]
       
        if self.fusion == 'addition':
            x = torch.sum(torch.stack(modalities), dim=0)
        elif self.fusion == 'concat':
            x = torch.flatten(torch.stack(modalities), start_dim=1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion}")

        return self.post_fusion(x)
    
class Retrieval_EmbedModel(RetrievalModel):
    def __init__(
        self,
        post_modality_specs: dict[str, ModalitySpec],
        article_modality_specs: dict[str, ModalitySpec],
        make_post_embed_model,
        make_article_embed_model=None,
        tied=False,
        tied_init=True
    ):
        super().__init__()

        if make_article_embed_model is None:
            make_article_embed_model = make_post_embed_model

        if tied or tied_init:
            post_modality_specs, article_modality_specs = modspec_asserts.equal(
                post_modality_specs, article_modality_specs, normalize=True
            )

        if tied:
            self.post_embed_model = make_post_embed_model(modality_specs=post_modality_specs)
            self.article_embed_model = self.post_embed_model
        else:
            self.post_embed_model = make_post_embed_model(modality_specs=post_modality_specs)
            self.article_embed_model = make_article_embed_model(modality_specs=article_modality_specs)

            if tied_init:
                self.article_embed_model.load_state_dict(
                    self.post_embed_model.state_dict()
                )

    def embed_post(self, post_tensors):
        return self.post_embed_model(post_tensors)

    def embed_article(self, article_tensors):
        return self.article_embed_model(article_tensors)
