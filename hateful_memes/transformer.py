import torch
import os
import pickle
import abc
import torch.nn as nn
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device


class TextMasking(torch.nn.Module):
    """
        Randomly mask input tokens using a special `mask` token.
    """
    def __init__(self, mask_prob: float, mask_token_id: int, mask_ignored_ids: Optional[List[int]] = None) -> None:
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.mask_ignored_ids = mask_ignored_ids or [] # ignore these tokens for masking

    def _init_full_mask(self, seq: torch.Tensor) -> torch.Tensor:
        # Returns `True` for tokens to not ignore in `seq`
        full_mask = torch.full_like(seq, True, dtype=torch.bool)
        for ignored_id in self.mask_ignored_ids:
            full_mask &= (seq != ignored_id)
        return full_mask

    def _get_mask_subset_with_prob(self, mask: torch.Tensor) -> torch.Tensor:
        # Returns a subset of input `mask`
        random_mask = torch.rand(mask.shape, device=mask.device) < self.mask_prob
        mask &= random_mask
        return mask

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mask_prob == 0:
            return seq
        else:
            mask = self._init_full_mask(seq)
            mask = self._get_mask_subset_with_prob(mask)
            masked_seq = seq.clone().detach()
            masked_seq.masked_fill_(mask, self.mask_token_id)
            return masked_seq


class LanguageEncoder(nn.Module):
    """Pre-trained text model that implements a caching system for fast embedding retrieval when no
    fine-tuning is required. It currently accepts all models defined by `sentence_transformers` library.
    """

    def __init__(self, model_name: str,
                 freeze: bool = True,
                 output_value: str = 'sentence_embedding',
                 mask_prob: float = 0.0,
                 normalize_embeddings: bool = True,
                 use_dataset_cache: bool = True,
                 cache_file: Optional[str] = None):
        """

        :param model_name: Text encoder name, see https://www.sbert.net/docs/pretrained_models.html
        :param freeze: whether the text encoder is fine-tuned or not
        :param output_value:  Default "sentence_embedding", to get sentence embeddings with shape (N, E)
            where N == batch size, E == embedding dimension
            Can be set to "token_embeddings" to get wordpiece token embeddings with shape (N, L, E)
            and attention mask with shape (N, L) where N == batch size, L == # tokens, E == embedding dimension.
        :param mask_prob: probability of randomly masking input tokens with mask tokens.
        :param normalize_embeddings: whether text embeddings are l2-normalized or not
            only if output_value == "sentence_embedding"
        :param use_dataset_cache: Cache the text embeddings computed in `forward` pass.
        :param cache_file: File name of cache to dump on disk.
        """

        super().__init__()
        assert output_value in {"token_embeddings", "sentence_embedding"}

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.use_dataset_cache = use_dataset_cache
        mask_ignore_token_ids = [self.model.tokenizer.pad_token_id,
                                 self.model.tokenizer.cls_token_id,
                                 self.model.tokenizer.sep_token_id]
        mask_token_id = self.model.tokenizer.mask_token_id
        self.mask = TextMasking(mask_prob, mask_token_id, mask_ignore_token_ids)
        self.freeze = freeze
        self.output_value = output_value
        self.cache_file = cache_file or ""
        self._cache = dict()
        if self.use_dataset_cache:
            self._cache = self._load_cache()

        if freeze: # no grad computed
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: List[str]):
        if self.use_dataset_cache:
            if self.model_name in self._cache:
                embed = []
                for txt in x:
                    if txt in self._cache[self.model_name]:
                        embed.append(self._cache[self.model_name][txt])
                if len(embed) == len(x):
                    x = torch.stack(embed, dim=0).cuda()
                    if self.normalize_embeddings:
                        x = torch.nn.functional.normalize(x, p=2, dim=1)
                    return x

        features = self.model.tokenize(x) # automatically truncate too large sentences
        features["input_ids"] = self.mask(features["input_ids"])
        features = batch_to_device(features, self.model.device)
        if self.freeze:
            with torch.no_grad():
                out_features = self.model.forward(features)
                if self.output_value == "sentence_embedding":
                    embeddings = out_features["sentence_embedding"]
                    embeddings = embeddings.detach()
                elif self.output_value == "token_embeddings":
                    for name in out_features:
                        out_features[name] = out_features[name].detach()
                    assert "attention_mask" in features
                    # !! Important: 'True' should indicate NOT attended positions (torch convention in attn layers)
                    out_features["attention_mask"] = ~features["attention_mask"].bool()
                    return out_features
        else:
            out_features = self.model.forward(features)
            if self.output_value == "sentence_embedding":
                embeddings = out_features["sentence_embedding"]
            elif self.output_value == "token_embeddings":
                assert "attention_mask" in features
                # !! Important: 'True' should indicate NOT attended positions (torch convention in attn layers)
                out_features["attention_mask"] = ~features["attention_mask"].bool()
                return out_features

        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        if self.use_dataset_cache:
            if self.model_name not in self._cache:
                self._cache[self.model_name] = dict()
            for i, txt in enumerate(x):
                self._cache[self.model_name][txt] = embeddings[i].cpu().detach()
        return embeddings

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self._cache = pickle.load(f)
            except FileNotFoundError:
                pass
        return self._cache

    def dump_cache(self):
        def update(d, u):  # updated nested dict
            for k, v in u.items():
                if isinstance(v, abc.Mapping):
                    d[k] = update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    cache = pickle.load(f)
                assert isinstance(cache, dict)
                update(self._cache, cache)
            except FileNotFoundError:
                pass
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            print("Impossible to dump cache: %s" % e)
            return False
        return True
