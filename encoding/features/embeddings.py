# encoding/features/static_token_extractor.py
from __future__ import annotations
from typing import Any, Dict, List, Union, Optional, Iterable, TYPE_CHECKING
import os
import re

from .base import BaseFeatureExtractor

if TYPE_CHECKING:
    import numpy as np
    from gensim.models import KeyedVectors

try:
    from tqdm.auto import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


class StaticEmbeddingFeatureExtractor(BaseFeatureExtractor):
    """
    Local-only static *token* embedding extractor (Word2Vec / GloVe).

    Input  (extract_features):
        - List[str]: list of tokens/words (preferred), order preserved
        - str: a raw string (will be tokenized using `tokenizer_pattern`)

    Output:
        - np.ndarray with shape [N, D], one row per input token.

    Config (Dict[str, Any]):
      - vector_path (str, required): local vectors path. Supported:
            *.kv                -> KeyedVectors.load (mmap capable)
            *.bin / *.bin.gz    -> word2vec binary (binary=True)
            *.w2v.txt           -> word2vec text WITH header (binary=False, no_header=False)
            *.txt / *.txt.gz    -> GloVe text WITHOUT header (binary=False, no_header=True)
      - lowercase (bool): lowercase tokens before lookup
            (GoogleNews: False; GloVe/Wiki-Giga: True)  [default: True]
      - oov_handling (str): one of:
            "copy_prev"  -> OOV copies the previous valid embedding (DEFAULT)
            "zero"       -> OOV becomes a zero vector (length preserved)
            "skip"       -> OOV is dropped (length may shrink)
            "error"      -> raise on first OOV
      - use_tqdm (bool): show progress bar for long inputs [default: True]
      - mmap (bool): memory-map .kv [default: True]
      - binary (Optional[bool]): force word2vec binary flag; auto-infer if None
      - no_header (Optional[bool]): force GloVe no-header; auto-infer if None
      - l2_normalize_tokens (bool): L2-normalize each token vector [default: False]
      - tokenizer_pattern (str): ONLY used if input is a single string.
            Default r"[A-Za-z0-9_']+" (keeps underscores)

        Note: This has also been tested with ENG1000. You just have to convert it to the .kv format first. We'll provide a script to do that!
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # ---- Required path
        vector_path = config.get("vector_path", "")
        if not vector_path:
            raise ValueError("'vector_path' is required.")
        # expanduser + abspath for clearer logs
        self.vector_path: str = os.path.abspath(os.path.expanduser(vector_path))
        if not os.path.exists(self.vector_path):
            raise FileNotFoundError(f"Vector file not found: {self.vector_path}")

        self.lowercase: bool = bool(config.get("lowercase", True))
        self.oov_handling: str = config.get("oov_handling", "copy_prev")
        if self.oov_handling not in {"copy_prev", "zero", "skip", "error"}:
            raise ValueError(
                "oov_handling must be 'copy_prev', 'zero', 'skip', or 'error'"
            )
        self.use_tqdm: bool = bool(config.get("use_tqdm", True))
        self.mmap: bool = bool(config.get("mmap", True))
        self.l2_normalize_tokens: bool = bool(config.get("l2_normalize_tokens", False))
        self.tokenizer_pattern: str = config.get("tokenizer_pattern", r"[A-Za-z0-9_']+")

        self._force_binary: Optional[bool] = config.get("binary", None)
        self._force_no_header: Optional[bool] = config.get("no_header", None)

        try:
            import torch
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"

        self._tok_re = re.compile(self.tokenizer_pattern)

        print(f"[StaticToken] Loading vectors: {self.vector_path}")
        self.kv = self._load_local_vectors(self.vector_path)
        self.dim = int(self.kv.vector_size)
        print(
            f"[StaticToken] Loaded ({self.dim}-D), vocab={len(self.kv.key_to_index):,}"
        )

    def extract_features(
        self,
        stimuli: Union[str, List[str]],
        **kwargs,
    ) -> np.ndarray:
        """
        Tokens -> [N, D], one row per input token. If `stimuli` is a string, it is tokenized.
        OOV handling per config (default: copy previous valid embedding).
        """
        import numpy as np

        # Normalize input to a token list
        if isinstance(stimuli, str):
            text = stimuli.lower() if self.lowercase else stimuli
            tokens = self._tok_re.findall(text)
        elif isinstance(stimuli, list):
            tokens = []
            for t in stimuli:
                if isinstance(t, str):
                    tokens.append(t.lower() if self.lowercase else t)
                else:
                    tokens.append(t)  # will be handled below
        else:
            raise TypeError(
                "extract_features expects a List[str] of tokens or a single string."
            )

        N = len(tokens)
        if N == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        iterator: Iterable[str] = (
            tqdm(tokens, desc="Embedding tokens", total=N) if self.use_tqdm else tokens
        )

        vecs: List[np.ndarray] = []
        last_valid: Optional[np.ndarray] = None  # for copy_prev

        for i, tok in enumerate(iterator):
            v: Optional[np.ndarray] = None

            if not isinstance(tok, str):
                if self.oov_handling == "error":
                    raise ValueError(f"Non-string token at index {i}: {tok!r}")
                elif self.oov_handling == "skip":
                    # Skip may shrink length
                    continue
                elif self.oov_handling == "copy_prev":
                    v = (
                        last_valid.copy()
                        if last_valid is not None
                        else np.zeros((self.dim,), dtype=np.float32)
                    )
                else:  # "zero"
                    v = np.zeros((self.dim,), dtype=np.float32)
            else:
                # String token: lookup
                if tok in self.kv.key_to_index:
                    v = self.kv.get_vector(tok).astype(np.float32, copy=False)
                    # only update last_valid when we have a real vector
                    last_valid = v.copy()
                else:
                    # OOV handling
                    if self.oov_handling == "error":
                        raise KeyError(f"OOV token at index {i}: {tok!r}")
                    elif self.oov_handling == "skip":
                        continue  # WARNING: length may shrink
                    elif self.oov_handling == "copy_prev":
                        v = (
                            last_valid.copy()
                            if last_valid is not None
                            else np.zeros((self.dim,), dtype=np.float32)
                        )
                    else:  # "zero"
                        v = np.zeros((self.dim,), dtype=np.float32)

            # Optional per-token L2 norm
            if self.l2_normalize_tokens:
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n

            vecs.append(v)

        if not vecs:
            return np.zeros((0, self.dim), dtype=np.float32)

        return np.stack([np.asarray(v, dtype=np.float32) for v in vecs], axis=0)

    def _load_local_vectors(self, path: str) -> KeyedVectors:
        from gensim.models import KeyedVectors
        ext = path.lower()

        if ext.endswith(".kv"):
            return KeyedVectors.load(path, mmap="r" if self.mmap else None)

        binary = (
            self._infer_binary(ext)
            if self._force_binary is None
            else bool(self._force_binary)
        )
        no_header = (
            self._infer_no_header(ext)
            if self._force_no_header is None
            else bool(self._force_no_header)
        )

        try:
            return KeyedVectors.load_word2vec_format(
                path, binary=binary, no_header=no_header
            )
        except Exception as e:
            # If *.txt mis-detected, flip no_header once and retry
            if ext.endswith(".txt") or ext.endswith(".txt.gz"):
                try:
                    return KeyedVectors.load_word2vec_format(
                        path, binary=False, no_header=not no_header
                    )
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to load vectors from {path}.\n"
                        f"Attempt1 (binary={binary}, no_header={no_header}) -> {e}\n"
                        f"Attempt2 (binary=False, no_header={not no_header}) -> {e2}\n"
                        "If this is raw GloVe, use no_header=True. If word2vec text, it must have a header."
                    )
            raise

    @staticmethod
    def _infer_binary(ext: str) -> bool:
        return ext.endswith(".bin") or ext.endswith(".bin.gz")

    @staticmethod
    def _infer_no_header(ext: str) -> bool:
        # Heuristics:
        #   *.w2v.txt      -> word2vec text WITH header => no_header=False
        #   *.txt/.txt.gz  -> assume GloVe text WITHOUT header => no_header=True
        # (binaries ignore no_header)
        if ext.endswith(".w2v.txt"):
            return False
        if ext.endswith(".txt") or ext.endswith(".txt.gz"):
            return True
        return False
