from __future__ import annotations

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from transformers import AutoModel, AutoProcessor, AutoFeatureExtractor
import torchaudio
from tqdm import tqdm


def auto_device(fn):
    def wrapper(self, *args, **kwargs):
        with torch.no_grad():
            return fn(self, *args, **kwargs)

    return wrapper


class SpeechFeatureExtractor:
    """
    Unified feature extractor for HF speech models (Whisper encoder, HuBERT, Wav2Vec2).

    - extract_features(wav_path, layer=None) -> (features [n_chunks, D], times [n_chunks])
    - extract_all_layers(wav_path) -> (layer_to_features {idx: [n_chunks, D]}, times [n_chunks])

    Notes:
      * Pooling over encoder time can be 'last' or 'mean'.
      * For Whisper, we call the ENCODER ONLY (model.get_encoder()).
      * 'layer' indices are 0-based over encoder blocks (exclude embeddings).
    """

    def __init__(
        self,
        model_name: str,
        chunk_size: float,  # seconds between chunk starts (stride)
        context_size: float,  # seconds of audio per window (window length)
        layer: str | int = "last",  # default layer for single-layer extraction
        pool: str = "last",  # 'last' or 'mean'
        device: Optional[str] = None,
        target_sample_rate: int = 16000,
        disable_tqdm: bool = False,
    ):
        assert pool in {"last", "mean"}, "pool must be 'last' or 'mean'"
        self.model_name = model_name
        self.chunk_size = float(chunk_size)
        self.context_size = float(context_size)
        self.layer = layer
        self.pool = pool
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.target_sample_rate = int(target_sample_rate)
        self.disable_tqdm = disable_tqdm

        # Load base model
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Detect model type & set up feature extractor + forward key
        self.model_type = getattr(self.model.config, "model_type", "").lower()
        if self.model_type == "whisper":
            # Whisper expects log-mel features
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self._forward_key = "input_features"
            self._encoder = self.model.get_encoder()  # use encoder only
        else:
            # HuBERT/Wav2Vec2 expect raw PCM
            try:
                proc = AutoProcessor.from_pretrained(model_name)
                self.feature_extractor = getattr(proc, "feature_extractor", proc)
            except Exception:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    model_name
                )
            self._forward_key = "input_values"
            self._encoder = self.model  # whole model acts as encoder here

    # Helpers
    def _prepare_inputs(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        # waveform: 1D CPU torch tensor
        inputs = self.feature_extractor(
            waveform.cpu().numpy(),
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _resolve_n_layers(self, hidden_states: Tuple[torch.Tensor, ...]) -> int:
        """
        hidden_states usually length = n_layers + 1 (embeddings + each block).
        expose 0..n_layers-1 over the *blocks* (exclude embeddings).
        """
        return len(hidden_states) - 1

    def _get_layer_tensor(
        self, hidden_states: Tuple[torch.Tensor, ...], layer: str | int
    ) -> torch.Tensor:
        """
        Return hidden state for a given layer index (0-based over blocks),
        or 'last' meaning last encoder block. Shift by +1 to skip embeddings.
        """
        if layer == "last":
            return hidden_states[-1]
        idx = int(layer)
        return hidden_states[idx + 1]

    def _pool_time(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, T, D]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [1, 1, D]
        return x[0, -1, :] if self.pool == "last" else x[0].mean(dim=0)

    def _load_and_resample(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] != 1:
            wav = wav.mean(0, keepdim=True)
        if sr != self.target_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.target_sample_rate)
        return wav.squeeze(0)  # [num_samples], CPU

    # The important part for the library. We need the same interface for all models, so that everything works
    # outside the box.
    @auto_device
    def extract_features(self, wav_path: str, layer: str | int | None = None):
        """
        Single-layer extraction stacked over chunks.

        Returns:
          features: [n_chunks, D]
          times:    [n_chunks]
        """
        layer = self.layer if layer is None else layer

        wav = self._load_and_resample(wav_path)
        chunk_samples = int(self.chunk_size * self.target_sample_rate)
        context_samples = int(self.context_size * self.target_sample_rate)
        total = wav.shape[0]

        if context_samples <= 0 or chunk_samples <= 0:
            raise ValueError("context_size and chunk_size must be > 0 seconds.")
        if total < context_samples:
            return np.empty((0, 0)), np.array([])

        n_chunks = (total - context_samples) // chunk_samples + 1
        features, times = [], []

        with tqdm(
            total=int(n_chunks), desc="Extracting features", disable=self.disable_tqdm
        ) as pbar:
            for i in range(int(n_chunks)):
                end = context_samples + i * chunk_samples
                start = max(0, end - context_samples)
                if end > total:
                    break

                window = wav[start:end]
                inputs = self._prepare_inputs(window)

                outputs = self._encoder(
                    **{self._forward_key: inputs[self._forward_key]},
                    output_hidden_states=True,
                )
                hs = outputs.hidden_states  # tuple [1, T, D]

                layer_t = self._get_layer_tensor(hs, layer)

                if layer_t.shape[1] == 0:
                    pbar.update(1)
                    continue

                vec = self._pool_time(layer_t)  # [D]
                features.append(vec.detach().cpu().numpy())
                times.append(end / self.target_sample_rate)
                pbar.update(1)

        features = np.stack(features) if len(features) else np.empty((0, 0))
        times = np.array(times)
        return features, times

    @auto_device
    def extract_all_layers(self, wav_path: str):
        """
        All-layers extraction stacked over chunks.

        Returns:
          layer_to_features: {layer_idx: [n_chunks, D]}
          times:             [n_chunks]
        """
        wav = self._load_and_resample(wav_path)
        chunk_samples = int(self.chunk_size * self.target_sample_rate)
        context_samples = int(self.context_size * self.target_sample_rate)
        total = wav.shape[0]

        if context_samples <= 0 or chunk_samples <= 0:
            raise ValueError("context_size and chunk_size must be > 0 seconds.")
        if total < context_samples:
            return {}, np.array([])

        n_chunks = (total - context_samples) // chunk_samples + 1
        layer_buffers: Dict[int, list[np.ndarray]] = {}
        times: list[float] = []

        with tqdm(
            total=int(n_chunks), desc="Extracting all layers", disable=self.disable_tqdm
        ) as pbar:
            for i in range(int(n_chunks)):
                end = context_samples + i * chunk_samples
                start = max(0, end - context_samples)
                if end > total:
                    break

                window = wav[start:end]
                inputs = self._prepare_inputs(window)

                outputs = self._encoder(
                    **{self._forward_key: inputs[self._forward_key]},
                    output_hidden_states=True,
                )
                hs = outputs.hidden_states  # tuple [1, T, D]

                if hs[-1].shape[1] == 0:
                    pbar.update(1)
                    continue

                n_layers = self._resolve_n_layers(hs)
                if not layer_buffers:
                    for li in range(n_layers):
                        layer_buffers[li] = []

                for li in range(n_layers):
                    layer_t = hs[li + 1]  # skip embeddings at index 0
                    vec = self._pool_time(layer_t)
                    layer_buffers[li].append(vec.detach().cpu().numpy())

                times.append(end / self.target_sample_rate)
                pbar.update(1)

        layer_to_features = {
            li: (np.stack(buf) if len(buf) else np.empty((0, 0)))
            for li, buf in layer_buffers.items()
        }
        return layer_to_features, np.array(times)
