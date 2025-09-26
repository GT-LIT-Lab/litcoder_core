from typing import Any, Dict, List, Union, Optional
import numpy as np
import torch
from transformer_lens import HookedTransformer

from .base import BaseFeatureExtractor


class LanguageModelFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor that uses HookedTransformer to extract embeddings from text.

    This extractor supports different language models and can extract features
    from either the last token or average across all tokens. It now supports
    multi-layer extraction with lazy loading.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the language model feature extractor.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - model_name (str): Name of the language model to use
                - layer_idx (int): Index of the layer to extract features from (for backward compatibility)
                - hook_type (str): Type of hook to use (default: "hook_resid_pre")
                - last_token (bool): Whether to use only the last token's features
                - device (str): Device to run the model on ('cuda' or 'cpu')
                - context_type (str): Type of context to use (fullcontext, nocontext, halfcontext)
        """
        super().__init__(config)
        self.model_name = config["model_name"]
        self.layer_idx = config.get("layer_idx", -1)  # For backward compatibility
        self.hook_type = config.get("hook_type", "hook_resid_pre")
        self.last_token = config.get("last_token", True)
        self.context_type = config.get("context_type", "fullcontext")

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Initialize model
        self.model = HookedTransformer.from_pretrained(
            self.model_name, device=self.device
        )
        self.model.eval()

    def extract_features(
        self, stimuli: Union[str, List[str]], layer_idx: Optional[int] = None, **kwargs
    ) -> np.ndarray:
        """Extract features from the input stimuli using a for loop.

        Args:
            stimuli (Union[str, List[str]]): Input text or list of texts
            layer_idx (Optional[int]): Specific layer to extract from. If None, uses self.layer_idx
            **kwargs: Additional arguments for feature extraction

        Returns:
            np.ndarray: Extracted features
        """
        if layer_idx is None:
            layer_idx = self.layer_idx

        if isinstance(stimuli, str):
            stimuli = [stimuli]

        # Process each stimulus individually
        all_features = []
        print(f"Processing {len(stimuli)} texts one at a time...")

        for i, text in enumerate(stimuli):
            if i % 10 == 0:
                print(f"Processing text {i+1}/{len(stimuli)}")

            # Extract features for the current text
            features = self._extract_single_features(text, layer_idx)
            all_features.append(features)

        # Stack all features
        return np.vstack(all_features)

    def extract_all_layers(
        self, stimuli: Union[str, List[str]], **kwargs
    ) -> Dict[int, np.ndarray]:
        """Extract features from all layers for the input stimuli.

        Args:
            stimuli (Union[str, List[str]]): Input text or list of texts
            **kwargs: Additional arguments for feature extraction

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping layer indices to features
        """
        if isinstance(stimuli, str):
            stimuli = [stimuli]

        # Process each stimulus individually
        all_layer_features = {}
        # use the logger: TODO: Taha
        print(f"Processing {len(stimuli)} texts for all layers...")

        for i, text in enumerate(stimuli):
            if i % 10 == 0:
                print(f"Processing text {i+1}/{len(stimuli)}")

            # Extract all layers for the current text
            layer_features = self._extract_single_text_all_layers(text)

            # Accumulate features across texts
            for layer_idx, features in layer_features.items():
                if layer_idx not in all_layer_features:
                    all_layer_features[layer_idx] = []
                all_layer_features[layer_idx].append(features)

        # Stack features for each layer
        for layer_idx in all_layer_features:
            all_layer_features[layer_idx] = np.vstack(all_layer_features[layer_idx])

        return all_layer_features

    def _extract_single_features(self, text: str, layer_idx: int) -> np.ndarray:
        """Extract features from a single text for a specific layer.

        Args:
            text (str): Input text
            layer_idx (int): Layer index to extract from

        Returns:
            np.ndarray: Extracted features for the text
        """
        # if the text is '' then return np.zeros(dimensions of the features)
        if text == "":
            return np.zeros((self.model.cfg.d_model)).reshape(
                -1, self.model.cfg.d_model
            )

        with torch.no_grad():
            # Process a single text
            _, cache = self.model.run_with_cache(
                text, prepend_bos=True, return_type=None  # Return the raw outputs
            )

            # Get features from the specified hook and layer
            hook_name = f"blocks.{layer_idx}.{self.hook_type}"
            features = cache[hook_name]

            # Handle last token or average across tokens
            if self.last_token:
                # Get the last token's features
                token_features = features[0, -1].unsqueeze(0)  # Add batch dimension
            else:
                # Average across all tokens
                token_features = (
                    features[0].mean(dim=0).unsqueeze(0)
                )  # Add batch dimension

            # Convert to numpy array
            return token_features.cpu().numpy()

    def _extract_single_text_all_layers(self, text: str) -> Dict[int, np.ndarray]:
        """Extract features from all layers for a single text.

        Args:
            text (str): Input text

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping layer indices to features
        """
        # if the text is '' then return zeros for all layers
        if text == "":
            empty_features = np.zeros((self.model.cfg.d_model)).reshape(
                -1, self.model.cfg.d_model
            )
            return {i: empty_features for i in range(self.model.cfg.n_layers)}

        with torch.no_grad():
            # Process a single text
            _, cache = self.model.run_with_cache(
                text, prepend_bos=True, return_type=None  # Return the raw outputs
            )

            # Extract features from all layers
            all_layer_features = {}
            for layer_idx in range(self.model.cfg.n_layers):
                hook_name = f"blocks.{layer_idx}.{self.hook_type}"
                features = cache[hook_name]

                # Handle last token or average across tokens
                if self.last_token:
                    # Get the last token's features
                    token_features = features[0, -1].unsqueeze(0)  # Add batch dimension
                else:
                    # Average across all tokens
                    token_features = (
                        features[0].mean(dim=0).unsqueeze(0)
                    )  # Add batch dimension

                # Convert to numpy array and store
                all_layer_features[layer_idx] = token_features.cpu().numpy()

            return all_layer_features

    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        required_params = ["model_name"]
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required parameter: {param}")

        if "layer_idx" in self.config:
            if not isinstance(self.config["layer_idx"], int):
                raise ValueError("layer_idx must be an integer")

        if "device" in self.config:
            if self.config["device"] not in ["cuda", "cpu"]:
                raise ValueError("device must be either 'cuda' or 'cpu'")

        if "context_type" in self.config:
            valid_context_types = ["fullcontext", "nocontext", "halfcontext"]
            if self.config["context_type"] not in valid_context_types:
                raise ValueError(f"context_type must be one of {valid_context_types}")
