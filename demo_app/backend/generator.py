"""
generator.py
============
Self-contained xLSTMGenerator — generates REMIGEN2 token strings using the
xLSTM recurrent step() method (O(N) time, constant GPU memory).

Copied from notebooks/xLSTM-4-recurrent-state/inference.py and refactored:
  - model_from_config() inlined (helibrunna dependency removed entirely)
  - HelibrunnaLanguageModel comparison import removed
  - sys.path manipulation removed (midiprocessor is pip-installed)
"""
from __future__ import annotations

import glob
import os
import time
from typing import Optional

import torch
from dacite import from_dict
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig


# ---------------------------------------------------------------------------
# Internal helper — inlined from helibrunna.source.utilities.model_from_config
# Only the xLSTMLMModel branch is kept (the only model type we use).
# ---------------------------------------------------------------------------

def _build_xlstm_model(model_config, device: str) -> torch.nn.Module:
    """
    Build an xLSTMLMModel from an OmegaConf DictConfig.
    Falls back to 'vanilla' backend when CUDA is not available.
    """
    if not torch.cuda.is_available():
        model_config.slstm_block.slstm.backend = "vanilla"
        model_config.mlstm_block.mlstm.backend = "vanilla"

    config_obj = from_dict(xLSTMLMModelConfig, OmegaConf.to_container(model_config))
    model = xLSTMLMModel(config_obj)
    model.reset_parameters()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class xLSTMGenerator:
    """
    Fast, O(N) autoregressive generator for xLSTM language models.

    Uses the model's recurrent step() method instead of the slow parallel
    forward() loop, giving constant GPU memory and linear time complexity.

    Parameters
    ----------
    model_path_or_repo : str
        Path to the run directory (containing checkpoint sub-folders) OR
        directly to a checkpoint folder.
    checkpoint_name : str, optional
        Name of a specific checkpoint sub-folder (e.g. "checkpoint-46000-last").
        If None, the folder ending in "-last" is auto-discovered.
    config_overrides : dict, optional
        OmegaConf-compatible overrides merged on top of config.yaml.
        E.g. {"context_length": 16384}.
    device : str
        "auto" (default) | "cuda" | "cpu"
    """

    def __init__(
        self,
        model_path_or_repo: str,
        checkpoint_name: Optional[str] = None,
        config_overrides: Optional[dict] = None,
        device: str = "auto",
    ) -> None:
        if config_overrides is None:
            config_overrides = {}

        self.device = self._resolve_device(device)
        print(f"[xLSTMGenerator] Loading on device: {self.device}")

        model_path, tokenizer_path = self._resolve_paths(model_path_or_repo, checkpoint_name)

        # Load and optionally override config
        config_path = os.path.join(model_path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.yaml not found at {config_path}")
        self.config = OmegaConf.load(config_path)
        if config_overrides:
            self.config = OmegaConf.merge(self.config, OmegaConf.create(config_overrides))

        # Build model
        self.model = _build_xlstm_model(self.config, device=self.device)

        # Load weights
        weights_path = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"model.safetensors not found at {weights_path}")
        print("[xLSTMGenerator] Loading weights...")
        state_dict = load_file(weights_path)

        # CPU-only permutation fix (not normally needed on GPU servers)
        if not torch.cuda.is_available():
            for key in list(state_dict):
                if key.endswith("xlstm.slstm_cell._recurrent_kernel_"):
                    state_dict[key] = state_dict[key].permute(0, 2, 1)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Load tokenizer
        tokenizer_json = os.path.join(tokenizer_path, "tokenizer.json")
        if not os.path.exists(tokenizer_json):
            raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_json}")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_json)
        print("[xLSTMGenerator] Model and tokenizer ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 2048,
        temperature: float = 0.8,
        seed: Optional[int] = None,
        end_tokens: list[str] = [],
        forbidden_tokens: list[str] = [],
    ) -> dict:
        """
        Generate tokens autoregressively using the recurrent formulation.

        Returns a dict with:
            tokens          : list[str]  — raw REMIGEN2 token strings
            output          : str        — space-joined token string
            actual_tokens   : int        — number of generated tokens
            generation_time_s : float
            tokens_per_second : float
            target_reached  : bool
        """
        if seed is not None:
            torch.manual_seed(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        end_token_ids = [
            self.tokenizer.vocab[t] for t in end_tokens if t in self.tokenizer.vocab
        ]
        ids_to_mask = [
            tid
            for t in forbidden_tokens
            for tid in self.tokenizer(t).input_ids
            if t in self.tokenizer.vocab
        ]

        t0 = time.time()

        # 1. Prefill prompt — build recurrent state token-by-token
        state, logits = self._prefill(inputs)

        generated_ids = []
        sequence_length = inputs.shape[1]

        # 2. Recurrent generation loop
        while sequence_length < max_length:
            # Mask special / forbidden tokens
            logits[:, :, self.tokenizer.all_special_ids] = float("-inf")
            if ids_to_mask:
                logits[:, :, ids_to_mask] = float("-inf")

            scaled = logits / temperature
            probs = torch.nn.functional.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs[0, -1], num_samples=1).unsqueeze(0)  # (1,1)

            tok_id = next_token[0, 0].item()
            generated_ids.append(tok_id)
            sequence_length += 1

            if tok_id in end_token_ids:
                break

            logits, state = self.model.step(next_token, state=state)

        elapsed = time.time() - t0
        num_generated = len(generated_ids)
        tps = num_generated / elapsed if elapsed > 0 else 0.0

        output_str = self.tokenizer.decode(generated_ids)
        tokens_list = output_str.split()

        return {
            "tokens": tokens_list,
            "output": output_str,
            "actual_tokens": num_generated,
            "generation_time_s": elapsed,
            "tokens_per_second": tps,
            "target_reached": sequence_length >= max_length,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _prefill(self, inputs: torch.Tensor):
        """Step through prompt tokens one-by-one to build the recurrent state."""
        state: dict = {}
        logits = None
        for i in range(inputs.shape[1]):
            token = inputs[:, i : i + 1]
            logits, state = self.model.step(token, state=state)
        return state, logits

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _resolve_paths(repo: str, checkpoint_name: Optional[str]):
        """
        Locate the checkpoint folder and tokenizer.

        Handles two usage patterns:
          1. repo is a RUN directory (contains checkpoint-* subfolders)
             → scans for a subfolder ending in '-last', or uses checkpoint_name
          2. repo is already a CHECKPOINT folder (contains config.yaml)
             → uses repo directly as the model path
        """
        # Case 2: repo is itself a checkpoint (has config.yaml inside)
        if os.path.exists(os.path.join(repo, "config.yaml")):
            model_path = repo
        elif checkpoint_name is not None:
            # Case 1a: explicit checkpoint name given
            model_path = os.path.join(repo, checkpoint_name)
        else:
            # Case 1b: auto-discover *-last folder
            candidates = glob.glob(os.path.join(repo, "checkpoint-*"))
            model_path = next((p for p in candidates if p.endswith("-last")), None)

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No valid checkpoint found in {repo!r}. "
                "Pass checkpoint_name or ensure a folder ending in '-last' exists."
            )

        # Tokenizer may live in the checkpoint folder or its parent (run dir)
        tokenizer_path = model_path
        if not os.path.exists(os.path.join(model_path, "tokenizer.json")):
            parent = os.path.dirname(model_path)
            if os.path.exists(os.path.join(parent, "tokenizer.json")):
                tokenizer_path = parent

        return model_path, tokenizer_path
