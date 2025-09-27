import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class QuantizationTask:
    index: int
    file_path: str
    file_name: str
    size_bytes: int


class MemoryManager:
    def __init__(self, memory_limit_bytes: int, logger: logging.Logger) -> None:
        self.memory_limit_bytes = int(memory_limit_bytes)
        self.logger = logger

    def estimate_required_bytes(self, weights_bytes: int) -> int:
        # Total = Weights + 1.2x Activations + 0.3x Overhead = 2.5x
        total_required = int(weights_bytes * 2.5)
        self.logger.debug(
            "Estimated required memory bytes=%d for weights bytes=%d",
            total_required,
            weights_bytes,
        )
        return total_required

    def can_process(self, weights_bytes: int) -> Tuple[bool, int]:
        required = self.estimate_required_bytes(weights_bytes)
        fits = required <= self.memory_limit_bytes
        return fits, required


class LongCatQuantizer:
    def __init__(self) -> None:
        # 1. Load Configuration
        self.cwd = os.getcwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_dir_env = os.environ.get("QUANTIZE_MODEL_DIR")
        if model_dir_env is None:
            self.model_directory = "/Users/macuser/models/longcat"
            self._bootstrap_basic_logging()
            logging.warning(
                "ENV QUANTIZE_MODEL_DIR not set. Using default: %s",
                self.model_directory,
            )
        else:
            self.model_directory = model_dir_env

        memory_gb_env = os.environ.get("QUANTIZE_MEMORY_GB")
        if memory_gb_env is None:
            self.memory_limit_gb = 192.0
            if not logging.getLogger().handlers:
                self._bootstrap_basic_logging()
            logging.warning(
                "ENV QUANTIZE_MEMORY_GB not set. Using default: %.2f GB",
                self.memory_limit_gb,
            )
        else:
            try:
                self.memory_limit_gb = float(memory_gb_env)
            except ValueError:
                self.memory_limit_gb = 192.0
                if not logging.getLogger().handlers:
                    self._bootstrap_basic_logging()
                logging.warning(
                    "ENV QUANTIZE_MEMORY_GB invalid '%s'. Falling back to default: %.2f GB",
                    memory_gb_env,
                    self.memory_limit_gb,
                )

        self.memory_limit_bytes = int(self.memory_limit_gb * (1024**3))

        # 2. Setup Logging
        self.general_log_path = os.path.join(
            self.cwd, f"{self.timestamp}_quantization_log.txt"
        )
        self.error_log_path = os.path.join(self.cwd, f"{self.timestamp}_error.log")
        self.logger = self._configure_logging()

        # Start banner with dynamic configuration
        self.logger.info("Starting Entropy-Weighted Quantization (EWQ) pipeline")
        self.logger.info("Working Directory: %s", self.cwd)
        self.logger.info("Model Directory: %s", self.model_directory)
        self.logger.info("Memory Limit: %.2f GB (%d bytes)", self.memory_limit_gb, self.memory_limit_bytes)
        self.logger.info("General Log: %s", self.general_log_path)
        self.logger.info("Error Log: %s", self.error_log_path)

        # 3. Inspect Architecture & build task list
        self.supported_extensions = (".safetensors", ".bin", ".pt", ".tensors")
        self.tasks: List[QuantizationTask] = self._discover_tasks()

        # 4. Initialize MemoryManager
        self.memory_manager = MemoryManager(self.memory_limit_bytes, self.logger)

        # Checkpointing
        self.checkpoint_path = os.path.join(self.cwd, "quantization_checkpoint.json")

    def _bootstrap_basic_logging(self) -> None:
        # Ensure warnings can be emitted before full logging setup
        if logging.getLogger().handlers:
            return
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    def _configure_logging(self) -> logging.Logger:
        logger = logging.getLogger("longcat_quantizer")
        logger.setLevel(logging.INFO)

        # Remove inherited handlers if re-instantiating in same process
        if logger.handlers:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler for general log (INFO and above)
        file_handler = logging.FileHandler(self.general_log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # File handler for error log (ERROR and above)
        error_handler = logging.FileHandler(self.error_log_path, encoding="utf-8")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        # Console handler to stdout for INFO messages
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _discover_tasks(self) -> List[QuantizationTask]:
        if not os.path.isdir(self.model_directory):
            self.logger.warning(
                "Model directory does not exist or is not a directory: %s",
                self.model_directory,
            )
            return []

        pattern = os.path.join(self.model_directory, "**", "*")
        candidate_paths = [p for p in glob(pattern, recursive=True) if os.path.isfile(p)]

        weight_files: List[Tuple[str, int]] = []
        for path in candidate_paths:
            _, ext = os.path.splitext(path)
            if ext.lower() in self.supported_extensions:
                try:
                    file_size = os.path.getsize(path)
                except OSError as exc:
                    self.logger.error("Failed to stat file '%s': %s", path, exc)
                    continue
                weight_files.append((path, file_size))

        # Sort by size ascending, then by path for determinism
        weight_files.sort(key=lambda item: (item[1], item[0]))

        tasks: List[QuantizationTask] = []
        for idx, (path, size) in enumerate(weight_files):
            tasks.append(
                QuantizationTask(
                    index=idx,
                    file_path=path,
                    file_name=os.path.basename(path),
                    size_bytes=size,
                )
            )

        if tasks:
            self.logger.info("Discovered %d weight files for quantization", len(tasks))
            for task in tasks:
                self.logger.info(
                    "Task %d: %s | size=%d bytes", task.index, task.file_path, task.size_bytes
                )
        else:
            self.logger.warning("No weight files found to quantize in: %s", self.model_directory)

        return tasks

    def _load_checkpoint(self) -> Optional[int]:
        if not os.path.isfile(self.checkpoint_path):
            return None
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            last_index = data.get("last_successful_layer")
            if isinstance(last_index, int):
                self.logger.info(
                    "Loaded checkpoint: last_successful_layer=%d", last_index
                )
                return last_index
            self.logger.warning("Checkpoint missing valid 'last_successful_layer'. Ignoring.")
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to read checkpoint '%s': %s", self.checkpoint_path, exc)
        return None

    def _save_checkpoint(self, last_successful_index: int) -> None:
        payload = {
            "last_successful_layer": int(last_successful_index),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self.logger.info(
                "Saved checkpoint after task index %d -> %s",
                last_successful_index,
                self.checkpoint_path,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to write checkpoint '%s': %s", self.checkpoint_path, exc)

    def _delete_checkpoint(self) -> None:
        if os.path.isfile(self.checkpoint_path):
            try:
                os.remove(self.checkpoint_path)
                self.logger.info("Deleted checkpoint file: %s", self.checkpoint_path)
            except OSError as exc:
                self.logger.error("Failed to delete checkpoint '%s': %s", self.checkpoint_path, exc)

    def load_weights(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load weights from a supported file into CPU memory as float32 numpy arrays.

        Supports:
        - .safetensors via safetensors + torch
        - .pt/.bin via torch.load
        - .tensors is not supported (raises RuntimeError)
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        self.logger.info("[LOAD] Loading weights from: %s", file_path)

        try:
            import torch  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "PyTorch is required to load weight files (.pt/.bin/.safetensors)."
            ) from exc

        if ext == ".safetensors":
            try:
                from safetensors.torch import safe_open  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "safetensors package is required to load .safetensors files."
                ) from exc

            tensors: Dict[str, np.ndarray] = {}
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    if not torch.is_floating_point(t):
                        # Skip non-floating tensors
                        continue
                    tensors[key] = t.detach().to(dtype=torch.float32, device="cpu").numpy()
            if not tensors:
                raise RuntimeError("No floating-point tensors found in .safetensors file.")
            self.logger.info("[LOAD] Loaded %d tensors from .safetensors", len(tensors))
            return tensors

        if ext in {".pt", ".bin"}:
            map_location = "cpu"
            obj = torch.load(file_path, map_location=map_location)
            # Common patterns: dict of tensors or {'state_dict': ...}
            if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
                state_dict = obj["state_dict"]
            elif isinstance(obj, dict):
                state_dict = obj
            else:
                raise RuntimeError("Unsupported .pt/.bin format: expected a dict or state_dict.")

            tensors = {}
            for name, t in state_dict.items():
                if hasattr(t, "detach"):
                    t_cpu = t.detach().to(dtype=torch.float32, device="cpu")
                    if torch.is_floating_point(t_cpu):
                        tensors[str(name)] = t_cpu.numpy()
            if not tensors:
                raise RuntimeError("No floating-point tensors found in .pt/.bin file.")
            self.logger.info("[LOAD] Loaded %d tensors from %s", len(tensors), ext)
            return tensors

        if ext == ".tensors":
            raise RuntimeError(".tensors format is not supported by this loader.")

        raise RuntimeError(f"Unsupported file extension: {ext}")

    def quantize_weights(
        self, weights: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
        """Quantize weights to int8 using a simple entropy-weighted scaling.

        For each tensor:
          - compute histogram-based Shannon entropy over 256 bins
          - normalize entropy to [0, 1]
          - compute scale = base_scale * (1.0 + 0.5 * entropy_norm)
          - q = clip(round(x * scale), -128, 127).astype(int8)

        Returns:
          - q_weights: dict of int8 numpy arrays
          - scales: dict of float (inverse of dequant step, used as scale)
          - entropies: dict of entropy bits
        """
        q_weights: Dict[str, np.ndarray] = {}
        scales: Dict[str, float] = {}
        entropies: Dict[str, float] = {}

        eps = 1e-6
        for name, arr in weights.items():
            if arr.dtype.kind not in {"f"}:
                continue
            x = arr.astype(np.float32, copy=False)

            # Shannon entropy over magnitude distribution
            abs_x = np.abs(x.reshape(-1))
            if abs_x.size == 0:
                H_bits = 0.0
            else:
                counts, _ = np.histogram(abs_x, bins=256, range=(0.0, float(abs_x.max() + eps)))
                total = counts.sum()
                if total == 0:
                    H_bits = 0.0
                else:
                    p = counts.astype(np.float64) / float(total)
                    # avoid log(0)
                    p = p[p > 0]
                    H_bits = float(-(p * (np.log(p) / np.log(2.0))).sum())

            entropy_norm = H_bits / np.log2(256.0)  # 0..1

            std = float(np.std(x))
            std = max(std, eps)
            base_scale = 127.0 / (3.0 * std)  # cover ~3 sigma
            scale = float(base_scale * (1.0 + 0.5 * entropy_norm))

            q = np.clip(np.rint(x * scale), -128, 127).astype(np.int8)

            q_weights[name] = q
            scales[name] = scale
            entropies[name] = H_bits

            self.logger.info(
                "[QUANTIZE] %s | std=%.6f, entropy=%.3f bits, scale=%.6f",
                name,
                std,
                H_bits,
                scale,
            )

        if not q_weights:
            raise RuntimeError("No quantizable floating tensors found.")

        self.logger.info("[QUANTIZE] Quantized %d tensors to int8", len(q_weights))
        return q_weights, scales, entropies

    def save_quantized_file(
        self,
        file_path: str,
        original_weights_bytes: int,
        q_weights: Dict[str, np.ndarray],
        scales: Dict[str, float],
        entropies: Dict[str, float],
    ) -> None:
        """Save quantized tensors and metadata.

        Writes:
          - <original>.ewq.npz: int8 tensors by name
          - <original>.ewq.meta.json: metadata including scales, entropies, and original size
        """
        out_npz = f"{file_path}.ewq.npz"
        out_meta = f"{file_path}.ewq.meta.json"

        # Save tensors
        try:
            np.savez_compressed(out_npz, **q_weights)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to write NPZ file: {out_npz}") from exc

        # Save metadata separately as JSON
        meta = {
            "original_file": file_path,
            "original_size_bytes": int(original_weights_bytes),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scales": {k: float(v) for k, v in scales.items()},
            "entropies_bits": {k: float(v) for k, v in entropies.items()},
            "dtype": "int8",
            "quantization": "entropy_weighted_symmetric_int8",
        }
        try:
            with open(out_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to write metadata JSON: {out_meta}") from exc

        self.logger.info(
            "[SAVE] Wrote quantized tensors: %s and metadata: %s",
            out_npz,
            out_meta,
        )

    def run(self) -> None:
        if not self.tasks:
            self.logger.info("No tasks to process. Exiting.")
            return

        resume_after_index: Optional[int] = self._load_checkpoint()
        start_index = 0
        if resume_after_index is not None:
            start_index = resume_after_index + 1
            if start_index >= len(self.tasks):
                self.logger.info(
                    "Checkpoint indicates all tasks completed (index %d). Cleaning up and exiting.",
                    resume_after_index,
                )
                self._delete_checkpoint()
                return
            self.logger.info("Resuming from task index %d", start_index)

        for task in self.tasks[start_index:]:
            self.logger.info(
                "Processing task %d: %s (size=%d bytes)",
                task.index,
                task.file_path,
                task.size_bytes,
            )

            fits, required_bytes = self.memory_manager.can_process(task.size_bytes)
            if not fits:
                self.logger.critical(
                    "Insufficient memory for task %d (%s). Required=%d bytes, Limit=%d bytes. Halting without checkpoint.",
                    task.index,
                    task.file_name,
                    required_bytes,
                    self.memory_manager.memory_limit_bytes,
                )
                # Halt immediately without saving checkpoint
                return

            try:
                weights = self.load_weights(task.file_path)
                q_weights, scales, entropies = self.quantize_weights(weights)
                self.save_quantized_file(
                    task.file_path,
                    task.size_bytes,
                    q_weights,
                    scales,
                    entropies,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.error(
                    "Task %d failed for file '%s': %s", task.index, task.file_path, exc
                )
                # Do not advance checkpoint on failure
                return

            # Only save checkpoint after successful processing of this task
            self._save_checkpoint(task.index)

        # If we finished all tasks, remove checkpoint
        self._delete_checkpoint()
        self.logger.info("All tasks completed successfully.")


def quantize_model_mixed_precision(
    model,
    calibration_loader,
    memory_budget_mb: float,
    logger: Optional[logging.Logger] = None,
    max_calib_batches: int = 5,
) -> "object":
    """Entropy-driven, memory-constrained mixed-precision quantization.

    Quantizes nn.Linear and nn.Conv2d layers to 4/8/16-bit based on activation entropy
    sensitivity under a total memory budget (in MB). The returned model uses lightweight
    wrappers that dequantize per-layer weights on-the-fly for inference.

    Args:
        model: A torch.nn.Module to be quantized (will be modified in-place).
        calibration_loader: DataLoader yielding batches; only the first element is used as input.
        memory_budget_mb: Target model size budget in megabytes.
        logger: Optional logger for progress output; a basic one is created if None.
        max_calib_batches: Limit on number of calibration batches to process.

    Returns:
        The quantized model (same instance) with mixed-precision wrappers applied.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("This function requires PyTorch to be installed.") from exc

    if logger is None:
        logger = logging.getLogger("mixed_precision_quantizer")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(ch)

    # ----- Discover quantizable layers -----
    quantizable_types = (nn.Linear, nn.Conv2d)
    layer_modules: Dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, quantizable_types):
            layer_modules[name] = module
    if not layer_modules:
        logger.warning("No quantizable layers (Linear/Conv2d) found. Returning model unchanged.")
        return model

    # ----- Collect activation outputs via forward hooks -----
    activation_samples: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_modules}

    def make_hook(key: str):
        def hook(module, inp, out):  # type: ignore[no-redef]
            # Store a small CPU sample for entropy estimation
            with torch.no_grad():
                try:
                    out_cpu = out.detach().to("cpu", dtype=torch.float32)
                    flat = out_cpu.view(-1)
                    if flat.numel() > 262144:
                        # Cap per-call sample to limit memory
                        flat = flat[:262144]
                    activation_samples[key].append(flat)
                except Exception:
                    pass
        return hook

    handles = []
    for name, module in layer_modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    batches_seen = 0
    with torch.no_grad():
        for batch in calibration_loader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            elif isinstance(batch, dict):
                # Try common keys
                inputs = batch.get("input") or batch.get("inputs") or next(iter(batch.values()))
            else:
                inputs = batch

            try:
                inputs = inputs.to("cpu")
            except Exception:
                pass

            try:
                _ = model(inputs)
            except Exception as exc:  # noqa: BLE001
                # If the model forward signature is incompatible, stop calibration early
                logger.warning("Calibration forward failed on a batch: %s", exc)
                break

            batches_seen += 1
            if batches_seen >= max_calib_batches:
                break

    # Remove hooks
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass
    if was_training:
        model.train()

    # ----- Entropy-based sensitivity per layer -----
    def shannon_entropy(values_np: np.ndarray) -> float:
        if values_np.size == 0:
            return 0.0
        abs_vals = np.abs(values_np)
        vmax = float(abs_vals.max()) if abs_vals.size else 0.0
        if vmax <= 0.0:
            return 0.0
        hist, _ = np.histogram(abs_vals, bins=256, range=(0.0, vmax))
        total = hist.sum()
        if total == 0:
            return 0.0
        p = hist.astype(np.float64) / float(total)
        p = p[p > 0]
        return float(-(p * (np.log(p) / np.log(2.0))).sum())

    layer_entropy_bits: Dict[str, float] = {}
    for name, samples in activation_samples.items():
        if not samples:
            layer_entropy_bits[name] = 0.0
            continue
        cat = torch.cat(samples, dim=0)
        if cat.numel() > 1_000_000:
            cat = cat[:1_000_000]
        layer_entropy_bits[name] = shannon_entropy(cat.numpy())

    # ----- Parameter counts and size models -----
    def module_param_count(mod: nn.Module) -> int:
        count = 0
        for p in mod.parameters(recurse=False):
            count += p.numel()
        return count

    layer_param_counts: Dict[str, int] = {name: module_param_count(m) for name, m in layer_modules.items()}

    def size_bytes_for_bits(param_count: int, bits: int) -> int:
        return int(np.ceil(param_count * (bits / 8.0)))

    budget_bytes = int(memory_budget_mb * 1024 * 1024)

    # ----- Greedy allocation: start at 4-bit, upgrade by sensitivity -----
    layer_bits: Dict[str, int] = {name: 4 for name in layer_modules.keys()}

    def total_size_current() -> int:
        return sum(size_bytes_for_bits(layer_param_counts[name], layer_bits[name]) for name in layer_bits)

    # Sort layers by entropy descending
    sorted_layers = sorted(layer_modules.keys(), key=lambda n: layer_entropy_bits.get(n, 0.0), reverse=True)

    improved = True
    while improved:
        improved = False
        for name in sorted_layers:
            current_b = layer_bits[name]
            next_b = 8 if current_b == 4 else (16 if current_b == 8 else 16)
            if current_b == 16:
                continue
            cur_total = total_size_current()
            delta = size_bytes_for_bits(layer_param_counts[name], next_b) - size_bytes_for_bits(layer_param_counts[name], current_b)
            if cur_total + delta <= budget_bytes:
                layer_bits[name] = next_b
                improved = True

        # If nothing changed in a full pass, stop

    total_final = total_size_current()
    if total_final > budget_bytes:
        logger.warning(
            "Budget not met even at minimal 4-bit for all layers. required=%d, budget=%d",
            total_final,
            budget_bytes,
        )

    # ----- Compute per-layer quantization params and wrap modules -----
    def compute_scale_zero(weight: torch.Tensor, act_samples: List[torch.Tensor], bits: int) -> Tuple[float, int]:
        eps = 1e-6
        w_std = float(weight.detach().to("cpu", dtype=torch.float32).std().item())
        a_std = 0.0
        if act_samples:
            cat = torch.cat(act_samples, dim=0)
            if cat.numel() > 1_000_000:
                cat = cat[:1_000_000]
            a_std = float(cat.std().item())
        std = max(w_std, a_std, eps)
        if bits >= 16:
            return 1.0, 0
        max_q = (1 << (bits - 1)) - 1  # 7 for 4-bit, 127 for 8-bit
        k_sigma = 3.0
        scale = float(max_q / (k_sigma * std))
        return scale, 0

    class MixedPrecisionLinear(nn.Module):
        def __init__(self, base: nn.Linear, bits: int, scale: float, zero: int):
            super().__init__()
            self.bits = int(bits)
            self.scale = float(scale)
            self.zero = int(zero)
            if self.bits >= 16:
                self.weight_fp = base.weight.detach().to(dtype=torch.float16, device="cpu")
                self.bias_fp = None if base.bias is None else base.bias.detach().to(dtype=torch.float16, device="cpu")
            else:
                w = base.weight.detach().to(dtype=torch.float32, device="cpu")
                max_q = (1 << (self.bits - 1)) - 1
                min_q = - (1 << (self.bits - 1))
                w_q = torch.clamp(torch.round(w * self.scale), min_q, max_q).to(dtype=torch.int8)
                self.weight_q = w_q
                self.bias_fp = None if base.bias is None else base.bias.detach().to(dtype=torch.float32, device="cpu")
            self.in_features = base.in_features
            self.out_features = base.out_features
            self.stride = None
            self.padding = None

        def forward(self, x):  # type: ignore[override]
            if self.bits >= 16:
                return F.linear(x, self.weight_fp.to(dtype=x.dtype), None if self.bias_fp is None else self.bias_fp.to(dtype=x.dtype))
            w = self.weight_q.to(dtype=torch.float32) / self.scale
            return F.linear(x, w.to(dtype=x.dtype), self.bias_fp.to(dtype=x.dtype) if self.bias_fp is not None else None)

    class MixedPrecisionConv2d(nn.Module):
        def __init__(self, base: nn.Conv2d, bits: int, scale: float, zero: int):
            super().__init__()
            self.bits = int(bits)
            self.scale = float(scale)
            self.zero = int(zero)
            self.stride = base.stride
            self.padding = base.padding
            self.dilation = base.dilation
            self.groups = base.groups
            self.padding_mode = base.padding_mode
            if self.bits >= 16:
                self.weight_fp = base.weight.detach().to(dtype=torch.float16, device="cpu")
                self.bias_fp = None if base.bias is None else base.bias.detach().to(dtype=torch.float16, device="cpu")
            else:
                w = base.weight.detach().to(dtype=torch.float32, device="cpu")
                max_q = (1 << (self.bits - 1)) - 1
                min_q = - (1 << (self.bits - 1))
                w_q = torch.clamp(torch.round(w * self.scale), min_q, max_q).to(dtype=torch.int8)
                self.weight_q = w_q
                self.bias_fp = None if base.bias is None else base.bias.detach().to(dtype=torch.float32, device="cpu")

        def forward(self, x):  # type: ignore[override]
            if self.bits >= 16:
                return F.conv2d(
                    x,
                    self.weight_fp.to(dtype=x.dtype),
                    None if self.bias_fp is None else self.bias_fp.to(dtype=x.dtype),
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            w = self.weight_q.to(dtype=torch.float32) / self.scale
            return F.conv2d(
                x,
                w.to(dtype=x.dtype),
                self.bias_fp.to(dtype=x.dtype) if self.bias_fp is not None else None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

    # Helper to set submodule by dotted path
    def set_submodule(root: nn.Module, path: str, new_mod: nn.Module) -> None:
        parts = path.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)

    # Build and apply wrappers
    for name, module in layer_modules.items():
        bits = layer_bits[name]
        scale, zero = compute_scale_zero(module.weight, activation_samples.get(name, []), bits)
        if isinstance(module, nn.Linear):
            wrapped = MixedPrecisionLinear(module, bits, scale, zero)
        else:
            wrapped = MixedPrecisionConv2d(module, bits, scale, zero)  # type: ignore[arg-type]
        set_submodule(model, name, wrapped)
        logger.info("Layer '%s' assigned %d-bit (scale=%.6f)", name, bits, scale)

    assigned_summary = {name: int(layer_bits[name]) for name in layer_modules}
    logger.info("Mixed-precision assignment complete. Layers: %s", assigned_summary)
    return model

def main() -> None:
    quantizer = LongCatQuantizer()
    quantizer.run()


if __name__ == "__main__":
    main()

