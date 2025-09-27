import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import List, Optional, Tuple


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

    def load_weights(self, file_path: str) -> None:
        # Placeholder - replace with actual library calls as needed
        self.logger.info("[LOAD] Loading FP16 weights from: %s", file_path)
        self.logger.info("[LOAD] Using MPS device if available (simulated)")

    def quantize_weights(self, file_name: str) -> None:
        # Placeholder - replace with actual EWQ quantization procedure
        self.logger.info("[QUANTIZE] Performing EWQ on: %s (simulated)", file_name)
        self.logger.info("[QUANTIZE] Using entropy-weighted kernels (simulated)")

    def save_quantized_file(self, file_name: str, original_weights_bytes: int) -> None:
        # Placeholder - replace with actual save routine
        self.logger.info(
            "[SAVE] Saving quantized artifact for: %s (original size: %d bytes) (simulated)",
            file_name,
            original_weights_bytes,
        )
        self.logger.info("[SAVE] Writing to target storage (simulated)")

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
                self.load_weights(task.file_path)
                self.quantize_weights(task.file_name)
                self.save_quantized_file(task.file_name, task.size_bytes)
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


def main() -> None:
    quantizer = LongCatQuantizer()
    quantizer.run()


if __name__ == "__main__":
    main()

