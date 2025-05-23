from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from copy import deepcopy
from packaging import version
from trainer.base import FinetuneTrainer

from transformers.trainer_pt_utils import (
    nested_detach,
)


from transformers.utils import (
    is_sagemaker_mp_enabled,
)

from accelerate.utils import (
    is_deepspeed_available,
)

if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import (
        smp_forward_only,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_deepspeed_available():
    import deepspeed

from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.sampler import Sampler
import logging
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)

class CurriculumSampler(Sampler[int]):
    """Samples elements sequentially according to a pre-defined list of indices."""
    def __init__(self, indices: List[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

class UnlearnTrainer(FinetuneTrainer):
    def __init__(
        self,
        curriculum_strategy: str | None = None, # e.g., "loss_hard_easy"
        curriculum_loss_cache_path: str | None = None, # Path to PRE-COMPUTED losses
        *args,
        **kwargs,
    ):
        """
        Initialize the trainer.

        Args:
            curriculum_strategy: Strategy for curriculum learning ('loss_hard_easy', 'loss_easy_hard', None).
            curriculum_loss_cache_path: Path to load pre-computed curriculum losses from.
            *args, **kwargs: Arguments passed to the parent Trainer.
        """
        super().__init__(*args, **kwargs)

        self.curriculum_strategy = curriculum_strategy
        self.curriculum_loss_cache_path = curriculum_loss_cache_path
        self.sorted_indices = None # Will be populated here if possible

        # --- Load Pre-computed Losses and Sort --- 
        if self.curriculum_strategy and self.curriculum_loss_cache_path and self.train_dataset:
            logger.info(f"Attempting to load pre-computed curriculum losses from: {self.curriculum_loss_cache_path}")
            if os.path.exists(self.curriculum_loss_cache_path):
                try:
                    losses = torch.load(self.curriculum_loss_cache_path)
                    if len(losses) == len(self.train_dataset):
                        logger.info(f"Successfully loaded {len(losses)} pre-computed losses.")
                        # --- Sort indices based on loaded loss --- 
                        indices = list(range(len(self.train_dataset)))
                        reverse_sort = self.curriculum_strategy == "loss_easy_hard"
                        logger.info(f"Sorting {len(indices)} indices based on loaded losses. Strategy: {self.curriculum_strategy} (Reverse sort: {reverse_sort})")

                        valid_pairs = [(i, l) for i, l in zip(indices, losses) if isinstance(l, (int, float)) and torch.isfinite(torch.tensor(l))]
                        invalid_indices = [i for i, l in zip(indices, losses) if not (isinstance(l, (int, float)) and torch.isfinite(torch.tensor(l)))]

                        if invalid_indices:
                            logger.warning(f"Found {len(invalid_indices)} invalid loss values (NaN/Inf) in cache file. Placing these at the end.")

                        sorted_valid = sorted(valid_pairs, key=lambda x: x[1], reverse=reverse_sort)
                        self.sorted_indices = [i for i, l in sorted_valid] + invalid_indices
                        logger.info("Finished sorting indices for curriculum based on loaded losses.")
                    else:
                        logger.error(f"Loaded loss count ({len(losses)}) doesn't match dataset size ({len(self.train_dataset)}). Disabling curriculum.")
                        self.curriculum_strategy = None # Disable
                except Exception as e:
                    logger.error(f"Failed to load or process curriculum losses from {self.curriculum_loss_cache_path}: {e}. Disabling curriculum.", exc_info=True)
                    self.curriculum_strategy = None # Disable
            else:
                logger.error(f"Curriculum loss cache file not found: {self.curriculum_loss_cache_path}. Curriculum disabled.")
                self.curriculum_strategy = None # Disable
        elif self.curriculum_strategy:
             logger.warning("Curriculum strategy specified, but no loss cache path or train_dataset provided. Curriculum disabled.")
             self.curriculum_strategy = None
        else:
            logger.info("No curriculum strategy specified.")
        # --- End Loading Logic --- 

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader. If curriculum is enabled and successful,
        it uses a sampler based on the pre-sorted indices.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if self.curriculum_strategy and self.sorted_indices:
             if len(self.sorted_indices) == len(train_dataset):
                 logger.info(f"Using CurriculumSampler with {len(self.sorted_indices)} indices based on strategy: {self.curriculum_strategy}")
                 train_sampler = CurriculumSampler(self.sorted_indices)
             else:
                  logger.error(f"CRITICAL: Sorted index count ({len(self.sorted_indices)}) mismatch with dataset size ({len(train_dataset)}). Falling back to default sampler.")
                  train_sampler = self._get_train_sampler()
        else:
             if self.curriculum_strategy and not self.sorted_indices:
                 logger.warning("Curriculum strategy was set, but sorted indices are not available. Falling back to default sampler.")
             train_sampler = self._get_train_sampler()


        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=getattr(self.args, 'worker_init_fn', None), 
        )

    # Adapted from Huggingface DPO Trainer: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    def _prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        The only change to this function is calling the Trainer's compute_loss, as it's often overridden by unlearning methods, and we want to maintain the Trainer's evaluation setup.
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        ### Call compute_loss of super class since overridden compute_loss is not be applicable to eval_dataset.
                        loss, outputs = super().compute_loss(
                            model, inputs, return_outputs=True
                        )
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
