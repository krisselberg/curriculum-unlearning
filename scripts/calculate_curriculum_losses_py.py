import argparse
import os
import logging
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

# Assuming your project structure allows these imports
# Adjust paths if necessary
import sys
# Add project root to path if needed, assuming this script is run from the project root
# or adjust relative paths below
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data import get_data, get_collators

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def calculate_losses(
    model_path: str,
    data_cfg_path: str,
    dataset_split: str,
    output_path: str,
    batch_size: int = 4, # Adjust as needed for CPU memory
    num_workers: int = 0
):
    """
    Calculates per-example losses for a given dataset split using a base model on CPU.

    Args:
        model_path: Path to the base model (Hugging Face ID or local path).
        data_cfg_path: Path to the OmegaConf (.yaml) file describing the data.
        dataset_split: The specific split name to process (e.g., 'forget01', 'train').
        output_path: File path to save the calculated losses.
        batch_size: Batch size for processing.
        num_workers: Number of workers for DataLoader.
    """
    logger.info("--- Starting Curriculum Loss Calculation ---")
    logger.info(f"Model Path: {model_path}")
    logger.info(f"Data Config Path: {data_cfg_path}")
    logger.info(f"Dataset Split: {dataset_split}")
    logger.info(f"Output Path: {output_path}")
    logger.info(f"Batch Size: {batch_size}")

    if not os.path.exists(data_cfg_path):
        logger.error(f"Data config file not found: {data_cfg_path}")
        sys.exit(1)

    # --- 1. Load Model and Tokenizer to CPU ---
    logger.info(f"Loading tokenizer from {model_path}...")
    try:
        # Trust remote code might be needed for some models/tokenizers
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Loading model {model_path} onto CPU...")
    try:
        # Load directly to CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            # Explicitly use float32 for CPU stability
            torch_dtype=torch.float32
        )
        model.eval() # Set to evaluation mode
        logger.info("Model loaded successfully on CPU.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        sys.exit(1)

    # --- 2. Load Data Config and Dataset ---
    logger.info("Loading data configuration...")
    try:
        # Load the specific data config needed
        # We assume data_cfg_path points to a file containing the 'data' node
        # However, this experiment config often only contains overrides.
        # We need to load the BASE dataset config to get all args like the HF 'path'.
        full_cfg = OmegaConf.load(data_cfg_path)

        # --- Determine and Load Base Dataset Config --- 
        # We assume the curriculum is based on the 'forget' dataset defined in the config
        # Construct the likely path to the base forget dataset config
        # This assumes a structure like configs/data/datasets/TOFU_QA_forget.yaml
        forget_key = list(full_cfg.data.forget.keys())[0] # e.g., 'TOFU_QA_forget'
        base_dataset_config_path = f"configs/data/datasets/{forget_key}.yaml"

        if not os.path.exists(base_dataset_config_path):
             logger.error(f"Base dataset config file not found: {base_dataset_config_path}")
             # Attempt fallback if experiment config is different
             base_dataset_config_path_alt = f"configs/data/datasets/{dataset_split}.yaml" # Less likely
             if os.path.exists(base_dataset_config_path_alt):
                  base_dataset_config_path = base_dataset_config_path_alt
             else:
                  sys.exit(1)

        logger.info(f"Loading base dataset configuration from: {base_dataset_config_path}")
        base_data_cfg_full = OmegaConf.load(base_dataset_config_path)
        # Navigate to the actual dataset definition within the base config file
        if forget_key not in base_data_cfg_full:
             logger.error(f"Key '{forget_key}' not found in base config file {base_dataset_config_path}")
             sys.exit(1)
        base_data_cfg = base_data_cfg_full[forget_key]
        logger.debug(f"Base dataset config loaded: {base_data_cfg}")
        # --- End Base Config Loading ---

        if 'data' not in full_cfg:
             # Maybe the path points directly to the data node config?
             data_cfg = full_cfg
        else:
             data_cfg = full_cfg.data # Assuming data config is under 'data' key

        # --- Merge Base and Experiment Config Args --- 
        # Get the specific forget dataset config from the EXPERIMENT file
        experiment_forget_cfg = data_cfg.forget[forget_key]

        # Start with base hf_args, then merge/override with experiment hf_args
        merged_hf_args = OmegaConf.merge(base_data_cfg.args.hf_args, experiment_forget_cfg.args.hf_args)
        # Ensure the specific split name from the command line is used
        merged_hf_args.name = dataset_split # Override name with the exact split needed
        logger.debug(f"Merged hf_args for get_data: {merged_hf_args}")
        
        # Prepare the config node structure expected by get_data
        # We only need to load the 'forget' part for loss calculation
        effective_data_cfg = OmegaConf.create({
            "forget": {
                forget_key: {
                    "handler": base_data_cfg.handler, # Get handler from base
                    "args": {
                        "hf_args": merged_hf_args
                        # Include other args from base_data_cfg.args if necessary
                        # Example: "question_key": base_data_cfg.args.question_key
                    }
                }
            }
        })
        # Copy necessary args from base_data_cfg.args (excluding hf_args)
        if hasattr(base_data_cfg, 'args'):
             for key, value in base_data_cfg.args.items():
                  if key != 'hf_args':
                       effective_data_cfg.forget[forget_key].args[key] = value

        # --- Load Base Model Config for Template Args --- 
        # The experiment config likely only overrides path, not template args.
        # Find the base model name from the experiment config defaults.
        base_model_name = None
        if 'defaults' in full_cfg:
             for default in full_cfg.defaults:
                  # --- Corrected Logic for DictConfig Items --- 
                  if isinstance(default, (dict, DictConfig)) and len(default) == 1:
                       key = list(default.keys())[0]
                       if isinstance(key, str) and key.strip().startswith('override /model'):
                            value = default[key]
                            if isinstance(value, str):
                                 base_model_name = value.strip()
                                 logger.info(f"Determined base model name: {base_model_name}")
                                 break # Found the model override
                  # --- End Corrected Logic ---

        if not base_model_name:
             # Fallback: try getting from model key directly if defaults override not found
             base_model_name = full_cfg.get('model',{}).get('_target_') # Or infer differently
             if not base_model_name: 
                  logger.error("Could not determine base model name from experiment config.")
                  sys.exit(1)

        base_model_config_path = f"configs/model/{base_model_name}.yaml"

        if not os.path.exists(base_model_config_path):
             logger.error(f"Base model config file not found: {base_model_config_path}")
             sys.exit(1)

        logger.info(f"Loading base model configuration for template_args from: {base_model_config_path}")
        base_model_cfg_full = OmegaConf.load(base_model_config_path)
        # The template_args are likely under the model name key or directly under model_args?
        # Adjust navigation as needed based on actual model config structure
        
        # --- Try top level first ---
        if 'template_args' in base_model_cfg_full:
             template_args = base_model_cfg_full.template_args
        # --- End top level check ---
        elif base_model_name in base_model_cfg_full and 'template_args' in base_model_cfg_full[base_model_name]:
            template_args = base_model_cfg_full[base_model_name].template_args
        # Add other potential locations if needed
        # elif 'model' in base_model_cfg_full and 'template_args' in base_model_cfg_full.model:
        #     template_args = base_model_cfg_full.model.template_args
        else:
             logger.error(f"Could not find 'template_args' in {base_model_config_path} (checked top level and under '{base_model_name}')")
             template_args = {} # Fallback to empty dict

        logger.debug(f"Loaded template_args: {template_args}")
        # --- End Template Args Loading ---

        logger.info(f"Loading dataset split '{dataset_split}'...")
        # Assuming get_data takes the data config node, mode, tokenizer, template_args
        # Pass 'eval' mode maybe? Or a specific mode if required for your splits
        data = get_data(
            effective_data_cfg, # Pass the merged and structured config
            mode='eval', # Use 'eval' or adjust as needed
            tokenizer=tokenizer, # Pass tokenizer
            template_args=template_args, # Pass template args
            # Do NOT pass specific_split here, as QADataset doesn't accept it
        )
        # get_data loads based on config splits ('forget', 'retain', etc.)
        # We need the dataset associated with the requested split name.
        # The structure might be data['forget'] which contains the actual dataset.
        if dataset_split in data:
             dataset = data[dataset_split]
        elif len(data) == 1:
            logger.warning(f"Dataset split '{dataset_split}' not found as a key in loaded data ({list(data.keys())}). Using the only loaded dataset.")
            dataset = next(iter(data.values()))
        else:
            raise ValueError(f"Could not find dataset for split '{dataset_split}'. Loaded data keys: {list(data.keys())}")
        logger.info(f"Loaded dataset split '{dataset_split}' with {len(dataset)} examples.")

    except Exception as e:
        logger.error(f"Failed to load data configuration or dataset: {e}", exc_info=True)
        sys.exit(1)

    # --- 3. Get Collator ---
    logger.info("Loading data collator...")
    try:
         # --- Load Collator Config Explicitly --- 
         # The experiment config doesn't contain the full collator definition.
         # We need to load the base collator config directly.
         # Assuming the standard collator is DataCollatorForSupervisedDataset based on unlearn.yaml
         collator_name = "DataCollatorForSupervisedDataset" # Hardcode based on inspection
         collator_config_path = f"configs/collator/{collator_name}.yaml"

         if not os.path.exists(collator_config_path):
              logger.error(f"Base collator config file not found: {collator_config_path}")
              sys.exit(1)

         logger.info(f"Loading base collator configuration from: {collator_config_path}")
         collator_cfg_full = OmegaConf.load(collator_config_path)
         # The actual config might be nested under the collator name key
         if collator_name not in collator_cfg_full:
              logger.error(f"Key '{collator_name}' not found in base config file {collator_config_path}")
              sys.exit(1)
         collator_cfg = collator_cfg_full[collator_name]
         logger.debug(f"Base collator config loaded: {collator_cfg}")
         # --- End Collator Config Loading ---

         # Prepare the structure expected by get_collators (dict with collator name as key)
         effective_collator_cfg = {collator_name: collator_cfg}

         # Pass the explicitly loaded config to get_collators
         collator = get_collators(effective_collator_cfg, tokenizer=tokenizer)
         logger.info("Data collator loaded.")
    except Exception as e:
        logger.error(f"Failed to load data collator: {e}", exc_info=True)
        sys.exit(1)


    # --- 4. Calculate Losses on CPU ---
    temp_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset), # Ensure original order
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False # Pinning memory is for GPU transfer
    )

    example_losses = []
    logger.info(f"Calculating per-example losses on CPU for {len(dataset)} examples...")
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(temp_loader), desc="Calculating Losses (CPU)"):
            # Ensure batch is on CPU (should be default)
            cpu_batch = {}
            for k, v in batch.items():
                 if isinstance(v, torch.Tensor):
                     cpu_batch[k] = v.to('cpu')
                 else:
                     cpu_batch[k] = v
            batch = cpu_batch

            # --- Extract Inputs (Adapt based on actual dataset structure) ---
            # This logic needs to mirror the structure your dataset/collator provides
            model_inputs = {}
            labels = None
            if isinstance(batch.get("forget"), dict) and "input_ids" in batch["forget"]:
                 forget_data = batch["forget"]
                 model_inputs["input_ids"] = forget_data.get("input_ids")
                 model_inputs["attention_mask"] = forget_data.get("attention_mask")
                 labels = forget_data.get("labels", forget_data.get("input_ids"))
            elif "input_ids" in batch: # Standard case
                 model_inputs["input_ids"] = batch.get("input_ids")
                 model_inputs["attention_mask"] = batch.get("attention_mask")
                 labels = batch.get("labels", batch.get("input_ids"))
            # Add elif for DPO 'rejected_' if needed for your curriculum basis
            # elif "rejected_input_ids" in batch: ...

            if model_inputs.get("input_ids") is None or labels is None:
                 logger.warning(f"Skipping batch: Could not find required input fields ('input_ids'/'labels', or 'forget' structure). Batch keys: {list(batch.keys())}")
                 current_batch_size = len(next(iter(batch.values()))) if batch else 0 # Estimate batch size
                 example_losses.extend([float('inf')] * current_batch_size)
                 continue
            # --- End Input Extraction ---

            # --- Shape Check/Fix (Optional but safer) ---
            try:
                if model_inputs.get("input_ids") is not None:
                    ids_tensor = model_inputs["input_ids"]
                    if ids_tensor.dim() == 3 and ids_tensor.shape[0] == 1:
                        ids_tensor = ids_tensor.squeeze(0)
                        model_inputs["input_ids"] = ids_tensor
                        if model_inputs.get("attention_mask") is not None:
                             mask_tensor = model_inputs["attention_mask"]
                             if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
                                 model_inputs["attention_mask"] = mask_tensor.squeeze(0)
                    if model_inputs["input_ids"].dim() != 2:
                         raise ValueError(f"Final input_ids shape is not 2D: {model_inputs['input_ids'].shape}")
            except Exception as shape_e:
                 logger.warning(f"Skipping batch due to shape error: {shape_e}")
                 current_batch_size = len(next(iter(batch.values()))) if batch else 0
                 example_losses.extend([float('inf')] * current_batch_size)
                 continue
            # --- End Shape Check ---

            # --- Perform forward pass and loss calculation ---
            logger.debug(f"Batch {batch_idx} - Input Shapes: input_ids={model_inputs.get('input_ids').shape}, attention_mask={model_inputs.get('attention_mask').shape}, labels={labels.shape}")
            logger.debug(f"Batch {batch_idx} - Input dtypes: input_ids={model_inputs.get('input_ids').dtype}, attention_mask={model_inputs.get('attention_mask').dtype}, labels={labels.dtype}")
            log_limit = 5 # Log first 5 elements
            if model_inputs.get('input_ids').numel() > 0:
                 logger.debug(f"Batch {batch_idx} - input_ids[:{log_limit},{log_limit}]: {model_inputs.get('input_ids').view(-1)[:log_limit]}")
            if labels.numel() > 0:
                logger.debug(f"Batch {batch_idx} - labels[:{log_limit},{log_limit}]: {labels.view(-1)[:log_limit]}")

            try:
                outputs = model(**model_inputs)

                # Log Logits
                logits = outputs.logits
                logger.debug(f"Batch {batch_idx} - Logits shape: {logits.shape}, dtype: {logits.dtype}")
                if torch.isnan(logits).any():
                    logger.warning(f"Batch {batch_idx} - Logits contain NaNs! Count: {torch.isnan(logits).sum()}")
                if torch.isinf(logits).any():
                    logger.warning(f"Batch {batch_idx} - Logits contain Infs! Count: {torch.isinf(logits).sum()}")
                if logits.numel() > 0:
                    logger.debug(f"Batch {batch_idx} - Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")

                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                logger.debug(f"Batch {batch_idx} - Shifted logits shape: {shift_logits.shape}, dtype: {shift_logits.dtype}")
                logger.debug(f"Batch {batch_idx} - Shifted labels shape: {shift_labels.shape}, dtype: {shift_labels.dtype}")

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                per_token_loss = loss_fct(flat_logits, flat_labels)

                # Log Per-Token Loss
                logger.debug(f"Batch {batch_idx} - Per-token loss shape: {per_token_loss.shape}, dtype: {per_token_loss.dtype}")
                if torch.isnan(per_token_loss).any():
                    logger.warning(f"Batch {batch_idx} - Per-token loss contains NaNs! Count: {torch.isnan(per_token_loss).sum()}")
                if torch.isinf(per_token_loss).any():
                    logger.warning(f"Batch {batch_idx} - Per-token loss contains Infs! Count: {torch.isinf(per_token_loss).sum()}")
                if per_token_loss.numel() > 0:
                    logger.debug(f"Batch {batch_idx} - Per-token loss stats: min={per_token_loss.min():.4f}, max={per_token_loss.max():.4f}, mean={per_token_loss.mean():.4f}")

                per_token_loss = per_token_loss.view(shift_logits.size(0), shift_logits.size(1))
                label_mask = (shift_labels != -100).float() # Ignore padding tokens
                sum_loss_per_example = (per_token_loss * label_mask).sum(dim=1)
                num_valid_tokens = label_mask.sum(dim=1).clamp(min=1) # Avoid division by zero

                # Log Summed Loss and Valid Tokens
                logger.debug(f"Batch {batch_idx} - Sum loss/example shape: {sum_loss_per_example.shape}, values[:{log_limit}]: {sum_loss_per_example[:log_limit]}")
                logger.debug(f"Batch {batch_idx} - Num valid tokens shape: {num_valid_tokens.shape}, values[:{log_limit}]: {num_valid_tokens[:log_limit]}")

                mean_loss_per_example = sum_loss_per_example / num_valid_tokens

                # Log Mean Loss per Example
                logger.debug(f"Batch {batch_idx} - Mean loss/example shape: {mean_loss_per_example.shape}, values[:{log_limit}]: {mean_loss_per_example[:log_limit]}")
                if torch.isnan(mean_loss_per_example).any():
                     logger.warning(f"Batch {batch_idx} - Mean loss/example contains NaNs! Count: {torch.isnan(mean_loss_per_example).sum()}")
                if torch.isinf(mean_loss_per_example).any():
                     logger.warning(f"Batch {batch_idx} - Mean loss/example contains Infs! Count: {torch.isinf(mean_loss_per_example).sum()}")

                # --- Log Individual Sample Loss --- 
                start_idx = batch_idx * batch_size
                q_key = dataset.question_key # Assumes dataset object has question_key attr
                for i in range(len(mean_loss_per_example)):
                     current_dataset_idx = start_idx + i
                     if current_dataset_idx < len(dataset.data):
                          try:
                               original_item = dataset.data[current_dataset_idx]
                               question = original_item[q_key]
                               loss_value = mean_loss_per_example[i].item()
                               # Truncate question for cleaner logging if needed
                               truncated_question = (question[:75] + '...') if len(question) > 75 else question
                               logger.info(f"Sample {current_dataset_idx}: Loss={loss_value:.4f}, Q: '{truncated_question}'")
                          except Exception as log_e:
                               logger.warning(f"Could not log details for sample index {current_dataset_idx}: {log_e}")
                     else:
                          logger.warning(f"Skipping logging for index {current_dataset_idx}, out of bounds for dataset size {len(dataset.data)}")
                # --- End Individual Logging --- 

                example_losses.extend(mean_loss_per_example.cpu().tolist()) # Ensure losses are on CPU

            except Exception as forward_e:
                logger.error(f"Batch {batch_idx} - Error during forward pass or loss calculation: {forward_e}", exc_info=True)
                # Log input shapes that caused the error
                logger.error(f"Batch {batch_idx} - Error occurred with Input Shapes: input_ids={model_inputs.get('input_ids').shape if model_inputs.get('input_ids') is not None else 'None'}, labels={labels.shape if labels is not None else 'None'}")
                current_batch_size = model_inputs.get("input_ids").shape[0] if model_inputs.get("input_ids") is not None else batch_size
                example_losses.extend([float('inf')] * current_batch_size)
                continue # Skip to next batch if this one failed

    if len(example_losses) != len(dataset):
         logger.warning(f"Loss calculation mismatch. Expected {len(dataset)}, got {len(example_losses)}. Output file might be incomplete or curriculum may fail.")

    # --- 5. Save Losses ---
    logger.info(f"Saving {len(example_losses)} calculated losses to {output_path}...")
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(example_losses, output_path)
        logger.info("Losses saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save losses: {e}", exc_info=True)
        sys.exit(1)

    logger.info("--- Curriculum Loss Calculation Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate per-example losses for curriculum learning.")
    parser.add_argument("--model_path", required=True, help="Path to the base model (HF ID or local path)")
    parser.add_argument("--data_cfg_path", required=True, help="Path to the OmegaConf data configuration file (.yaml)")
    parser.add_argument("--dataset_split", required=True, help="Name of the dataset split to process (e.g., forget01)")
    parser.add_argument("--output_path", required=True, help="File path to save the calculated losses")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing (adjust for CPU memory)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")

    args = parser.parse_args()

    calculate_losses(
        model_path=args.model_path,
        data_cfg_path=args.data_cfg_path,
        dataset_split=args.dataset_split,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
