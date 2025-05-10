#!/bin/bash


export TRITON_CACHE_DIR=/scratch/network/kselberg/.triton
export HF_HOME=/scratch/network/kselberg/.cache/huggingface
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
)

trainers_experiments=(
    "GradAscent unlearn/tofu/default.yaml"
    "GradDiff unlearn/tofu/default.yaml"
    "NPO unlearn/tofu/default.yaml"
    "DPO unlearn/tofu/idk.yaml"
    "RMU  unlearn/tofu/default.yaml"
)
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)


per_device_train_batch_size=4 # on two gpus would make effective batch size 32
gradient_accumulation_steps=4


########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################


for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            trainer=$(echo $trainer_experiment | cut -d' ' -f1)
            experiment=$(echo $trainer_experiment | cut -d' ' -f2)
            
            # Construct the data config path based on the experiment path
            # Assumes experiment path like 'unlearn/tofu/default.yaml' maps to 'configs/experiment/unlearn/tofu/default.yaml'
            data_cfg_path="configs/experiment/${experiment}"
            
            task_name=tofu_${model}_${forget_split}_${trainer} 
            model_path=open-unlearning/tofu_${model}_full
            loss_cache_path="saves/curriculum_losses/${task_name}_losses.pt"
            echo ${task_name}: Unlearning ${model_path} using ${trainer}

            # Check if the loss cache file exists. If not, calculate losses.
            if [ ! -f "$loss_cache_path" ]; then # Use standard test operator
                echo "Loss cache file $loss_cache_path not found. Calculating losses..."
                # Call the user's script to calculate losses
                # Assuming it takes these arguments - adjust if necessary
                bash scripts/calculate_curriculum_losses.sh \
                    --model_path "$model_path" \
                    --data_cfg_path "$data_cfg_path" \
                    --dataset_split "$forget_split" \
                    --output_path "$loss_cache_path" # Pass all required args
                
                # Check again if the file was created successfully
                if [ ! -f "$loss_cache_path" ]; then
                     echo "ERROR: Loss calculation script failed to create $loss_cache_path. Aborting."
                     exit 1
                fi
                echo "Losses calculated and saved to $loss_cache_path"
            else
                echo "Using existing loss cache file: $loss_cache_path"
            fi
            # --- End loss calculation check ---

            # Unlearn
            CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
            src/train.py --config-name=unlearn.yaml \
            experiment=${experiment} \
            trainer=${trainer} \
            task_name=${task_name} \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            ++trainer.curriculum_strategy="loss_hard_easy" \
            ++trainer.curriculum_loss_cache_path="$loss_cache_path"

            # Eval
            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            model=${model} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            paths.output_dir=saves/unlearn/${task_name}/evals \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
        done
    done
done