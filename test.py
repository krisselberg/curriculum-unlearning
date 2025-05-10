import json
import os

def merge_forget05_summaries():
    base_model_name = "Llama-3.2-1B-Instruct"
    forget_split = "forget05"
    
    trainers = [
        "GradAscent",
        "GradDiff",
        "NPO",
        "DPO",
        "RMU"
    ]
    
    consolidated_results = {}
    
    base_save_dir = "saves/unlearn"
    output_prefix = "open-unlearning/" # Prefix for the keys in the output JSON
    
    for trainer in trainers:
        task_name = f"tofu_{base_model_name}_{forget_split}_{trainer}"
        summary_file_path = os.path.join(base_save_dir, task_name, "evals", "TOFU_SUMMARY.json")
        
        # Key for the output dictionary, matching your example
        output_key = os.path.join(output_prefix, base_save_dir, task_name, "evals", "TOFU_SUMMARY.json")
        
        if os.path.exists(summary_file_path):
            try:
                with open(summary_file_path, 'r') as f:
                    data = json.load(f)
                consolidated_results[output_key] = data
                print(f"Successfully read: {summary_file_path}")
            except json.JSONDecodeError:
                print(f"ERROR: Could not decode JSON from {summary_file_path}")
                consolidated_results[output_key] = {"error": "Failed to decode JSON"}
            except Exception as e:
                print(f"ERROR: Could not read file {summary_file_path}. Reason: {e}")
                consolidated_results[output_key] = {"error": f"Failed to read file: {e}"}
        else:
            print(f"WARNING: File not found - {summary_file_path}")
            consolidated_results[output_key] = {"error": "File not found"}
            
    return consolidated_results

if __name__ == "__main__":
    merged_data = merge_forget05_summaries()
    
    # Output the merged data as a JSON string
    output_json_string = json.dumps(merged_data, indent=2)
    print("\n--- Merged JSON Output for forget05 ---")
    print(output_json_string)
    
    # Optionally, save to a file
    # output_filename = "merged_forget05_summaries.json"
    # with open(output_filename, 'w') as outfile:
    #     json.dump(merged_data, outfile, indent=2)
    # print(f"\nMerged data also saved to {output_filename}")
