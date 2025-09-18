import torch
from safetensors import safe_open
from safetensors.torch import save_file


"""
 Define your LoRAs and their weights, scaled to target ~1.0 strength in ComfyUI. The sum of all weights should not overpower the model. The sum should be in the range of 0.9 to 1.2. Below example weights will overpower the model, you need to minimize the model strength to 0.2 to achieve a good quality image if you run the script as is. Experiment.

 """
lora_recipes = [
    ("/ComfyUI/models/loras/Qwen/Qwen-NSFW.safetensors", 0.60),
    ("/ComfyUI/models/loras/Qwen/Qwen-NSFW-Beta2.safetensors", 0.65),
    ("/ComfyUI/models/loras/Qwen/Qwen-NSFW-Beta3.safetensors", 0.65),
    ("/ComfyUI/models/loras/Qwen/Qwen-NSFW-Beta4.safetensors", 0.86),
]
# The output path
output_lora_path = "/ComfyUI/models/loras/Qwen/NSFW_Meta-B4.safetensors"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

keys = set()
for lora_path, _ in lora_recipes:
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        keys.update(f.keys())

print(f"Found {len(keys)} unique keys to merge.")

combined_state_dict = {}
for lora_path, weight in lora_recipes:
    print(f"Adding {lora_path} with weight {weight}...")
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key).to(device)
            if key in combined_state_dict:
                combined_state_dict[key] += weight * tensor
            else:
                combined_state_dict[key] = weight * tensor

print(f"Saving new combined LoRA to {output_lora_path}...")
save_file({k: v.to("cpu") for k, v in combined_state_dict.items()}, output_lora_path)
print("Done! You now have a single linear merged LoRA.")