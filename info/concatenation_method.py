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
# The output path for your new, combined "Meta-LoRA"
output_lora_path = "ComfyUI/models/loras/Qwen/Qwen-Image-merge.safetensors"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


prefixes = set()
for lora_path, _ in lora_recipes:
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.endswith(".lora_down.weight"):
                prefix = key[:-len(".lora_down.weight")]
                prefixes.add(prefix)

print(f"Found {len(prefixes)} unique adapted modules to merge.")


combined_state_dict = {}
for prefix in sorted(prefixes):
    A_list = []
    B_list = []
    new_r = 0
    in_features = None
    out_features = None
    has_alpha = False

    for lora_path, weight in lora_recipes:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            A_key = f"{prefix}.lora_down.weight"
            B_key = f"{prefix}.lora_up.weight"
            alpha_key = f"{prefix}.alpha"

            if A_key in f.keys() and B_key in f.keys():
                A = f.get_tensor(A_key).to(device)
                B = f.get_tensor(B_key).to(device)
                r_i = A.shape[0]
                assert A.shape == (r_i, A.shape[1]), f"Unexpected A shape in {lora_path}"
                assert B.shape == (B.shape[0], r_i), f"Unexpected B shape in {lora_path}"

                if in_features is None:
                    in_features = A.shape[1]
                    out_features = B.shape[0]
                else:
                    assert in_features == A.shape[1], f"Dim mismatch in {lora_path}"
                    assert out_features == B.shape[0], f"Dim mismatch in {lora_path}"

                if alpha_key in f.keys():
                    alpha = f.get_tensor(alpha_key).item()
                    scaling = alpha / r_i
                    has_alpha = True
                else:
                    scaling = 1.0

                s = weight * scaling
                if s == 0:
                    continue
                sqrt_s = torch.sqrt(torch.tensor(s)).to(device)

                A = A * sqrt_s
                B = B * sqrt_s

                A_list.append(A)
                B_list.append(B)
                new_r += r_i

    if new_r > 0:
        combined_A = torch.cat(A_list, dim=0)
        combined_B = torch.cat(B_list, dim=1)
        combined_state_dict[f"{prefix}.lora_down.weight"] = combined_A.to("cpu")
        combined_state_dict[f"{prefix}.lora_up.weight"] = combined_B.to("cpu")
        if has_alpha:
            combined_state_dict[f"{prefix}.alpha"] = torch.tensor(new_r)


print(f"Saving new combined LoRA to {output_lora_path}...")
save_file(combined_state_dict, output_lora_path)
print("Done! You now have a single, combined LoRA")