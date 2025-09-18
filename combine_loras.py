import torch
import comfy.utils
import folder_paths

class CombineLoras:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name1": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the first LoRA."}),
                "strength1": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Strength for the first LoRA."}),
                "lora_name2": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the second LoRA."}),
                "strength2": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Strength for the second LoRA."}),
            }
        }

    RETURN_TYPES = ("LORA_MODEL",)
    FUNCTION = "combine"
    CATEGORY = "loaders"
    DESCRIPTION = "Combine two LoRAs by merging their weights based on the given strengths."

    def combine(self, lora_name1, strength1, lora_name2, strength2):
        lora_path1 = folder_paths.get_full_path_or_raise("loras", lora_name1)
        lora1 = comfy.utils.load_torch_file(lora_path1, safe_load=True)

        lora_path2 = folder_paths.get_full_path_or_raise("loras", lora_name2)
        lora2 = comfy.utils.load_torch_file(lora_path2, safe_load=True)

        merged = {}
        all_keys = set(lora1.keys()) | set(lora2.keys())
        for key in all_keys:
            val1 = lora1.get(key, torch.zeros_like(lora2[key]) if key in lora2 else None)
            val2 = lora2.get(key, torch.zeros_like(lora1[key]) if key in lora1 else None)
            if val1 is not None and val2 is not None:
                merged[key] = strength1 * val1 + strength2 * val2
            elif val1 is not None:
                merged[key] = strength1 * val1
            elif val2 is not None:
                merged[key] = strength2 * val2

        return (merged,)

NODE_CLASS_MAPPINGS = {
    "CombineLoras": CombineLoras,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombineLoras": "Combine LoRAs",
}