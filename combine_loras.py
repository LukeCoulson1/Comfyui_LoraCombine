import torch
import comfy.utils
import folder_paths
import os
import sys

# Add the info directory to path to import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
info_dir = os.path.join(current_dir, "info")
if info_dir not in sys.path:
    sys.path.append(info_dir)

try:
    from check_compatibility import check_lora_compatibility, get_lora_info
    from merge_methods import MERGE_METHODS
    COMPATIBILITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import compatibility modules: {e}")
    COMPATIBILITY_AVAILABLE = False
    MERGE_METHODS = {
        "linear": {
            "function": None,
            "description": "Linear merge (fallback)",
            "best_for": "General purpose"
        }
    }


class CombineLoras:
    @classmethod
    def INPUT_TYPES(s):
        method_choices = list(MERGE_METHODS.keys())
        
        return {
            "required": {
                "lora_name1": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the first LoRA."}),
                "strength1": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Strength for the first LoRA."}),
                "lora_name2": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the second LoRA."}),
                "strength2": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Strength for the second LoRA."}),
                "merge_method": (method_choices, {"default": method_choices[0], "tooltip": "Method to use for merging LoRAs"}),
                "check_compatibility": ("BOOLEAN", {"default": True, "tooltip": "Check if LoRAs are compatible before merging"}),
            }
        }

    RETURN_TYPES = ("LORA_MODEL",)
    FUNCTION = "combine"
    CATEGORY = "loaders"
    DESCRIPTION = "Combine two LoRAs using various merging methods with compatibility checking."

    def combine(self, lora_name1, strength1, lora_name2, strength2, merge_method, check_compatibility):
        # Get full paths
        lora_path1 = folder_paths.get_full_path_or_raise("loras", lora_name1)
        lora_path2 = folder_paths.get_full_path_or_raise("loras", lora_name2)
        
        # Check compatibility if requested and available
        if check_compatibility and COMPATIBILITY_AVAILABLE:
            is_compatible, compat_info, lora_type = check_lora_compatibility(lora_path1, lora_path2)
            
            if not is_compatible:
                issues = compat_info.get("issues", ["Unknown compatibility issue"])
                error_msg = f"LoRA compatibility check failed! Cannot merge incompatible LoRAs:\n"
                for issue in issues:
                    error_msg += f"  - {issue}\n"
                error_msg += f"Compatibility details: {compat_info}\n"
                error_msg += "Please use compatible LoRAs or disable compatibility checking."
                
                # Print to console as well
                print(f"ERROR: {error_msg}")
                
                # Raise an exception to stop the process
                raise ValueError(error_msg)
            else:
                print(f"LoRAs are compatible ({lora_type}). Proceeding with {merge_method} merge.")
        
        # Load LoRA files
        lora1 = comfy.utils.load_torch_file(lora_path1, safe_load=True)
        lora2 = comfy.utils.load_torch_file(lora_path2, safe_load=True)
        
        # Apply the selected merge method
        if COMPATIBILITY_AVAILABLE and merge_method in MERGE_METHODS:
            merge_function = MERGE_METHODS[merge_method]["function"]
            if merge_function:
                try:
                    merged = merge_function(lora1, lora2, strength1, strength2)
                    print(f"Successfully merged using {merge_method} method")
                except Exception as e:
                    print(f"Error with {merge_method} method: {e}")
                    print("Falling back to linear merge")
                    merged = self._fallback_linear_merge(lora1, lora2, strength1, strength2)
            else:
                merged = self._fallback_linear_merge(lora1, lora2, strength1, strength2)
        else:
            # Fallback to original linear merge
            merged = self._fallback_linear_merge(lora1, lora2, strength1, strength2)

        return (merged,)
    
    def _fallback_linear_merge(self, lora1, lora2, strength1, strength2):
        """Fallback linear merge method (original implementation)"""
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

        return merged


class LoraInfo:
    """Optional node to display LoRA information"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA to analyze"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_info"
    CATEGORY = "loaders"
    DESCRIPTION = "Get detailed information about a LoRA file."

    def get_info(self, lora_name):
        if not COMPATIBILITY_AVAILABLE:
            return ("Compatibility checking not available",)
        
        try:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            info = get_lora_info(lora_path)
            
            if "error" in info:
                return (f"Error reading LoRA: {info['error']}",)
            
            info_str = f"LoRA Type: {info['type']}\n"
            info_str += f"Number of keys: {info['num_keys']}\n"
            info_str += f"Sample keys: {', '.join(info['sample_keys'])}\n"
            info_str += f"Sample shapes: {info['sample_shapes']}"
            
            return (info_str,)
            
        except Exception as e:
            return (f"Error analyzing LoRA: {str(e)}",)


NODE_CLASS_MAPPINGS = {
    "CombineLoras": CombineLoras,
    "LoraInfo": LoraInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombineLoras": "Combine LoRAs",
    "LoraInfo": "LoRA Info",
}