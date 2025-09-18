import torch
from safetensors import safe_open


def check_lora_compatibility(lora_path1, lora_path2):
    """
    Check if two LoRAs are compatible for merging.
    
    Returns:
        tuple: (is_compatible, compatibility_info, lora_type)
    """
    try:
        with safe_open(lora_path1, framework="pt", device="cpu") as f1:
            keys1 = set(f1.keys())
            
        with safe_open(lora_path2, framework="pt", device="cpu") as f2:
            keys2 = set(f2.keys())
        
        # Detect LoRA type based on key patterns
        lora_type1 = detect_lora_type(keys1)
        lora_type2 = detect_lora_type(keys2)
        
        compatibility_info = {
            "lora1_type": lora_type1,
            "lora2_type": lora_type2,
            "common_keys": len(keys1 & keys2),
            "unique_keys1": len(keys1 - keys2),
            "unique_keys2": len(keys2 - keys1),
            "total_keys1": len(keys1),
            "total_keys2": len(keys2)
        }
        
        # Check basic compatibility
        is_compatible = True
        issues = []
        
        # Type compatibility check
        if lora_type1 != lora_type2:
            issues.append(f"LoRA types don't match: {lora_type1} vs {lora_type2}")
            is_compatible = False
        
        # Check for some common keys (they should share at least some structure)
        if len(keys1 & keys2) == 0:
            issues.append("No common keys found between LoRAs")
            is_compatible = False
        
        # Check for dimension compatibility on common keys
        if is_compatible:
            dim_issues = check_dimension_compatibility(lora_path1, lora_path2, keys1 & keys2)
            if dim_issues:
                issues.extend(dim_issues)
                is_compatible = False
        
        compatibility_info["issues"] = issues
        compatibility_info["is_compatible"] = is_compatible
        
        return is_compatible, compatibility_info, lora_type1
        
    except Exception as e:
        return False, {"error": str(e)}, "unknown"


def detect_lora_type(keys):
    """
    Detect the type of LoRA based on key patterns.
    """
    key_list = list(keys)
    
    # Check for standard LoRA patterns
    if any(".lora_down.weight" in key for key in key_list):
        return "standard_lora"
    elif any(".lora_A" in key or ".lora_B" in key for key in key_list):
        return "peft_lora"
    elif any("q_proj" in key or "k_proj" in key or "v_proj" in key for key in key_list):
        return "transformer_lora"
    elif any(".weight" in key for key in key_list):
        return "generic_weights"
    else:
        return "unknown"


def check_dimension_compatibility(lora_path1, lora_path2, common_keys):
    """
    Check if dimensions of common keys are compatible.
    """
    issues = []
    sample_keys = list(common_keys)[:5]  # Check first 5 common keys
    
    try:
        with safe_open(lora_path1, framework="pt", device="cpu") as f1, \
             safe_open(lora_path2, framework="pt", device="cpu") as f2:
            
            for key in sample_keys:
                if key in f1.keys() and key in f2.keys():
                    tensor1 = f1.get_tensor(key)
                    tensor2 = f2.get_tensor(key)
                    
                    if tensor1.shape != tensor2.shape:
                        issues.append(f"Dimension mismatch for '{key}': {tensor1.shape} vs {tensor2.shape}")
                        
    except Exception as e:
        issues.append(f"Error checking dimensions: {str(e)}")
    
    return issues


def get_lora_info(lora_path):
    """
    Get detailed information about a LoRA file.
    """
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            lora_type = detect_lora_type(keys)
            
            # Get some sample tensor info
            sample_shapes = {}
            for key in keys[:3]:  # First 3 keys
                tensor = f.get_tensor(key)
                sample_shapes[key] = list(tensor.shape)
            
            return {
                "type": lora_type,
                "num_keys": len(keys),
                "sample_keys": keys[:5],
                "sample_shapes": sample_shapes
            }
    except Exception as e:
        return {"error": str(e)}