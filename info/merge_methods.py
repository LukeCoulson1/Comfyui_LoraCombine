import torch
from safetensors import safe_open


def linear_merge_method(lora_dict1, lora_dict2, strength1, strength2):
    """
    Linear merging method: directly adds weighted tensors together.
    This is the simplest and fastest method.
    
    Args:
        lora_dict1, lora_dict2: Loaded LoRA dictionaries
        strength1, strength2: Merge strengths for each LoRA
    
    Returns:
        dict: Merged LoRA dictionary
    """
    merged = {}
    all_keys = set(lora_dict1.keys()) | set(lora_dict2.keys())
    
    for key in all_keys:
        val1 = lora_dict1.get(key, None)
        val2 = lora_dict2.get(key, None)
        
        if val1 is not None and val2 is not None:
            # Both LoRAs have this key
            if val1.shape != val2.shape:
                # Handle shape mismatch by zero-padding smaller tensor
                max_shape = [max(s1, s2) for s1, s2 in zip(val1.shape, val2.shape)]
                if list(val1.shape) != max_shape:
                    padded_val1 = torch.zeros(max_shape, dtype=val1.dtype, device=val1.device)
                    padded_val1[tuple(slice(0, s) for s in val1.shape)] = val1
                    val1 = padded_val1
                if list(val2.shape) != max_shape:
                    padded_val2 = torch.zeros(max_shape, dtype=val2.dtype, device=val2.device)
                    padded_val2[tuple(slice(0, s) for s in val2.shape)] = val2
                    val2 = padded_val2
            
            merged[key] = strength1 * val1 + strength2 * val2
        elif val1 is not None:
            # Only first LoRA has this key
            merged[key] = strength1 * val1
        elif val2 is not None:
            # Only second LoRA has this key
            merged[key] = strength2 * val2
    
    return merged


def concatenation_merge_method(lora_dict1, lora_dict2, strength1, strength2):
    """
    Concatenation merging method: concatenates LoRA down/up matrices.
    This method preserves the LoRA structure better but is more complex.
    
    Args:
        lora_dict1, lora_dict2: Loaded LoRA dictionaries
        strength1, strength2: Merge strengths for each LoRA
    
    Returns:
        dict: Merged LoRA dictionary
    """
    # Find all unique module prefixes
    prefixes = set()
    
    for lora_dict in [lora_dict1, lora_dict2]:
        for key in lora_dict.keys():
            if key.endswith(".lora_down.weight"):
                prefix = key[:-len(".lora_down.weight")]
                prefixes.add(prefix)
            elif key.endswith(".lora_A"):
                prefix = key[:-len(".lora_A")]
                prefixes.add(prefix)
    
    if not prefixes:
        # Fallback to linear method if no LoRA structure found
        return linear_merge_method(lora_dict1, lora_dict2, strength1, strength2)
    
    merged = {}
    device = "cpu"  # Work on CPU to avoid memory issues
    
    for prefix in sorted(prefixes):
        # Try different key patterns
        patterns = [
            (".lora_down.weight", ".lora_up.weight", ".alpha"),
            (".lora_A", ".lora_B", ".alpha"),
        ]
        
        merged_this_prefix = False
        
        for down_suffix, up_suffix, alpha_suffix in patterns:
            down_key = f"{prefix}{down_suffix}"
            up_key = f"{prefix}{up_suffix}"
            alpha_key = f"{prefix}{alpha_suffix}"
            
            # Collect matrices from both LoRAs
            A_list = []
            B_list = []
            new_r = 0
            in_features = None
            out_features = None
            has_alpha = False
            
            for lora_dict, weight in [(lora_dict1, strength1), (lora_dict2, strength2)]:
                if weight == 0:
                    continue
                    
                if down_key in lora_dict and up_key in lora_dict:
                    A = lora_dict[down_key].to(device)
                    B = lora_dict[up_key].to(device)
                    r_i = A.shape[0]
                    
                    # Validate dimensions
                    if len(A.shape) != 2 or len(B.shape) != 2:
                        continue
                    if A.shape[0] != B.shape[1]:
                        continue
                    
                    if in_features is None:
                        in_features = A.shape[1]
                        out_features = B.shape[0]
                    else:
                        if in_features != A.shape[1] or out_features != B.shape[0]:
                            continue
                    
                    # Handle alpha scaling
                    if alpha_key in lora_dict:
                        alpha = lora_dict[alpha_key].item()
                        scaling = alpha / r_i
                        has_alpha = True
                    else:
                        scaling = 1.0
                    
                    # Apply weight and scaling
                    s = weight * scaling
                    if s == 0:
                        continue
                        
                    sqrt_s = torch.sqrt(torch.abs(torch.tensor(s))).to(device)
                    if s < 0:
                        sqrt_s = -sqrt_s
                    
                    A = A * sqrt_s
                    B = B * sqrt_s
                    
                    A_list.append(A)
                    B_list.append(B)
                    new_r += r_i
            
            # Merge if we found compatible matrices
            if new_r > 0 and A_list and B_list:
                combined_A = torch.cat(A_list, dim=0)
                combined_B = torch.cat(B_list, dim=1)
                
                merged[down_key] = combined_A
                merged[up_key] = combined_B
                
                if has_alpha:
                    merged[alpha_key] = torch.tensor(new_r, dtype=torch.float32)
                
                merged_this_prefix = True
                break
        
        # If concatenation failed, try linear merge for this prefix
        if not merged_this_prefix:
            for key in lora_dict1.keys():
                if key.startswith(prefix):
                    val1 = lora_dict1.get(key)
                    val2 = lora_dict2.get(key)
                    
                    if val1 is not None and val2 is not None:
                        merged[key] = strength1 * val1 + strength2 * val2
                    elif val1 is not None:
                        merged[key] = strength1 * val1
                    elif val2 is not None:
                        merged[key] = strength2 * val2
    
    # Handle any remaining keys not covered by prefixes
    all_keys = set(lora_dict1.keys()) | set(lora_dict2.keys())
    for key in all_keys:
        if key not in merged:
            val1 = lora_dict1.get(key)
            val2 = lora_dict2.get(key)
            
            if val1 is not None and val2 is not None:
                merged[key] = strength1 * val1 + strength2 * val2
            elif val1 is not None:
                merged[key] = strength1 * val1
            elif val2 is not None:
                merged[key] = strength2 * val2
    
    return merged


def weighted_average_method(lora_dict1, lora_dict2, strength1, strength2):
    """
    Weighted average method: normalizes strengths and averages.
    This ensures the merged LoRA maintains reasonable magnitude.
    
    Args:
        lora_dict1, lora_dict2: Loaded LoRA dictionaries
        strength1, strength2: Merge strengths for each LoRA
    
    Returns:
        dict: Merged LoRA dictionary
    """
    # Normalize weights
    total_weight = abs(strength1) + abs(strength2)
    if total_weight == 0:
        # Return empty dict if both weights are zero
        return {}
    
    norm_strength1 = strength1 / total_weight
    norm_strength2 = strength2 / total_weight
    
    return linear_merge_method(lora_dict1, lora_dict2, norm_strength1, norm_strength2)


# Available merge methods
MERGE_METHODS = {
    "linear": {
        "function": linear_merge_method,
        "description": "Linear merge: Direct weighted addition (fastest, simplest)",
        "best_for": "General purpose, quick merging"
    },
    "concatenation": {
        "function": concatenation_merge_method,
        "description": "Concatenation merge: Preserves LoRA structure (slower, more sophisticated)",
        "best_for": "Preserving LoRA internal structure and relationships"
    },
    "weighted_average": {
        "function": weighted_average_method,
        "description": "Weighted average: Normalized linear merge (prevents overpowering)",
        "best_for": "Maintaining balanced strength regardless of input weights"
    }
}