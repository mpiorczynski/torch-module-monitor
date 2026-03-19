import torch


def extract_activation_tensor(output):
    """Extract the activation tensor from a module output.

    Many modules return tuples (output, cache, ...) instead of plain tensors.
    This extracts the first tensor element from such outputs.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    return output
