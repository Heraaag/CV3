import torch


def select_scores(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Selects the score of the given label for every sample.

    Args:
        scores (torch.Tensor): scores of shape (N, C) for N samples and C classes.
        labels (torch.Tensor): class indices of shape (N,).

    Returns:
        torch.Tensor: selected scores of shape (N,) with out[i] = scores[i, labels[i]].
    """
    selected = None

    ########################################################################
    # TODO:                                                                #
    # Select scores[i, labels[i]] for every sample i without a loop.       #
    # Indexing with tensors (or a PyTorch function) does it in one line.   #
    # selected = ...                                                       #
    ########################################################################


    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return selected


def stable_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Computes the softmax exp(x) / sum(exp(x)) along dimension dim.

    Args:
        x (torch.Tensor): scores of any shape, e.g. logits of shape (N, C).
        dim (int): the dimension along which the probabilities sum to 1.

    Returns:
        torch.Tensor: probabilities with the same shape as x.
    """
    probs = None

    ########################################################################
    # TODO:                                                                #
    # Compute the softmax with basic tensor operations. exp() overflows    #
    # for large inputs, so subtract the maximum along dim first. This      #
    # does not change the result (why?). keepdim=True helps to broadcast.  #
    # probs = ...                                                          #
    ########################################################################


    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return probs


def colorize_labels(label_map: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Converts a segmentation label map into an RGB image.

    Args:
        label_map (torch.Tensor): class indices of shape (H, W) or (B, H, W).
        palette (torch.Tensor): one RGB color per class, shape (C, 3).

    Returns:
        torch.Tensor: colors of shape (*label_map.shape, 3) with the dtype of the palette.
    """
    colors = None

    ########################################################################
    # TODO:                                                                #
    # Look up the color of every pixel without a loop.                     #
    # colors = ...                                                         #
    ########################################################################


    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return colors
