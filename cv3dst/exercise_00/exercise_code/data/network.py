import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        ########################################################################
        # TODO:                                                                #
        # Define the layers of your network for input images of shape         #
        # (B, 1, 28, 28). The notebook suggests two architectures. Keep the    #
        # number of parameters below 500,000. The test server creates the      #
        # network with Net(), so do not add arguments without default values.  #
        # You can define further modules, e.g. a residual block, as classes   #
        # in this file.                                                        #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = None

        ########################################################################
        # TODO:                                                                #
        # Compute the raw class scores (logits) of shape (B, num_classes).     #
        # Do not apply a softmax: nn.CrossEntropyLoss applies it itself.       #
        # logits = ...                                                         #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return logits
