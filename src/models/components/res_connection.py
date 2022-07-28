from torch import nn


class ResConnection(nn.Module):
    def __init__(
        self,
        inner_block: nn.Module,
    ):
        """
            Given an inner-block for the residual connection, wraps it with a residual block.
            - Note that the inner block output should be from the same dim of its input!
        """

        super().__init__()

        self.inner_block = inner_block

    def forward(self, x):

        residual = self.inner_block(x)
        assert residual.shape == x.shape, "ResConnection: input and output should have the same dim in residual block."

        return x + residual
