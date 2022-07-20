from torch import nn


class FullyConnectedNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,

        # First Option: one dim to all hidden layers
        hidden_layers_unified_dim: int = None,
        hidden_layers_count: int = None,

        # Second Option: list of hidden dimensions
        hidden_layers_list_dim: list = None,

        with_dropout_hidden: float = 0,
        with_batch_norm_hidden: bool = True,
        with_batch_norm_end: bool = True,
    ):
        """
        Generates block of H * [Linear -> BN -> ReLU] -> [Linear -> BN]
        - The last block shown is always present (even when hidden_layers_count = 0).
        - Must be given with `input_size`, the dimension of the input
        - One may define the hidden layers EITHER via `hidden_layers_unified_dim` or `hidden_layers_list_dim`

        Flags:
            with_dropout_hidden: defines probability of dropout (p=0 will exclude dropout, recommended when including BN)
            with_batch_norm_hidden: whether to include batch-norm layer after each linear activation
            with_batch_norm_end: whether to include batch-norm layer in the very end.
        """
        super().__init__()

        assert (hidden_layers_count is not None and hidden_layers_unified_dim is not None) \
               or (hidden_layers_list_dim is not None), "FullyConnectedNet: Initialization must define either options."

        layers_list = []
        if hidden_layers_list_dim is None:
            hidden_layers_list_dim = hidden_layers_count * [hidden_layers_unified_dim]

        # Generate hidden layers
        in_dim = input_size
        for out_dim in hidden_layers_list_dim:
            layers_list += [nn.Linear(in_dim, out_dim)]
            if with_batch_norm_hidden:
                layers_list += [nn.BatchNorm1d(out_dim)]
            layers_list += [nn.ReLU()]
            if with_dropout_hidden != 0:
                layers_list += [nn.Dropout(p=with_dropout_hidden)]

            in_dim = out_dim

        # Generate the ending layer
        layers_list += [nn.Linear(in_dim, output_size)]
        if with_batch_norm_end:
            layers_list += [nn.BatchNorm1d(output_size)]

        # Build all into sequential module
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)
