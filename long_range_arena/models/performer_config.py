from transformers import BertConfig


class PerformerConfig(BertConfig):
    model_type = "performer"

    def __init__(
            self,
            qk_dim_multiplier=4,
            kernel_epsilon=1e-3,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.qk_dim_multiplier = qk_dim_multiplier
        self.kernel_epsilon = kernel_epsilon
