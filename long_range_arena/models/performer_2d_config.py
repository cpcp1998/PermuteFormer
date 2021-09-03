from transformers import BertConfig


class Performer2DConfig(BertConfig):
    model_type = "performer_2d"

    def __init__(
            self,
            qk_dim_multiplier=4,
            kernel_epsilon=1e-3,
            prefix=1,
            postfix=1,
            width=32,
            height=32,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.qk_dim_multiplier = qk_dim_multiplier
        self.kernel_epsilon = kernel_epsilon
        self.prefix = prefix
        self.postfix = postfix
        self.width = 32
        self.height = 32
