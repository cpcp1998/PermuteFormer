from transformers import BertConfig


class NystromformerConfig(BertConfig):
    model_type = "nystromformer"

    def __init__(
            self,
            num_landmarks=64,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_landmarks = num_landmarks
