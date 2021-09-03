from transformers import BertConfig


class LinformerConfig(BertConfig):
    model_type = "linformer"

    def __init__(
            self,
            linformer_k=256,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.linformer_k = linformer_k
