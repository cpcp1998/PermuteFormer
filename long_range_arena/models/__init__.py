from .bert import BertModel
from .bert_prenorm import BertModel as BertPrenormModel
from transformers import (
    BertConfig,
)
from .performer_config import PerformerConfig
from .performer_prenorm import BertModel as PerformerPrenormModel
from .performer_2d_config import Performer2DConfig
from .performer_2d_prenorm import BertModel as Performer2DModel
from .linformer_config import LinformerConfig
from .linformer_prenorm import BertModel as LinformerPrenormModel
from .nystromformer_config import NystromformerConfig
from .nystromformer_prenorm import BertModel as NystromformerPrenormModel
from transformers import ReformerConfig
from .reformer_prenorm import BertModel as ReformerModel
from .bert_prenorm_sin import BertModel as BertPrenormSinModel


MODEL_MAP = {
    "bert": (BertConfig, BertModel),
    "bert-prenorm": (BertConfig, BertPrenormModel),
    "performer-prenorm": (PerformerConfig, PerformerPrenormModel),
    "performer-2d": (Performer2DConfig, Performer2DModel),
    "linformer": (LinformerConfig, LinformerPrenormModel),
    "nystromformer": (NystromformerConfig, NystromformerPrenormModel),
    "reformer": (ReformerConfig, ReformerModel),
    "bert-prenorm-sin": (BertConfig, BertPrenormSinModel),
}
