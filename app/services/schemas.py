from typing import Literal
from pydantic import BaseModel


class DecisionOutput(BaseModel):
    action: Literal["analisar_prontuarios", "consultar_pop", "responder_direto"]