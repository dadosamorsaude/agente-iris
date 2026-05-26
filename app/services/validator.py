import re
from typing import Optional
from pydantic import BaseModel

class ValidationResult(BaseModel):
    is_valid: bool
    output: str
    error_type: Optional[str] = None
    should_retry: bool = False

def validate_response(response: str) -> ValidationResult:
    """
    Ported validation logic from n8n workflow.
    """
    response = response.strip()

    # 1. Technical error mappings
    error_mappings = {
        "sql_error": {
            "pattern": r"syntax|invalid query|database error",
            "message": "Houve um problema ao consultar o banco de dados. Nossa equipe foi notificada.",
            "retry": False
        },
        "timeout": {
            "pattern": r"timeout|timed out",
            "message": "A consulta demorou muito. Tente uma pergunta mais específica.",
            "retry": True
        },
        "auth": {
            "pattern": r"unauthorized|permission denied",
            "message": "Problema de autenticação. Entre em contato com o suporte.",
            "retry": False
        }
    }

    for key, config in error_mappings.items():
        if re.search(config["pattern"], response, re.IGNORECASE):
            return ValidationResult(
                is_valid=False,
                output=config["message"],
                error_type=key,
                should_retry=config["retry"]
            )

    # 2. Content validations
    if len(response) < 10:
        return ValidationResult(
            is_valid=False,
            output="A resposta ficou muito curta. Pode tentar novamente com mais detalhes?",
            error_type="response_too_short",
            should_retry=True
        )

    # 3. Placeholder check
    if re.search(r"\{\{.*?\}\}|\$\{.*?\}", response):
        return ValidationResult(
            is_valid=False,
            output="Erro ao processar a resposta. Nossa equipe foi notificada.",
            error_type="unresolved_placeholders",
            should_retry=False
        )

    # 4. Success
    return ValidationResult(
        is_valid=True,
        output=response
    )
