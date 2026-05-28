import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normaliza o texto do usuário para fins de classificação de intenção:
    - Converte para minúsculas.
    - Remove acentuação e caracteres diacríticos.
    - Substitui múltiplos espaços por um espaço simples.
    - Remove espaços no início e fim.
    """
    if not text:
        return ""
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


GREETINGS = {"oi", "ola", "olá", "bom dia", "boa tarde", "boa noite", "e ai", "eai", "hello", "hi", "ping"}
HELP_TERMS = {"ajuda", "help", "comandos", "como funciona", "o que voce faz", "o que você faz"}
THANKS = {"obrigado", "obrigada", "valeu", "muito obrigado", "muito obrigada"}

SAMPLE_TERMS = [
    "amostra", "amostras", "exemplo", "exemplos", "caso exemplo",
    "casos exemplo", "alguns casos", "algumas linhas", "linhas de exemplo",
    "registros de exemplo", "me mostre casos", "me mostre exemplos",
    "traga exemplos", "traga amostras"
]

DETAIL_TERMS = [
    "liste", "listar", "listagem", "trecho", "trechos", "evidencia",
    "evidencias", "detalhe", "detalhes", "detalhado", "detalhada",
    "caso", "casos", "score", "estratificacao", "informacoes",
    "motivo", "motivos", "justificativa", "justificativas",
    "levaram", "levou", "influenciaram"
]

AGGREGATION_TERMS = [
    "quantos", "quantas", "quantidade", "contagem", "total", "soma",
    "media", "percentual", "porcentagem", "distribuicao", "agregado",
    "consolidado", "relatorio"
]


def light_interaction(text: str) -> str | None:
    """Detecta interações simples (saudações, agradecimentos) sem acionar o pipeline completo."""
    cleaned = normalize_text(text)

    if cleaned in GREETINGS:
        return "Olá, eu sou a Iris, em que posso te ajudar?"
    if cleaned in HELP_TERMS:
        return (
            "Eu posso ajudar com análises de cirurgia de catarata, prontuários, "
            "classificações, evidências, contagens e relatórios clínicos."
        )
    if cleaned in THANKS:
        return "Disponha. Se precisar de uma análise, é só me enviar o pedido."
    return None


def extract_sample_size(text: str) -> int:
    patterns = [
        r"\b(\d{1,2})\s+(amostras|exemplos|casos|linhas|registros)\b",
        r"\b(amostras|exemplos|casos|linhas|registros)\s+(de\s+)?(\d{1,2})\b",
        r"\b(top|primeiros|primeiras)\s+(\d{1,2})\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            for group in match.groups():
                if group and group.isdigit():
                    val = int(group)
                    if val > 0:
                        return min(val, 20)
    return 5


def detect_intent(original_input: str) -> dict:
    cleaned = normalize_text(original_input)

    is_simple = cleaned in GREETINGS | HELP_TERMS | THANKS

    sample_mode = False
    if not is_simple:
        sample_mode = any(term in cleaned for term in SAMPLE_TERMS)

    sample_size = extract_sample_size(cleaned) if sample_mode else None

    wants_rows = False
    if not is_simple:
        wants_rows = sample_mode or any(term in cleaned for term in DETAIL_TERMS)

    has_aggregation_intent = False
    if not is_simple and not sample_mode:
        has_aggregation_intent = any(term in cleaned for term in AGGREGATION_TERMS)

    if sample_mode:
        output_mode = "sample"
    elif wants_rows:
        output_mode = "rows"
    else:
        output_mode = "summary"

    return {
        "originalInput": original_input,
        "is_simple": is_simple,
        "sample_mode": sample_mode,
        "sample_size": sample_size,
        "wants_rows": wants_rows,
        "has_aggregation_intent": has_aggregation_intent,
        "has_detail_intent": wants_rows,
        "output_mode": output_mode,
        "error": False,
        "light_response": light_interaction(original_input),
    }
