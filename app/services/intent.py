import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normaliza texto para classificacao: minuscula, sem acentos, espacos normalizados."""
    if not text:
        return ""
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


GREETINGS = {"oi", "ola", "ola", "bom dia", "boa tarde", "boa noite", "e ai", "eai", "hello", "hi", "ping"}
HELP_TERMS = {"ajuda", "help", "comandos", "como funciona", "o que voce faz"}
THANKS = {"obrigado", "obrigada", "valeu", "muito obrigado", "muito obrigada"}

SAMPLE_TERMS = [
    "amostra", "amostras", "exemplo", "exemplos", "caso exemplo",
    "casos exemplo", "alguns casos", "algumas linhas", "linhas de exemplo",
    "registros de exemplo", "me mostre casos", "me mostre exemplos",
    "traga exemplos", "traga amostras",
]

DETAIL_TERMS = [
    "liste", "listar", "listagem", "trecho", "trechos", "evidencia",
    "evidencias", "detalhe", "detalhes", "detalhado", "detalhada",
    "caso", "casos", "score", "estratificacao", "informacoes",
    "motivo", "motivos", "justificativa", "justificativas",
    "levaram", "levou", "influenciaram",
]

AGGREGATION_TERMS = [
    "quantos", "quantas", "quantidade", "contagem", "total", "soma",
    "media", "percentual", "porcentagem", "distribuicao", "agregado",
    "consolidado", "relatorio", "relatoiro",
]

# Pedido explicito de caracterizar/segregar pacientes em grupos clinicos.
GROUPING_TERMS = [
    "caracterize", "caracterizar", "caracterizacao",
    "segregue", "segregar", "segregacao",
    "estratifique", "estratificar", "estratificacao",
    "classifique os pacientes", "classifique pacientes",
    "agrupe os pacientes", "agrupe pacientes", "agrupar pacientes",
    "por grupo", "por grupos",
    "por classificacao", "por classificacao clinica",
    "por categoria", "por categorias",
    "liste por", "lista por",
    "liste os positivos", "liste positivos",
    "liste os negativos", "liste negativos",
    "liste os provaveis", "liste provaveis",
    "liste os pos-operatorios", "liste pos-operatorios", "liste pos operatorios",
    "detalhe os positivos", "detalhe os negativos",
    "detalhe os provaveis", "detalhe os pos-operatorios",
    "pacientes positivos", "pacientes provaveis",
    "pacientes negativos", "pacientes pos-operatorios", "pacientes pos operatorios",
    "grupos de pacientes",
]


def light_interaction(text: str):
    cleaned = normalize_text(text)
    if cleaned in GREETINGS:
        return "Ola, eu sou a Iris, em que posso te ajudar?"
    if cleaned in HELP_TERMS:
        return (
            "Eu posso ajudar com analises de cirurgia de catarata, prontuarios, "
            "classificacoes, evidencias, contagens e relatorios clinicos."
        )
    if cleaned in THANKS:
        return "Disponha. Se precisar de uma analise, e so me enviar o pedido."
    return None


def extract_sample_size(text: str) -> int:
    patterns = [
        r"\b(\d{1,2})\s+(amostras|exemplos|casos|linhas|registros)\b",
        r"\b(amostras|exemplos|casos|linhas|registros)\s+(de\s+)?(\d{1,2})\b",
        r"\b(top|primeiros|primeiras)\s+(\d{1,2})\b",
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


def detect_grouped_lists(text: str) -> bool:
    """Detecta pedido explicito de caracterizacao com listas por grupo clinico."""
    cleaned = normalize_text(text)
    if not cleaned:
        return False
    return any(term in cleaned for term in GROUPING_TERMS)


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

    wants_grouped_lists = False
    if not is_simple:
        wants_grouped_lists = any(term in cleaned for term in GROUPING_TERMS)

    if wants_grouped_lists:
        output_mode = "mixed"
    elif sample_mode:
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
        "wants_rows": wants_rows or wants_grouped_lists,
        "has_aggregation_intent": has_aggregation_intent,
        "has_detail_intent": wants_rows or wants_grouped_lists,
        "wants_grouped_lists": wants_grouped_lists,
        "output_mode": output_mode,
        "error": False,
        "light_response": light_interaction(original_input),
    }
