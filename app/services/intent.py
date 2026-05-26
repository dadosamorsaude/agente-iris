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
    # Converte para minúsculas
    text = text.lower()
    # Remove acentos
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    # Substitui múltiplos espaços por um espaço simples
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sample_size(text: str) -> int:
    """
    Extrai o número de amostras solicitadas pelo usuário usando padrões regex:
    1. ex: "5 amostras", "10 exemplos"
    2. ex: "amostras de 3", "exemplos de 12"
    3. ex: "top 8", "primeiros 15"
    Limita o resultado a no máximo 20 e no mínimo 1. Fallback é 5.
    """
    patterns = [
        r"\b(\d{1,2})\s+(amostras|exemplos|casos|linhas|registros)\b",
        r"\b(amostras|exemplos|casos|linhas|registros)\s+(de\s+)?(\d{1,2})\b",
        r"\b(top|primeiros|primeiras)\s+(\d{1,2})\b"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Pega o primeiro grupo que contém dígitos
            for group in match.groups():
                if group and group.isdigit():
                    val = int(group)
                    if val > 0:
                        return min(val, 20)
    return 5


def detect_intent(original_input: str) -> dict:
    """
    Classifica deterministicamente a intenção da pergunta do usuário com base nas regras do n8n:
    - Identifica se é uma saudação/ajuda simples (is_simple).
    - Identifica se solicita amostras individuais (sample_mode) e calcula o sample_size.
    - Identifica se necessita de evidências ou linhas detalhadas (wants_rows).
    - Identifica se contém intenções agregadoras como contagem/relatórios (has_aggregation_intent).
    - Define o output_mode final ("sample", "rows" ou "summary").
    """
    cleaned = normalize_text(original_input)
    
    simple_inputs = {
        "ola", "oi", "bom dia", "boa tarde", "boa noite", "e ai", "eai", 
        "hello", "hi", "ping", "ajuda", "help", "comandos", 
        "como funciona", "o que voce faz"
    }
    
    sample_terms = [
        "amostra", "amostras", "exemplo", "exemplos", "caso exemplo", 
        "casos exemplo", "alguns casos", "algumas linhas", "linhas de exemplo", 
        "registros de exemplo", "me mostre casos", "me mostre exemplos", 
        "traga exemplos", "traga amostras"
    ]
    
    detail_terms = [
        "liste", "listar", "listagem", "trecho", "trechos", "evidencia", 
        "evidencias", "detalhe", "detalhes", "detalhado", "detalhada", 
        "caso", "casos", "score", "estratificacao", "informacoes", 
        "motivo", "motivos", "justificativa", "justificativas", 
        "levaram", "levou", "influenciaram"
    ]
    
    aggregation_terms = [
        "quantos", "quantas", "quantidade", "contagem", "total", "soma", 
        "media", "percentual", "porcentagem", "distribuicao", "agregado", 
        "consolidado", "relatorio"
    ]
    
    is_simple = cleaned in simple_inputs
    
    sample_mode = False
    if not is_simple:
        sample_mode = any(term in cleaned for term in sample_terms)
        
    sample_size = extract_sample_size(cleaned) if sample_mode else None
    
    wants_rows = False
    if not is_simple:
        wants_rows = sample_mode or any(term in cleaned for term in detail_terms)
        
    has_aggregation_intent = False
    if not is_simple and not sample_mode:
        has_aggregation_intent = any(term in cleaned for term in aggregation_terms)
        
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
        "error": False
    }
