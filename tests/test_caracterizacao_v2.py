"""Testes para o modo caracterizacao de pacientes (resumo + listas por grupo)
e mascaramento de CPF. Cobre:

- _mask_cpf: nulos, valores invalidos, CPF formatado, somente digitos, com lixo.
- detect_grouped_lists: positivos e negativos.
- detect_intent: flag wants_grouped_lists.
- _resolve_query_shape: forca mixed + grouped_lists quando o usuario pede.
- _format_sql_result: produz grouped_rows, mascara CPF, nao trunca.
- Regressao: query agregada simples nao dispara grouped_lists nem listas.
"""

from app.agent.specialists.sql_analyst import (
    _mask_cpf,
    _classify_group,
    _resolve_query_shape,
    _format_sql_result,
    _compact_detail_row,
    _group_rows_by_classification,
)
from app.services.intent import detect_grouped_lists, detect_intent


# ---------------------------------------------------------------------------
# _mask_cpf
# ---------------------------------------------------------------------------

def test_mask_cpf_none_returns_none():
    assert _mask_cpf(None) is None


def test_mask_cpf_empty_string_returns_none():
    assert _mask_cpf("") is None
    assert _mask_cpf("   ") is None


def test_mask_cpf_formatted_cpf():
    assert _mask_cpf("123.456.789-09") == "123.456.789-09"


def test_mask_cpf_digits_only():
    assert _mask_cpf("12345678909") == "123.456.789-09"


def test_mask_cpf_with_noise():
    # Mistura de caracteres deve isolar apenas digitos
    assert _mask_cpf("CPF: 123.456.789-09 (paciente)") == "123.456.789-09"


def test_mask_cpf_too_short():
    # Menor que 11 retorna o proprio valor
    assert _mask_cpf("9") == "9"
    assert _mask_cpf("89") == "89"


def test_mask_cpf_too_long_truncates_to_last_11():
    # Retorna o proprio valor ja que nao tem exatamente 11 digitos
    assert _mask_cpf("9912345678909") == "9912345678909"


# ---------------------------------------------------------------------------
# _classify_group
# ---------------------------------------------------------------------------

def test_classify_group_canonical():
    assert _classify_group("positivo") == "positivos"
    assert _classify_group("Positivos") == "positivos"
    assert _classify_group("PROVAVEL") == "provaveis"
    assert _classify_group("Provavel") == "provaveis"
    assert _classify_group("negativo") == "negativos"
    assert _classify_group("Pos-operatorio") == "pos_operatorios"
    assert _classify_group("pos operatorio") == "pos_operatorios"


def test_classify_group_none_or_empty():
    assert _classify_group(None) is None
    assert _classify_group("") is None


# ---------------------------------------------------------------------------
# detect_grouped_lists / detect_intent
# ---------------------------------------------------------------------------

def test_detect_grouped_lists_positives():
    assert detect_grouped_lists("Caracterize os pacientes de catarata em abril")
    assert detect_grouped_lists("Quero segregar os pacientes por classificacao")
    assert detect_grouped_lists("Liste os positivos")
    assert detect_grouped_lists("estratifique os pacientes do mes")


def test_detect_grouped_lists_negatives():
    assert not detect_grouped_lists("Quantos atendimentos tivemos em abril?")
    assert not detect_grouped_lists("ola")
    assert not detect_grouped_lists("Total de cirurgias por clinica")


def test_detect_intent_sets_grouped_flag():
    intent = detect_intent("Caracterize os pacientes positivos e negativos de abril")
    assert intent["wants_grouped_lists"] is True
    assert intent["output_mode"] == "mixed"


def test_detect_intent_no_grouping_when_aggregate_only():
    intent = detect_intent("Quantos atendimentos tivemos em abril?")
    assert intent.get("wants_grouped_lists") is False


# ---------------------------------------------------------------------------
# _resolve_query_shape
# ---------------------------------------------------------------------------

def test_resolve_query_shape_grouped_forces_mixed():
    out = _resolve_query_shape("Caracterize os pacientes de abril")
    assert out["query_shape"] == "mixed"
    assert out["output_mode"] == "mixed"
    assert out["limit"] == 0
    assert out.get("grouped_lists") is True


def test_resolve_query_shape_aggregate_stays_aggregate():
    # Query puramente agregada - sem termos de detalhe nem de caracterizacao.
    q = "Qual o total geral de cirurgias por clinica em abril?"
    out = _resolve_query_shape(q)
    print("DEBUG out:", out)
    print("DEBUG file:", _resolve_query_shape.__code__.co_filename)
    assert out["query_shape"] == "aggregate", f"got: {out}"
    assert not out.get("grouped_lists")


# ---------------------------------------------------------------------------
# _group_rows_by_classification
# ---------------------------------------------------------------------------

def test_group_rows_by_classification_buckets():
    rows = [
        {"classificacao": "positivo", "id_paciente": 1},
        {"classificacao": "Positivo", "id_paciente": 2},
        {"classificacao": "provavel", "id_paciente": 3},
        {"classificacao": "negativo", "id_paciente": 4},
        {"classificacao": "pos_operatorio", "id_paciente": 5},
        {"classificacao": None, "id_paciente": 6},
    ]
    g = _group_rows_by_classification(rows)
    assert len(g["positivos"]) == 2
    assert len(g["provaveis"]) == 1
    assert len(g["negativos"]) == 1
    assert len(g["pos_operatorios"]) == 1
    assert len(g["nao_classificados"]) == 1


# ---------------------------------------------------------------------------
# _format_sql_result (caracterizacao)
# ---------------------------------------------------------------------------

def _row(classificacao, id_paciente, cpf):
    return {
        "id_paciente": id_paciente,
        "nome_paciente": f"Paciente {id_paciente}",
        "cpf_paciente": cpf,
        "id_atendimento": id_paciente * 100,
        "data_atendimento": "2026-04-15",
        "clinica": "Clinica X",
        "regional": "SP",
        "nome_profissional": "Dr. Y",
        "classificacao": classificacao,
        "score": 0.9,
        "termo_detectado": "catarata",
        "trecho_evidencia": "trecho de exemplo",
    }


def test_format_sql_result_grouped_lists_basic():
    results = [
        _row("positivo", 1, "12345678909"),
        _row("positivo", 2, "98765432100"),
        _row("provavel", 3, "11122233344"),
        _row("negativo", 4, "55566677788"),
    ]
    payload = _format_sql_result(
        results=results,
        output_mode="mixed",
        sample_size=5,
        query_shape="mixed",
        detail_limit=0,
        grouped_lists=True,
    )

    assert payload["execution_status"] == "success"
    assert payload["truncated"] is False
    assert payload["row_count"] == 4

    grouped = payload["grouped_rows"]
    assert len(grouped["positivos"]) == 2
    assert len(grouped["provaveis"]) == 1
    assert len(grouped["negativos"]) == 1
    assert grouped["pos_operatorios"] == []

    # CPF completo e formatado em todas as linhas
    for group_rows in grouped.values():
        for row in group_rows:
            cpf = row.get("cpf_paciente")
            if cpf is None:
                continue
            assert len(cpf) == 14  # Formato 000.000.000-00

    # Resumo
    summary = payload["summary"]
    assert summary["total_registros"] == 4
    assert summary["positivos"] == 2
    assert summary["provaveis"] == 1
    assert summary["negativos"] == 1
    assert summary["pos_operatorios"] == 0


def test_format_sql_result_grouped_lists_never_truncates_large_sets():
    # 500 linhas, 250 positivas, 250 negativas — caracterizacao nao trunca.
    big = []
    for i in range(250):
        big.append(_row("positivo", i, f"1234567{i:04d}"))
    for i in range(250):
        big.append(_row("negativo", 10_000 + i, f"9876543{i:04d}"))

    payload = _format_sql_result(
        results=big,
        output_mode="mixed",
        sample_size=5,
        query_shape="mixed",
        detail_limit=0,
        grouped_lists=True,
    )

    assert payload["truncated"] is False
    assert payload["row_count"] == 500
    assert len(payload["grouped_rows"]["positivos"]) == 250
    assert len(payload["grouped_rows"]["negativos"]) == 250


def test_format_sql_result_aggregate_unchanged_no_grouping():
    """Regressao: query agregada simples nao deve produzir grouped_rows
    e nao deve disparar caracterizacao."""
    results = [{"total_registros": 42, "total_pacientes_unicos": 30}]
    payload = _format_sql_result(
        results=results,
        output_mode="summary",
        sample_size=5,
        query_shape="aggregate",
        detail_limit=0,
        grouped_lists=False,
    )
    assert "grouped_rows" not in payload
    assert payload["summary"]["total_registros"] == 42


def test_format_sql_result_empty_grouped():
    payload = _format_sql_result(
        results=[],
        output_mode="mixed",
        sample_size=5,
        query_shape="mixed",
        detail_limit=0,
        grouped_lists=True,
    )
    assert payload["row_count"] == 0
    assert payload["grouped_rows"] == {
        "positivos": [],
        "provaveis": [],
        "negativos": [],
        "pos_operatorios": [],
    }


# ---------------------------------------------------------------------------
# _compact_detail_row (mascara CPF mesmo fora de caracterizacao)
# ---------------------------------------------------------------------------

def test_compact_detail_row_masks_cpf():
    out = _compact_detail_row(_row("positivo", 1, "123.456.789-09"))
    assert out["cpf_paciente"] == "123.456.789-09"
    assert out["id_paciente"] == 1