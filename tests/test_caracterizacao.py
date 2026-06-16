"""Testes para o modo caracterizacao de pacientes e mascaramento de CPF."""
from app.agent.specialists.sql_analyst import (
    _mask_cpf,
    _classify_group,
    _resolve_query_shape,
    _format_sql_result,
    _compact_detail_row,
    _group_rows_by_classification,
    extract_laterality,
)
from app.services.intent import detect_grouped_lists, detect_intent


# _mask_cpf
def test_mask_cpf_none():
    assert _mask_cpf(None) is None

def test_mask_cpf_empty():
    assert _mask_cpf("") is None
    assert _mask_cpf("   ") is None

def test_mask_cpf_formatted():
    assert _mask_cpf("123.456.789-09") == "123.456.789-09"

def test_mask_cpf_digits_only():
    assert _mask_cpf("12345678909") == "123.456.789-09"

def test_mask_cpf_with_noise():
    assert _mask_cpf("CPF: 123.456.789-09 (paciente)") == "123.456.789-09"

def test_mask_cpf_too_short():
    assert _mask_cpf("9") == "9"
    assert _mask_cpf("89") == "89"

def test_mask_cpf_too_long():
    assert _mask_cpf("9912345678909") == "9912345678909"


# _classify_group
def test_classify_group_canonical():
    assert _classify_group("positivo") == "positivos"
    assert _classify_group("PROVAVEL") == "provaveis"
    assert _classify_group("negativo") == "negativos"
    assert _classify_group("Pos-operatorio") == "pos_operatorios"
    assert _classify_group("pos operatorio") == "pos_operatorios"

def test_classify_group_empty():
    assert _classify_group(None) is None
    assert _classify_group("") is None


# detect_grouped_lists / detect_intent
def test_detect_grouped_lists_positives():
    assert detect_grouped_lists("Caracterize os pacientes de abril")
    assert detect_grouped_lists("Liste os positivos")
    assert detect_grouped_lists("estratifique os pacientes")

def test_detect_grouped_lists_negatives():
    assert not detect_grouped_lists("ola")
    assert not detect_grouped_lists("Total de cirurgias")

def test_detect_intent_sets_grouped_flag():
    intent = detect_intent("Caracterize os pacientes de abril")
    assert intent["wants_grouped_lists"] is True
    assert intent["output_mode"] == "mixed"


# _resolve_query_shape
def test_resolve_query_shape_grouped_forces_mixed():
    out = _resolve_query_shape("Caracterize os pacientes de abril")
    assert out["query_shape"] == "mixed"
    assert out["limit"] == 0
    assert out.get("grouped_lists") is True

def test_resolve_query_shape_aggregate_stays_aggregate():
    # Termos so de agregacao, sem detail nem grouping.
    out = _resolve_query_shape("Qual o total de cirurgias?")
    assert out["query_shape"] == "aggregate"
    assert not out.get("grouped_lists")


# _group_rows_by_classification
def test_group_rows_buckets():
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


# helpers para _format_sql_result
def _row(cls, idp, cpf):
    return {
        "id_paciente": idp,
        "nome_paciente": f"Paciente {idp}",
        "cpf_paciente": cpf,
        "id_atendimento": idp * 100,
        "data_atendimento": "2026-04-15",
        "clinica": "Clinica X",
        "regional": "SP",
        "nome_profissional": "Dr Y",
        "classificacao": cls,
        "score": 0.9,
        "termo_detectado": "catarata",
        "trecho_evidencia": "trecho",
    }


def test_format_grouped_basic():
    results = [
        _row("positivo", 1, "12345678909"),
        _row("positivo", 2, "98765432100"),
        _row("provavel", 3, "11122233344"),
        _row("negativo", 4, "55566677788"),
    ]
    p = _format_sql_result(results, "mixed", 5, "mixed", 0, grouped_lists=True)
    assert p["truncated"] is False
    assert p["row_count"] == 4
    g = p["grouped_rows"]
    assert len(g["positivos"]) == 2
    assert len(g["provaveis"]) == 1
    assert len(g["negativos"]) == 1
    assert g["pos_operatorios"] == []
    for rows in g.values():
        for r in rows:
            cpf = r.get("cpf_paciente")
            if cpf is not None:
                assert len(cpf) == 14  # Formato 000.000.000-00
    s = p["summary"]
    assert s["total_registros"] == 4
    assert s["positivos"] == 2


def test_format_grouped_never_truncates():
    big = []
    for i in range(250):
        big.append(_row("positivo", i, f"1234567{i:04d}"))
    for i in range(250):
        big.append(_row("negativo", 10000 + i, f"9876543{i:04d}"))
    p = _format_sql_result(big, "mixed", 5, "mixed", 0, grouped_lists=True)
    assert p["truncated"] is False
    assert p["row_count"] == 500
    assert len(p["grouped_rows"]["positivos"]) == 250
    assert len(p["grouped_rows"]["negativos"]) == 250


def test_format_aggregate_no_grouping():
    results = [{"total_registros": 42, "total_pacientes_unicos": 30}]
    p = _format_sql_result(results, "summary", 5, "aggregate", 0, grouped_lists=False)
    assert "grouped_rows" not in p
    assert p["summary"]["total_registros"] == 42


def test_format_empty_grouped():
    p = _format_sql_result([], "mixed", 5, "mixed", 0, grouped_lists=True)
    assert p["row_count"] == 0
    assert p["grouped_rows"] == {
        "positivos": [], "provaveis": [], "negativos": [], "pos_operatorios": [],
    }


def test_compact_detail_row_masks_cpf():
    out = _compact_detail_row(_row("positivo", 1, "123.456.789-09"))
    assert out["cpf_paciente"] == "123.456.789-09"
    assert out["id_paciente"] == 1

def test_compact_detail_row_no_cpf():
    out = _compact_detail_row(_row("positivo", 1, None))
    assert out["cpf_paciente"] is None


# extract_laterality tests
def test_extract_laterality_both_eyes():
    assert extract_laterality({"termo_detectado": "catarata", "trecho_evidencia": "olho esquerdo", "anamnese": "cirurgia em Ambos os Olhos"}) == "AO"
    assert extract_laterality({"termo": "AO", "trecho_evidencia": "catarata"}) == "AO"
    assert extract_laterality({"termo": "ao", "trecho_evidencia": "catarata"}) is None
    assert extract_laterality({"conduta": "encaminhado ao especialista"}) is None
    assert extract_laterality({"observacao": "catarata bilateral"}) == "AO"
    assert extract_laterality({"prescricao": "colirio em A.O. de 4/4h"}) == "AO"

def test_extract_laterality_right_eye():
    assert extract_laterality({"termo": "catarata OD", "trecho_evidencia": "olho direito"}) == "OD"
    assert extract_laterality({"anamnese": "catarata em olho dir"}) == "OD"

def test_extract_laterality_left_eye():
    assert extract_laterality({"termo": "catarata OE", "trecho_evidencia": "olho esquerdo"}) == "OE"
    assert extract_laterality({"obs_atend_oftalmo": "catarata em olho esq"}) == "OE"

def test_extract_laterality_both_indicated_separately():
    assert extract_laterality({"conduta": "cirurgia no OD e OE"}) == "AO"
