from datetime import datetime, timedelta
import zoneinfo

BRASILIA = zoneinfo.ZoneInfo("America/Sao_Paulo")


def get_dates() -> dict:
    """Returns date strings for hoje, ontem, amanha and semana_passada in Brasilia timezone."""
    hoje = datetime.now(tz=BRASILIA)
    ontem = hoje - timedelta(days=1)
    amanha = hoje + timedelta(days=1)
    semana_passada = hoje - timedelta(days=7)

    return {
        "hoje": hoje.strftime("%Y-%m-%d"),
        "ontem": ontem.strftime("%Y-%m-%d"),
        "amanha": amanha.strftime("%Y-%m-%d"),
        "semana_passada": semana_passada.strftime("%Y-%m-%d"),
    }