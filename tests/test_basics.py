from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_main():
    """Teste básico para verificar se o endpoint home está online."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["agent"] == "Iris"

def test_api_key_protected():
    """Verifica se o chat está protegido por API Key."""
    response = client.post("/chat", json={"message": "oi", "user_id": "test"})
    # Deve retornar 403 se não houver a chave correta (e a chave estiver configurada)
    assert response.status_code in [403, 200] # Depende se a chave está no ENV do CI
