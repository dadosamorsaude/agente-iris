import asyncio
from app.agent.evaluator import evaluate_response
from app.services.evaluation_store import save_evaluation

async def force_log():
    user_id = "user_valida_hoje"
    pergunta = "Teste de auditoria final"
    resposta = "A conformidade de BH foi de 100%."
    mock_athena = [{"sql": "SELECT 1", "results": [{"val": 1}]}]
    
    print("1. Avaliando...")
    eval_res = await evaluate_response(pergunta, resposta, mock_athena)
    print(f"2. Score: {eval_res.get('score')}")
    
    print("3. Salvando...")
    await save_evaluation(user_id, pergunta, resposta, mock_athena, eval_res)
    print("4. Sucesso! Registro deve estar no banco.")

if __name__ == "__main__":
    asyncio.run(force_log())
