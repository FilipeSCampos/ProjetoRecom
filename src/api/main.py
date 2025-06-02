from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from models.hybrid import HybridRecommender
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sistema de Recomendação Híbrido",
    description="API para recomendar materiais educacionais",
    version="1.0.0"
)

# Carregar recomendador
recommender = HybridRecommender()
try:
    recommender.load_models()
    logger.info("Modelos carregados com sucesso!")
except Exception as e:
    logger.error(f"Erro ao carregar modelos: {str(e)}")

class RecommendationRequest(BaseModel):
    aluno_id: int
    top_k: int = 10

class MaterialResponse(BaseModel):
    id_material: int
    titulo: str
    score: float

@app.post("/recommend", response_model=list[MaterialResponse])
def get_recommendations(request: RecommendationRequest):
    try:
        recommendations = recommender.recommend(
            request.aluno_id, 
            request.top_k
        )
        return [
            MaterialResponse(
                id_material=rec['id_material'],
                titulo=rec['titulo'],
                score=rec['score']
            )
            for rec in recommendations
        ]
    except Exception as e:
        logger.exception("Erro na geração de recomendações")
        raise HTTPException(
            status_code=500, 
            detail=f"Erro interno: {str(e)}"
        )

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)