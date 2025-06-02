import pandas as pd
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.knowledge_model import KnowledgeBasedRecommender
from models.hybrid import HybridRecommender
from evaluation.metrics import evaluate_offline
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training_pipeline():
    logger.info("Iniciando pipeline de treinamento")
    
    # Garantir que a pasta models existe
    os.makedirs('models', exist_ok=True)

    logger.info("Treinando modelo baseado em conteúdo") 
    cb_model = ContentBasedRecommender()
    cb_model.load_data()
    cb_model.train()
    # Add this line to save the content-based model
    joblib.dump(cb_model, 'models/content_based_model.pkl')
    
    #logger.info("Treinando modelo colaborativo")
    #collab_model = CollaborativeRecommender()
    #collab_model.train()
    #joblib.dump(collab_model, 'models/collaborative_model.pkl')

    logger.info("Treinando modelo baseado em conhecimento") 
    kb_model = KnowledgeBasedRecommender()
    kb_model.load_data()
    kb_model.train()
    joblib.dump(kb_model, 'models/knowledge_model.pkl')
    
    logger.info("Treinando modelo híbrido")
    hybrid_model = HybridRecommender()
    hybrid_model.load_models()
    
    # Ensure hybrid model has materials data
    if hybrid_model.materiais is None:
        hybrid_model.materiais = pd.read_csv('data/processed/materiais_processed.csv')
    
    joblib.dump(hybrid_model, 'models/hybrid_model.pkl')
    
    logger.info("Pipeline concluído com sucesso!")

   
if __name__ == '__main__':
    run_training_pipeline()