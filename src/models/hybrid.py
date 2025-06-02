import pandas as pd
import numpy as np
import joblib
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender
from .knowledge_model import KnowledgeBasedRecommender

class HybridRecommender:
    def __init__(self):
        self.content_model = None
        self.collab_model = None
        self.knowledge_model = None
        self.materiais = None
        
    def load_models(self):
        """Carrega os modelos treinados"""
        try:
            # Load base models
            self.content_model = ContentBasedRecommender()
            self.content_model.load_model()
            
            self.collaborative_model = joblib.load('models/collaborative_model.pkl')
            self.knowledge_model = joblib.load('models/knowledge_model.pkl')
            
            # Load materials data
            self.materiais = pd.read_csv('data/processed/materiais_processed.csv')
            
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelos: {str(e)}")
    def recommend(self, aluno_id, top_k=10):
        if not all([self.content_model, self.collab_model, self.knowledge_model]):
            self.load_models()
        
        # Obter recomendações de cada modelo
        content_recs = self.content_model.recommend(aluno_id, top_k*3)
        material_ids = [rec[0] for rec in content_recs]
        
        # Modificação aqui: carrega o modelo corretamente
        collab_model = CollaborativeRecommender().load_model()
        collab_recs = collab_model.recommend(aluno_id, material_ids, top_k*3)
        
        knowledge_recs = self.knowledge_model.recommend(aluno_id, top_k*3)
        knowledge_ids = [rec['id_material'] for rec in knowledge_recs]
            
        # Combinar scores
        combined_scores = {}
        
        # Content-based scores
        for material_id, score in content_recs:
            combined_scores[material_id] = score * 0.4
            
        # Collaborative scores
        for material_id, score in collab_recs:
            if material_id in combined_scores:
                combined_scores[material_id] += score * 0.4
            else:
                combined_scores[material_id] = score * 0.4
                
        # Knowledge-based scores
        for material_id in knowledge_ids:
            if material_id in combined_scores:
                combined_scores[material_id] += 0.2
            else:
                combined_scores[material_id] = 0.2
                
        # Ordenar e selecionar top_k
        sorted_recs = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Adicionar detalhes dos materiais
        result = []
        material_details = self.materiais.set_index('id_material')
        for material_id, score in sorted_recs:
            if material_id in material_details.index:
                result.append({
                    'id_material': material_id,
                    'score': score,
                    'titulo': material_details.loc[material_id, 'titulo']
                })
                
        return result

