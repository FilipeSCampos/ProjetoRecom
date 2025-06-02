import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

class KnowledgeBasedRecommender:
    def __init__(self):
        self.materiais = None
        self.alunos = None
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """Carrega os dados processados"""
        try:
            self.materiais = pd.read_csv('data/processed/materiais_processed.csv')
            self.alunos = pd.read_csv('data/processed/alunos_processed.csv')
            
            if len(self.materiais) == 0:
                raise ValueError("Dataset de materiais está vazio")
            if len(self.alunos) == 0:
                raise ValueError("Dataset de alunos está vazio")
                
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar dados: {str(e)}")
            
    def train(self):
        """Prepara o modelo baseado em conhecimento"""
        if self.materiais is None or self.alunos is None:
            self.load_data()
            
        # Normalize numerical features if any exist
        numeric_cols = self.materiais.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.materiais[numeric_cols] = self.scaler.fit_transform(self.materiais[numeric_cols])
    
    def recommend(self, aluno_id, top_k=10):
        """Gera recomendações baseadas em regras para um aluno"""
        try:
            if self.materiais is None or self.alunos is None:
                self.load_data()
                
            # Get student data
            aluno = self.alunos[self.alunos['id_aluno'] == aluno_id].iloc[0]
            
            # Calculate relevance score for each material
            scores = []
            for _, material in self.materiais.iterrows():
                score = self._calculate_relevance(aluno, material)
                scores.append((material['id_material'], score))
            
            # Sort by score and return top k
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            # Ensure we have recommendations
            if not sorted_scores:
                # If no recommendations, return random materials
                random_materials = self.materiais.sample(n=min(top_k, len(self.materiais)))
                return [(row['id_material'], 0.5) for _, row in random_materials.iterrows()]
                
            return sorted_scores[:top_k]
            
        except Exception as e:
            # Return some default recommendations if there's an error
            random_materials = self.materiais.sample(n=min(top_k, len(self.materiais)))
            return [(row['id_material'], 0.5) for _, row in random_materials.iterrows()]
    
    def _calculate_relevance(self, aluno, material):
        """Calculate relevance score between student and material"""
        score = 0.0
        
        # Example rules (customize based on your specific needs)
        # Level matching
        if 'nivel' in material and 'nivel' in aluno:
            score += 1.0 if material['nivel'] == aluno['nivel'] else 0.0
            
        # Area matching
        if 'area' in material and 'areas_interesse' in aluno:
            areas_interesse = str(aluno['areas_interesse']).split(',')
            score += 1.0 if material['area'] in areas_interesse else 0.0
            
        # Normalize score
        score = score / 2.0  # Divide by number of rules
        
        return score
        
    def load_model(self):
        """Load saved model state"""
        loaded_model = joblib.load('models/knowledge_model.pkl')
        self.materiais = loaded_model.materiais
        self.alunos = loaded_model.alunos
        self.scaler = loaded_model.scaler
        return self


