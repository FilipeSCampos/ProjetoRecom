import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class ContentBasedRecommender:
    def __init__(self):
        self.alunos = None
        self.materiais = None
        self.similarity_matrix = None
        self.feature_cols = None
        
    def load_data(self):
        """Carrega os dados processados"""
        try:
            self.alunos = pd.read_csv('data/processed/alunos_processed.csv')
            self.materiais = pd.read_csv('data/processed/materiais_processed.csv')
            if len(self.alunos) == 0:
                raise ValueError("Dataset de alunos está vazio")
            if len(self.materiais) == 0:
                raise ValueError("Dataset de materiais está vazio")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar dados: {str(e)}")

    def train(self):
        """Treina o modelo e salva em disco"""
        if self.alunos is None or self.materiais is None:
            self.load_data()
            
        # Identificar colunas comuns
        aluno_cols = set(self.alunos.columns) - {'id_aluno', 'nome', 'curso', 'periodo', 
                                            'disciplinas_cursadas', 'areas_interesse'}
        material_cols = set(self.materiais.columns) - {'id_material', 'titulo', 'tipo', 
                                                    'area', 'nivel', 'descricao', 'autor'}
        common_cols = list(aluno_cols & material_cols)
        
        if not common_cols:
            common_cols = ['nivel_norm']  # Garante pelo menos uma coluna
        
        # Calcular similaridade
        self.similarity_matrix = cosine_similarity(
            self.alunos[common_cols].values, 
            self.materiais[common_cols].values
        )
        self.feature_cols = common_cols
        
        # Save the model instance directly instead of creating a dictionary
        os.makedirs('models', exist_ok=True)
        joblib.dump(self, 'models/content_based_model.pkl')
        print("Modelo baseado em conteúdo treinado e salvo com sucesso!")

    def load_model(self):
        """Carrega o modelo salvo"""
        # Load the entire model instance
        loaded_model = joblib.load('models/content_based_model.pkl')
        self.similarity_matrix = loaded_model.similarity_matrix
        self.feature_cols = loaded_model.feature_cols
        self.alunos = loaded_model.alunos
        self.materiais = loaded_model.materiais
        
    def recommend(self, aluno_id, top_k=10):
        """Gera recomendações para um aluno"""
        if self.similarity_matrix is None:
            self.load_model()
            
        if self.alunos is None:
            self.load_data()
            
        try:
            aluno_idx = self.alunos[self.alunos['id_aluno'] == aluno_id].index[0]
        except IndexError:
            raise ValueError(f"Aluno ID {aluno_id} não encontrado no dataset")
            
        similarity_scores = self.similarity_matrix[aluno_idx]
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]
        
        return [
            (self.materiais.iloc[idx]['id_material'], similarity_scores[idx])
            for idx in top_indices
        ]