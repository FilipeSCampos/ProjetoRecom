from surprise import SVD, Dataset, Reader
import pandas as pd
import joblib
import numpy as np

class CollaborativeRecommender:
    def __init__(self):
        self.model = None
        self.materiais = None
        self.interactions = None
        self.trainset = None
        
    def load_data(self):
        """Load both interactions and materials data"""
        self.interactions = pd.read_csv('data/processed/interacoes_processed.csv')
        self.materiais = pd.read_csv('data/processed/materiais_processed.csv')
        
        # Create Surprise reader object
        reader = Reader(rating_scale=(0, 5))
        
        # Load the data into the Surprise format
        data = Dataset.load_from_df(
            self.interactions[['id_aluno', 'id_material', 'score']], 
            reader
        )
        
        # Build the full trainset
        self.trainset = data.build_full_trainset()
        return self.trainset
    
    def train(self):
        """Treina o modelo colaborativo"""
        try:
            if self.trainset is None:
                self.load_data()
            
            # Train SVD model
            self.model = SVD(n_factors=100)
            self.model.fit(self.trainset)
            
        except Exception as e:
            raise RuntimeError(f"Erro no treinamento: {str(e)}")

    def predict(self, aluno_id, material_id):
        """Predict rating for a user-item pair"""
        try:
            # Convert raw ids to inner ids
            user_inner_id = self.trainset.to_inner_uid(aluno_id)
            item_inner_id = self.trainset.to_inner_iid(material_id)
            pred = self.model.predict(user_inner_id, item_inner_id)
            return pred.est
        except Exception:
            return self.model.default_prediction_

    def recommend(self, aluno_id, top_k=10):
        """Gera recomendações para um usuário"""
        if self.model is None:
            raise RuntimeError("Modelo não foi treinado")
            
        if self.materiais is None:
            self.load_data()
            
        try:
            # Get all material IDs
            all_materials = self.materiais['id_material'].tolist()
            
            # Predict ratings for all materials
            predictions = []
            for material_id in all_materials:
                pred_rating = self.predict(aluno_id, material_id)
                predictions.append((material_id, pred_rating))
            
            # Sort by predicted rating and return top N
            sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
            return sorted_preds[:top_k]
            
        except Exception as e:
            raise RuntimeError(f"Erro ao gerar recomendações: {str(e)}")

    def load_model(self):
        """Load saved model"""
        loaded_model = joblib.load('models/collaborative_model.pkl')
        self.model = loaded_model.model
        self.materiais = loaded_model.materiais
        self.interactions = loaded_model.interactions
        self.trainset = loaded_model.trainset
        return self