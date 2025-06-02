import pandas as pd
import joblib
import logging
import os
from evaluation.metrics import evaluate_offline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model_name, model, test_data, catalog_size, similarity_matrix=None):
    """Evaluate a single model and save its metrics"""
    
    logger.info(f"Evaluating {model_name}...")
    
    metrics = evaluate_offline(
        model,
        test_data,
        catalog_size,
        similarity_matrix
    )
    
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    # Save metrics to individual report
    with open(f'reports/{model_name}_metrics.txt', 'w') as f:
        f.write(f"{model_name} - Métricas de Avaliação\n")
        f.write("===============================\n\n")
        f.write(f"Precision@5: {metrics['precision@5']:.4f}\n")
        f.write(f"Recall@5: {metrics['recall@5']:.4f}\n")
        f.write(f"F1@5: {metrics['f1@5']:.4f}\n")
        f.write(f"Precision@10: {metrics['precision@10']:.4f}\n")
        f.write(f"Recall@10: {metrics['recall@10']:.4f}\n")
        f.write(f"F1@10: {metrics['f1@10']:.4f}\n")
        f.write(f"RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"Coverage: {metrics['coverage']:.4f}\n")
        f.write(f"Diversity: {metrics.get('diversity', 'N/A')}\n")
    
    logger.info(f"Saved evaluation metrics for {model_name}")
    return metrics

def main():
    # Load test data
    try:
        test_data = pd.read_csv('data/processed/interacoes_processed.csv')
        if len(test_data) == 0:
            raise ValueError("Dataset de interações está vazio")
            
        # Get only user IDs that exist in alunos dataset
        alunos = pd.read_csv('data/processed/alunos_processed.csv')
        test_data = test_data[test_data['id_aluno'].isin(alunos['id_aluno'])]
        
        train_size = int(0.8 * len(test_data))
        test_data = test_data[train_size:]
        
        catalog_size = len(pd.read_csv('data/processed/materiais_processed.csv'))
        
        # Load content model and ensure data is loaded
        content_model = joblib.load('models/content_based_model.pkl')
        content_model.load_data()  # Explicitly load data
        similarity_matrix = content_model.similarity_matrix
        
        # Dictionary to store all metrics
        all_metrics = {}
        
        # Evaluate Content-Based Model
       # all_metrics['content_based'] = evaluate_model(
       #     'content_based',
       #     content_model,
       #     test_data,
       #     catalog_size,
       #     similarity_matrix
       # )
        
        # Evaluate Collaborative Model
       # collab_model = joblib.load('models/collaborative_model.pkl')
       # all_metrics['collaborative'] = evaluate_model(
       #     'collaborative',
       #     collab_model,
       #     test_data,
       #     catalog_size
       # )
        
        # Evaluate Knowledge-Based Model
        kb_model = joblib.load('models/knowledge_model.pkl')
        all_metrics['knowledge_based'] = evaluate_model(
            'knowledge_based',
            kb_model,
            test_data,
            catalog_size
        )
        
        # Evaluate Hybrid Model
        hybrid_model = joblib.load('models/hybrid_model.pkl')
        all_metrics['hybrid'] = evaluate_model(
            'hybrid',
            hybrid_model,
            test_data,
            catalog_size,
            similarity_matrix
        )
        
        # Generate comparative report
        with open('reports/comparative_metrics.txt', 'w') as f:
            f.write("Comparative Model Evaluation\n")
            f.write("==========================\n\n")
            
            metrics = ['precision@10', 'recall@10', 'f1@10', 'coverage', 'diversity']
            models = ['content_based', 'collaborative', 'knowledge_based', 'hybrid']
            
            # Header
            f.write(f"{'Metric':<15}")
            for model in models:
                f.write(f"{model:<15}")
            f.write("\n")
            f.write("-" * 75 + "\n")
            
            # Metrics
            for metric in metrics:
                f.write(f"{metric:<15}")
                for model in models:
                    value = all_metrics[model].get(metric, 'N/A')
                    if isinstance(value, float):
                        f.write(f"{value:.4f}".ljust(15))
                    else:
                        f.write(f"{value}".ljust(15))
                f.write("\n")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()