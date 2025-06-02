import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error

def calculate_precision_recall(test_interactions, recommendations, k=10):
    """Calcula Precision@K e Recall@K"""
    tp = 0
    relevant = len(test_interactions)
    recommended = set(recommendations[:k])
    
    for interaction in test_interactions:
        if interaction in recommended:
            tp += 1
            
    precision = tp / k
    recall = tp / relevant if relevant > 0 else 0
    return precision, recall

def calculate_f1(precision, recall):
    """Calcula F1-Score"""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def calculate_rmse(actual_ratings, predicted_ratings):
    """Calcula RMSE"""
    return np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))

def calculate_coverage(recommendations, catalog_size):
    """Calcula cobertura do catálogo"""
    unique_items = set()
    for rec_list in recommendations.values():
        unique_items.update(rec_list)
    return len(unique_items) / catalog_size

def calculate_diversity(recommendations, similarity_matrix):
    """Calcula diversidade das recomendações"""
    diversity_scores = []
    for rec_list in recommendations.values():
        if len(rec_list) < 2:
            continue
        list_diversity = 0
        count = 0
        for i in range(len(rec_list)):
            for j in range(i+1, len(rec_list)):
                item_i = rec_list[i]
                item_j = rec_list[j]
                similarity = similarity_matrix[item_i][item_j]
                list_diversity += 1 - similarity
                count += 1
        if count > 0:
            diversity_scores.append(list_diversity / count)
    return np.mean(diversity_scores) if diversity_scores else 0

def evaluate_offline(model, test_data, catalog_size, similarity_matrix=None):
    """Avaliação offline completa"""
    results = {
        'precision@5': [],
        'recall@5': [],
        'f1@5': [],
        'precision@10': [],
        'recall@10': [],
        'f1@10': [],
        'rmse': [],
        'coverage': [],
        'diversity': []
    }
    
    all_recommendations = {}
    actual_ratings = []
    predicted_ratings = []
    
    for user_id, test_interactions in test_data.groupby('id_aluno'):
        # Gerar recomendações
        recommendations = model.recommend(user_id, 10)
        
        # Convert recommendations to list of IDs
        # Handle both tuple format (id_material, score) and dict format {'id_material': id, 'score': score}
        if isinstance(recommendations[0], tuple):
            rec_ids = [rec[0] for rec in recommendations]  # Extract first element of tuple
            rec_scores = [rec[1] for rec in recommendations]  # Extract second element of tuple
        else:
            rec_ids = [rec['id_material'] for rec in recommendations]
            rec_scores = [rec['score'] for rec in recommendations]
            
        all_recommendations[user_id] = rec_ids
        
        # Calcular métricas por usuário
        test_items = test_interactions['id_material'].tolist()
        
        prec5, rec5 = calculate_precision_recall(test_items, rec_ids, 5)
        f1_5 = calculate_f1(prec5, rec5)
        
        prec10, rec10 = calculate_precision_recall(test_items, rec_ids, 10)
        f1_10 = calculate_f1(prec10, rec10)
        
        # Coletar para RMSE
        for _, row in test_interactions.iterrows():
            actual = row['score']
            if isinstance(recommendations[0], tuple):
                pred = next(
                    (score for id_mat, score in recommendations if id_mat == row['id_material']),
                    None
                )
            else:
                pred = next(
                    (rec['score'] for rec in recommendations if rec['id_material'] == row['id_material']),
                    None
                )
            
            if pred is not None:
                actual_ratings.append(actual)
                predicted_ratings.append(pred)
        
        # Armazenar resultados
        results['precision@5'].append(prec5)
        results['recall@5'].append(rec5)
        results['f1@5'].append(f1_5)
        results['precision@10'].append(prec10)
        results['recall@10'].append(rec10)
        results['f1@10'].append(f1_10)
    
    # Calcular métricas agregadas
    results['rmse'] = calculate_rmse(actual_ratings, predicted_ratings)
    results['coverage'] = calculate_coverage(all_recommendations, catalog_size)
    
    if similarity_matrix is not None:
        results['diversity'] = calculate_diversity(all_recommendations, similarity_matrix)
    
    # Calcular médias
    for key in ['precision@5', 'recall@5', 'f1@5', 'precision@10', 'recall@10', 'f1@10']:
        results[key] = np.mean(results[key])
    
    return results