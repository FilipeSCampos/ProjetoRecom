import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os
from pathlib import Path

def create_common_features(alunos, materiais):
    """Cria features compatíveis entre alunos e materiais"""
    encoders = {}
    
    # 1. Área Principal
    curso_area_map = {
        'Engenharia de Software': 'Programação',
        'Ciência de Dados': 'IA', 
        'Engenharia de Computação': 'Hardware',
        'SI': 'Gestão'
    }
    alunos['area_principal'] = alunos['curso'].map(curso_area_map)
    
    # 2. One-Hot Encoding para Áreas
    area_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    area_encoder.fit(pd.concat([
        alunos[['area_principal']].rename(columns={'area_principal':'area'}),
        materiais[['area']]
    ]))
    encoders['area'] = area_encoder
    
    # Aplicar encoding
    area_cols = [f'area_{a}' for a in area_encoder.categories_[0]]
    alunos[area_cols] = area_encoder.transform(
        alunos[['area_principal']].rename(columns={'area_principal':'area'})
    )
    materiais[area_cols] = area_encoder.transform(materiais[['area']])
    
    # 3. Nível de Dificuldade
    nivel_map = {'Iniciante':0, 'Intermediário':1, 'Avançado':2}
    materiais['nivel'] = materiais['nivel'].map(nivel_map)
    alunos['nivel'] = alunos['periodo'].apply(lambda x: 0 if x < 3 else (1 if x < 6 else 2))
    
    # Normalização
    nivel_scaler = MinMaxScaler()
    alunos['nivel_norm'] = nivel_scaler.fit_transform(alunos[['nivel']])
    materiais['nivel_norm'] = nivel_scaler.transform(materiais[['nivel']])
    encoders['nivel'] = nivel_scaler
    
    # 4. Tipo de Material
    tipo_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    tipo_encoder.fit(materiais[['tipo']])
    encoders['tipo'] = tipo_encoder
    
    tipo_cols = [f'tipo_{t}' for t in tipo_encoder.categories_[0]]
    materiais[tipo_cols] = tipo_encoder.transform(materiais[['tipo']])
    for col in tipo_cols:
        alunos[col] = 0.5  # Valor neutro
        
    return alunos, materiais, encoders

def preprocess_data():
    # 1. Carregar dados
    alunos = pd.read_csv('data/raw/alunos.csv')
    materiais = pd.read_csv('data/raw/materiais.csv') 
    interacoes = pd.read_csv('data/raw/interacoes.csv')
    
    # 2. Criar features comuns
    alunos_processed, materiais_processed, encoders = create_common_features(alunos, materiais)
    
    # 3. Processar interações
    interacoes['score'] = (
        0.4 * (interacoes['avaliacao'] / 5) + 
        0.3 * (interacoes['duracao_minutos'] / 300) + 
        0.3 * interacoes['tipo_interacao'].map({'visualizacao':0.5, 'leitura':1.0})
    )
    
    # 4. Salvar dados
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/encoders', exist_ok=True)
    
    alunos_processed.to_csv('data/processed/alunos_processed.csv', index=False)
    materiais_processed.to_csv('data/processed/materiais_processed.csv', index=False)
    interacoes.to_csv('data/processed/interacoes_processed.csv', index=False)
    
    # 5. Salvar encoders
    for name, encoder in encoders.items():
        joblib.dump(encoder, f'models/encoders/{name}_encoder.pkl')
    
    # 6. Verificação segura
    common_cols = list(set(alunos_processed.columns) & set(materiais_processed.columns))
    print("\nFeatures comuns:")
    print("Alunos:", alunos_processed[common_cols].columns.tolist())
    print("Materiais:", materiais_processed[common_cols].columns.tolist())
    
    return alunos_processed, materiais_processed, interacoes

if __name__ == '__main__':
    print("Iniciando pré-processamento...")
    try:
        alunos, materiais, interacoes = preprocess_data()
        print("\nPré-processamento concluído com sucesso!")
        print(f"Alunos: {len(alunos)} registros")
        print(f"Materiais: {len(materiais)} registros") 
        print(f"Interações: {len(interacoes)} registros")
    except Exception as e:
        print(f"\nErro durante o pré-processamento: {str(e)}")
        raise