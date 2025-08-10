import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model(train_df):
    """
    RandomForestRegressor 모델을 학습시키는 함수.
    """
    if train_df is None:
        print("Error: 훈련 데이터가 유효하지 않습니다.")
        return None
        
    # 피처(X)와 타겟(y) 분리
    X = train_df.drop(['ID', 'stress_score'], axis=1)
    y = train_df['stress_score']
    
    # 모델 초기화 및 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y.values.ravel())
    
    return model

def predict_and_evaluate(model, X_train, y_train):
    """
    모델의 성능을 평가하는 함수.
    """
    if model is None or X_train is None or y_train is None:
        print("Error: 모델 또는 데이터가 유효하지 않습니다.")
        return None
    
    # 훈련 데이터와 검증 데이터로 나누기
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 검증 데이터로 예측
    y_pred = model.predict(X_val_split)
    
    # 모델 성능 평가 (RMSE)
    rmse = np.sqrt(mean_squared_error(y_val_split, y_pred))
    
    print(f"모델의 검증 데이터 RMSE: {rmse:.4f}")
    
    return rmse