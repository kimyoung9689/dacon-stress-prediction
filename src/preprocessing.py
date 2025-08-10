import pandas as pd
import numpy as np

def load_data(file_path):
    """
    주어진 경로에서 CSV 파일을 읽어와 DataFrame으로 반환하는 함수.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} 파일이 존재하지 않습니다. 경로를 확인해주세요.")
        return None

def handle_missing_values(df):
    """
    결측치를 처리하는 함수.
    'medical_history', 'family_medical_history', 'edu_level'은 'unknown'으로 채우고,
    'mean_working'은 'smoke_status'와 'edu_level' 그룹별 중앙값으로 채웁니다.
    """
    if df is None:
        return None
    
    df['medical_history'] = df['medical_history'].fillna('unknown')
    df['family_medical_history'] = df['family_medical_history'].fillna('unknown')
    df['edu_level'] = df['edu_level'].fillna('unknown')
    
    grouped_median = df.groupby(['smoke_status', 'edu_level'])['mean_working'].transform('median')
    df['mean_working'] = df['mean_working'].fillna(grouped_median)
    
    if df['mean_working'].isnull().sum() > 0:
        total_median = df['mean_working'].median()
        df['mean_working'] = df['mean_working'].fillna(total_median)
        
    return df

def handle_outliers(df, column='bone_density'):
    """
    특정 컬럼의 이상치를 처리하는 함수.
    'bone_density' 컬럼의 음수 값을 0으로 변경합니다.
    """
    if df is None:
        return None
        
    df.loc[df[column] < 0, column] = 0
    
    return df

def one_hot_encode(df):
    """
    범주형 변수를 원-핫 인코딩하는 함수.
    """
    if df is None:
        return None
    
    categorical_cols = ['gender', 'activity', 'smoke_status', 'sleep_pattern',
                        'medical_history', 'family_medical_history', 'edu_level']
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

if __name__ == '__main__':
    # 예시: 이 파일 자체를 실행했을 때 동작하는 코드
    print("이 코드는 main.py에서 호출되도록 설계되었습니다.")
    print("개별 함수를 테스트하려면 여기에서 실행할 수 있습니다.")
