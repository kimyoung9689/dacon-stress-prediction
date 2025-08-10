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
    
    # 1. 'medical_history', 'family_medical_history' 결측치 처리
    df['medical_history'] = df['medical_history'].fillna('unknown')
    df['family_medical_history'] = df['family_medical_history'].fillna('unknown')
    
    # 2. 'edu_level' 결측치 처리
    df['edu_level'] = df['edu_level'].fillna('unknown')
    
    # 3. 'mean_working' 결측치 처리
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

if __name__ == '__main__':
    # 예시: 이 파일 자체를 실행했을 때 동작하는 코드
    # 실제로는 main.py에서 이 함수들을 호출하게 됩니다.
    
    # 데이터 불러오기
    # 주의: 이 파일은 src 폴더에 있으므로, data 폴더는 한 단계 상위 디렉터리에 있습니다.
    train_file_path = '../data/train.csv'
    test_file_path = '../data/test.csv'
    
    train_df_raw = load_data(train_file_path)
    test_df_raw = load_data(test_file_path)

    if train_df_raw is not None and test_df_raw is not None:
        # 결측치 처리
        train_df_processed = handle_missing_values(train_df_raw.copy())
        test_df_processed = handle_missing_values(test_df_raw.copy())

        # 이상치 처리
        train_df_processed = handle_outliers(train_df_processed)
        test_df_processed = handle_outliers(test_df_processed)

        print("전처리 완료! 처리된 데이터의 결측치 확인:")
        print("Train data missing values after preprocessing:")
        print(train_df_processed.isnull().sum())
        
        print("\nTest data missing values after preprocessing:")
        print(test_df_processed.isnull().sum())
    else:
        print("데이터 로드에 실패하여 전처리 단계를 건너뜁니다.")