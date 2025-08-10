import pandas as pd

def add_bmi_feature(df):
    """
    체중(weight)과 키(height)를 사용하여 BMI를 계산하는 파생 변수를 추가합니다.
    """
    if df is None:
        return None

    # 키(cm)를 미터(m)로 변환
    df['height_m'] = df['height'] / 100
    
    # BMI 계산: weight(kg) / height(m)^2
    df['BMI'] = df['weight'] / (df['height_m'] ** 2)
    
    # 사용한 중간 컬럼 삭제
    df.drop('height_m', axis=1, inplace=True)
    
    return df

def one_hot_encode(df):
    """
    범주형 변수를 원-핫 인코딩하는 함수.
    """
    if df is None:
        return None

    # 인코딩할 범주형 컬럼 리스트
    categorical_cols = ['gender', 'activity', 'smoke_status', 'sleep_pattern',
                        'medical_history', 'family_medical_history', 'edu_level']
    
    # pd.get_dummies()를 사용하여 원-핫 인코딩
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df