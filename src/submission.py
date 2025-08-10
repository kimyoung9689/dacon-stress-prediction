# 제출 파일 만드는 코드



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def make_submission_file(model, X_test_path, sample_submission_path, output_filename):
    """
    학습된 모델을 사용하여 예측값을 생성하고 제출 파일을 만듭니다.
    
    Args:
        model: 예측에 사용할 학습된 모델.
        X_test_path (str): 테스트 데이터 파일 경로.
        sample_submission_path (str): 샘플 제출 파일 경로.
        output_filename (str): 생성할 제출 파일 이름.
    """
    # 1. 테스트 데이터 불러오기
    X_test = pd.read_csv(X_test_path)
    
    # 2. 테스트 데이터로 예측
    final_predictions = model.predict(X_test.drop('ID', axis=1))
    
    # 3. 샘플 제출 파일 불러오기
    submission_df = pd.read_csv(sample_submission_path)
    
    # 4. 예측 결과를 제출 파일 형식에 맞추기
    submission_df['stress_score'] = final_predictions
    
    # 5. 제출 파일 저장
    submission_df.to_csv(output_filename, index=False)
    
    print(f"제출 파일 '{output_filename}'이 성공적으로 생성되었습니다.")

# 이 함수는 다른 코드(예: model_training.ipynb)에서 아래와 같이 호출할 수 있습니다.
# (이 부분은 파일에 저장하지 않아도 됩니다. 사용법 예시입니다.)
#
# # model_training.ipynb에서...
# from submission import make_submission_file
#
# # 모델 학습 후...
# # model = ...
#
# # make_submission_file 함수 호출
# make_submission_file(model, 
#                      '../preprocessed_data/X_test.csv', 
#                      '../data/sample_submission.csv', 
#                      'submission_baseline.csv')