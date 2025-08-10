import pandas as pd

def make_submission_file(model, test_df, sample_submission_path, output_filename):
    """
    학습된 모델을 사용하여 예측값을 생성하고 제출 파일을 만듭니다.

    Args:
        model: 예측에 사용할 학습된 모델.
        test_df (pd.DataFrame): 전처리된 테스트 데이터.
        sample_submission_path (str): 샘플 제출 파일 경로.
        output_filename (str): 생성할 제출 파일 이름.
    """
    if model is None or test_df is None:
        print("Error: 모델 또는 테스트 데이터가 유효하지 않습니다.")
        return

    try:
        # 'ID' 컬럼 제외
        X_test = test_df.drop('ID', axis=1)
        
        # 예측
        final_predictions = model.predict(X_test)
        
        # 샘플 제출 파일 불러오기
        submission_df = pd.read_csv(sample_submission_path)
        
        # 예측 결과를 제출 파일 형식에 맞추기
        submission_df['stress_score'] = final_predictions
        
        # 제출 파일 저장
        submission_df.to_csv(output_filename, index=False)
        
        print(f"제출 파일 '{output_filename}'이 성공적으로 생성되었습니다.")
        
    except Exception as e:
        print(f"제출 파일 생성 중 오류가 발생했습니다: {e}")