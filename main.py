import pandas as pd
import os
from src.preprocessing import load_data, handle_missing_values, handle_outliers
from src.feature_engineering import add_bmi_feature, one_hot_encode
from src.modeling import train_model
from src.submission import make_submission_file

def main():
    """
    전체 머신러닝 파이프라인을 실행하는 메인 함수.
    """
    print("1. 데이터 로딩 시작...")
    train_df = load_data('data/train.csv')
    test_df = load_data('data/test.csv')

    if train_df is None or test_df is None:
        print("데이터 로딩에 실패하여 종료합니다.")
        return

    print("2. 데이터 전처리 시작...")
    train_df = handle_missing_values(train_df.copy())
    test_df = handle_missing_values(test_df.copy())
    train_df = handle_outliers(train_df)
    test_df = handle_outliers(test_df)
    print("   - 전처리 완료.")

    print("3. 피처 엔지니어링 시작...")
    train_df = add_bmi_feature(train_df)
    test_df = add_bmi_feature(test_df)
    train_df = one_hot_encode(train_df)
    test_df = one_hot_encode(test_df)

    # train과 test 데이터프레임의 컬럼을 동일하게 맞춰주기 (원-핫 인코딩 후 발생할 수 있는 문제 해결)
    train_cols = list(train_df.drop('stress_score', axis=1).columns)
    test_cols = list(test_df.columns)
    
    missing_in_test = set(train_cols) - set(test_cols)
    for col in missing_in_test:
        test_df[col] = 0
    
    missing_in_train = set(test_cols) - set(train_cols)
    for col in missing_in_train:
        train_df[col] = 0
    
    test_df = test_df[train_df.drop('stress_score', axis=1).columns]
    
    print("   - 피처 엔지니어링 완료.")

    print("4. 모델 학습 시작...")
    model = train_model(train_df)
    print("   - 모델 학습 완료.")

    print("5. 제출 파일 생성 시작...")
    make_submission_file(model, test_df, 'data/sample_submission.csv', 'submission.csv')
    print("   - 모든 과정 완료.")

if __name__ == '__main__':
    main()