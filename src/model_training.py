import pandas as pd

try:
    # 샘플 제출 파일 불러오기 (경로 수정)
    submission_df = pd.read_csv('../data/sample_submission.csv')
    
    # 예측 결과를 'stress_score' 컬럼에 할당
    submission_df['stress_score'] = final_predictions
    
    # 제출 파일 저장
    submission_df.to_csv('submission_baseline.csv', index=False)
    
    print("제출 파일 'submission_baseline.csv'가 성공적으로 생성되었습니다.")

except FileNotFoundError:
    print("파일 경로를 찾을 수 없습니다. 'data' 폴더의 위치와 'sample_submission.csv' 파일 이름을 확인해주세요.")