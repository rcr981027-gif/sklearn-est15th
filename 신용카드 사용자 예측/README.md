# 신용카드 사용자 연체 예측 AI 경진대회 베이스라인

이 프로젝트는 Dacon에서 주최한 **신용카드 사용자 연체 예측 AI 경진대회**의 베이스라인 모델을 구현한 Jupyter Notebook입니다.  
RandomForestClassifier와 Stratified K-Fold 검증 방식을 사용하여 기본적인 예측 모델을 구축하고 제출 파일을 생성합니다.

## 📂 파일 구조
```
.
├── credit_card_prediction.ipynb  # 메인 분석 및 모델링 노트
├── output/                       # 생성된 제출 파일이 저장되는 폴더
│   └── baseline_submission.csv
└── ../data/                      # 데이터셋 폴더 (상위 디렉토리의 data 폴더 참조)
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

## 🚀 주요 기능 및 분석 절차

1. **데이터 로드 및 확인**
   - 학습(Train) 및 테스트(Test) 데이터를 불러오고 컬럼 정보를 확인합니다.

2. **데이터 전처리 (Preprocessing)**
   - **결측치 처리**: `occyp_type` (직업 유형)의 결측치를 'None'으로 대체합니다.
   - **파생변수 생성**:
     - `age`: `DAYS_BIRTH`를 이용하여 나이 계산
     - `worked_years`: `DAYS_EMPLOYED`를 이용하여 근무 연수 계산
     - `begin_month`: 신용카드 발급 후 경과 개월 수 절댓값 처리
   - **불필요한 컬럼 삭제**: `index`, `DAYS_BIRTH`, `DAYS_EMPLOYED`, `FLAG_MOBIL` 등
   - **인코딩 (Encoding)**: 범주형 변수(Categorical Features)에 대해 `LabelEncoder` 적용

3. **데이터 탐색 (EDA)**
   - `occyp_type`(직업 유형) 등 주요 변수의 분포를 시각화하여 데이터 특성을 파악합니다.

4. **모델링 (Modeling)**
   - **알고리즘**: `RandomForestClassifier`
   - **검증 방식**: `Stratified K-Fold (n_splits=5)`를 사용하여 데이터 불균형을 고려한 교차 검증 수행
   - **평가 지표**: Log Loss

5. **제출 파일 생성**
   - 학습된 5개의 모델의 예측 확률을 평균(Ensemble)내어 최종 결과를 도출합니다.
   - 결과는 `output/baseline_submission.csv`로 저장됩니다.

## 🛠 사용된 라이브러리
- **Python 3.x**
- **Pandas**: 데이터 처리
- **NumPy**: 수치 연산
- **Matplotlib / Seaborn**: 데이터 시각화
- **Scikit-learn**: 머신러닝 모델링 및 전처리

## 🏃‍♂️ 실행 방법
1. 데이터 파일(`train.csv`, `test.csv`, `sample_submission.csv`)이 `../data/` 경로에 있는지 확인합니다.
2. `credit_card_prediction.ipynb` 파일을 실행하여 순차적으로 셀을 실행합니다.
3. 모든 과정이 완료되면 `output/` 폴더에 생성된 `baseline_submission.csv` 파일을 확인합니다.

---
**Note**: 이 코드는 베이스라인으로, 성능 향상을 위해 Feature Engineering, 하이퍼파라미터 튜닝, 다른 모델(XGBoost, LightGBM 등) 적용을 시도해 볼 수 있습니다.
