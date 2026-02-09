# Pandas 학습 자료

Pandas 라이브러리의 기초부터 데이터 전처리, 시각화까지 학습하는 튜토리얼 모음입니다.

## 파일 목록

| 파일명 | 주제 | 학습 내용 |
|--------|------|----------|
| `1_pandas.ipynb` | Series와 DataFrame 생성 | pandas import, Series/DataFrame 생성, index/columns 설정 |
| `2_pandas.ipynb` | 파일 읽기와 기본 정보 | read_csv, head/tail, info, describe, shape, 정렬 |
| `3_pandas.ipynb` | 데이터 선택과 필터링 | Column 선택, loc/iloc, Boolean 인덱싱, isin, 결측값 확인 |
| `4_pandas.ipynb` | 데이터 조작과 통계 | copy, row/column 추가삭제, 통계함수, 피벗테이블, GroupBy |
| `5_pandas.ipynb` | 결측값과 중복값 처리 | fillna, dropna, drop_duplicates, drop |
| `6_pandas.ipynb` | 데이터 합치기 | concat, merge (left/right/inner/outer) |
| `7_pandas.ipynb` | 데이터 타입 변환 | astype, to_datetime, dt 속성 활용 |
| `8_pandas.ipynb` | 함수 적용 | apply, lambda, map |
| `9_pandas.ipynb` | DataFrame 연산 | Column간 연산, 숫자 연산, NaN 연산, DataFrame간 연산 |
| `10_pandas.ipynb` | 데이터 타입 선택과 인코딩 | select_dtypes, 원핫인코딩 (get_dummies) |
| `11_pandas.ipynb` | 데이터 전처리 실전 | rename, strip, replace, 실전 데이터 정제 |
| `12_pandas.ipynb` | 데이터 시각화 | plot (line, bar, hist, kde, box, pie, scatter 등) |

## 상세 설명

### 1_pandas.ipynb - Series와 DataFrame 생성
- pandas 패키지 import (`import pandas as pd`)
- Series 생성: 1차원 데이터 구조
- DataFrame 생성: 리스트 또는 딕셔너리로 생성
- columns, index 설정 방법

### 2_pandas.ipynb - 파일 읽기와 기본 정보
- `pd.read_csv()`: CSV 파일 읽기
- `head()`, `tail()`: 상위/하위 데이터 확인
- `info()`: 데이터 타입, 결측값 확인
- `describe()`: 통계 정보 확인
- `shape`: 행/열 개수 확인
- `sort_index()`, `sort_values()`: 정렬

### 3_pandas.ipynb - 데이터 선택과 필터링
- Column 선택: `df['column']`, `df.column`
- `loc`: 라벨 기반 인덱싱
- `iloc`: 위치 기반 인덱싱
- Boolean 인덱싱: 조건을 활용한 필터링
- `isin()`: 특정 값 포함 여부 확인
- `isna()`, `isnull()`, `notnull()`: 결측값 확인

### 4_pandas.ipynb - 데이터 조작과 통계
- `copy()`: DataFrame 복사
- row/column 추가 및 삭제
- 통계 함수: `min()`, `max()`, `sum()`, `mean()`, `var()`, `std()`, `count()`, `median()`, `mode()`
- `pivot_table()`: 피벗 테이블 생성
- `groupby()`: 그룹별 집계
- Multi-Index 활용

### 5_pandas.ipynb - 결측값과 중복값 처리
- `fillna()`: 결측값 채우기 (특정값, 평균, 최빈값 등)
- `dropna()`: 결측값이 있는 행/열 제거 (axis, how 옵션)
- `drop_duplicates()`: 중복값 제거
- `drop()`: 특정 행/열 제거
- `reset_index()`: 인덱스 초기화

### 6_pandas.ipynb - 데이터 합치기
- `pd.concat()`: DataFrame 단순 연결 (axis=0: 행방향, axis=1: 열방향)
- `pd.merge()`: 특정 키 기준 병합
  - `how='left'`: 왼쪽 기준 병합
  - `how='right'`: 오른쪽 기준 병합
  - `how='inner'`: 교집합 병합
  - `how='outer'`: 합집합 병합
- `left_on`, `right_on`: 다른 컬럼명으로 병합

### 7_pandas.ipynb - 데이터 타입 변환
- `astype()`: 데이터 타입 변환 (int, float, object 등)
- `pd.to_datetime()`: datetime 타입 변환
- `dt` 속성: year, month, day, hour, minute, second, dayofweek 추출

### 8_pandas.ipynb - 함수 적용
- `apply()`: 사용자 정의 함수 적용
- `lambda`: 간단한 익명 함수 적용
- `map()`: 딕셔너리를 활용한 값 매핑

### 9_pandas.ipynb - DataFrame 연산
- Column간 연산: `+`, `-`, `*`, `/`, `%`
- Column과 숫자 연산
- NaN이 포함된 경우의 연산
- DataFrame간 연산
- `mean()`, `sum()` with axis 옵션

### 10_pandas.ipynb - 데이터 타입 선택과 인코딩
- `select_dtypes()`: 특정 데이터 타입의 컬럼만 선택
  - `include='object'`: 문자열 컬럼
  - `exclude='object'`: 숫자형 컬럼
- `pd.get_dummies()`: 원핫인코딩

### 11_pandas.ipynb - 데이터 전처리 실전
- `rename()`: 컬럼명 변경
- `str.strip()`: 공백 제거
- `str.replace()`: 문자열 치환
- 실전 데이터 정제 과정 (공백, 콤마, 하이픈 처리)
- `groupby()` 활용한 집계 분석

### 12_pandas.ipynb - 데이터 시각화
- `plot()` 함수와 kind 옵션:
  - `line`: 선 그래프 (연속 데이터)
  - `bar`, `barh`: 막대 그래프 (그룹 비교)
  - `hist`: 히스토그램 (분포)
  - `kde`: 커널 밀도 그래프
  - `box`: 박스 플롯 (이상치 탐지)
  - `area`: 면적 그래프
  - `pie`: 파이 그래프 (점유율)
  - `scatter`: 산점도 (상관관계)
  - `hexbin`: 고밀도 산점도
- 한글 폰트 설정 (Colab 환경)

## 학습 순서

1번부터 12번까지 순서대로 학습하는 것을 권장합니다.

## 사용 데이터셋

- [국내 아이돌 평판지수](http://bit.ly/ds-korean-idol)
- [국내 아이돌 연봉, 가족수](http://bit.ly/ds-korean-idol-2)
- [민간 아파트 가격동향](https://bit.ly/ds-house-price)
