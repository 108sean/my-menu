import re
import pandas as pd
import streamlit as st
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 페이지 설정
st.set_page_config(page_title="급식 규칙 찾기", layout="wide")
st.title("🍱 급식 규칙 찾기 (연관 규칙 분석)")

# 1. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/songtan_meals_184.csv"

@st.cache_data
def load_and_preprocess_data(url):
    df = pd.read_csv(url, encoding="utf-8")
    
    transactions = []
    all_menus = set()
    menu_counts = {}  # 개별 메뉴별 총 등장 일수 저장
    
    for row in df["메뉴"].dropna():
        # 괄호 및 괄호 안의 내용 제거 (예: 김치(10.12) -> 김치)
        cleaned_row = re.sub(r"\([^)]*\)", "", str(row))
        # 세로줄(|) 기준 분리 및 공백 제거
        items = [item.strip() for item in cleaned_row.split("|") if item.strip()]
        # 중복 메뉴 제거 후 리스트화
        items = list(set(items))
        if items:
            transactions.append(items)
            all_menus.update(items)
            for item in items:
                menu_counts[item] = menu_counts.get(item, 0) + 1
            
    return transactions, sorted(list(all_menus)), menu_counts

transactions, unique_menus, menu_counts = load_and_preprocess_data(DATA_URL)

# 2. 연관 규칙 분석 (One-hot Encoding & Apriori)
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 최소 지지도(min_support)를 설정하여 빈발 항목집합 도출 (최대 2개 항목 쌍 기준)
frequent_itemsets = apriori(df_encoded, min_support=1/len(transactions), use_colnames=True)

# 연관 규칙 생성
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.0, num_itemsets=len(frequent_itemsets))

# 조건(antecedents)과 결과(consequents)가 각각 1개인 항목 쌍(1:1 규칙)만 추출
rules = rules[(rules['antecedents'].apply(lambda x: len(x) == 1)) & 
              (rules['consequents'].apply(lambda x: len(x) == 1))].copy()

# 데이터 프레임 열 정리 및 명칭 변경
rules["조건"] = rules["antecedents"].apply(lambda x: list(x)[0])
rules["결과"] = rules["consequents"].apply(lambda x: list(x)[0])
rules["동시"] = (rules["support"] * len(transactions)).round().astype(int)
rules["지지도"] = rules["support"].round(4)
rules["신뢰도"] = rules["confidence"].round(4)
rules["향상도"] = rules["lift"].round(4)

# 필요한 열만 선택
rules_df = rules[["조건", "결과", "동시", "지지도", "신뢰도", "향상도"]]

# 3. 사이드바 - 제어 요소
st.sidebar.header("🔍 검색 및 정렬 옵션")

# 메뉴 선택 옵션 (전체 보기 포함)
selected_menu = st.sidebar.selectbox("특정 메뉴 검색", ["전체"] + unique_menus)

# 최소 동시 일수 슬라이더 (1~10일, 기본값 1)
min_cooccurrence = st.sidebar.slider("최소 동시 일수 설정", min_value=1, max_value=10, value=1, step=1)

# 정렬 기준 선택
sort_by = st.sidebar.radio("정렬 기준", ["향상도 순", "신뢰도 순", "동시 순"])
sort_column_map = {"향상도 순": "향상도", "신뢰도 순": "신뢰도", "동시 순": "동시"}

# 4. 데이터 필터링 및 정렬
filtered_rules = rules_df.copy
