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
            
    return transactions, sorted(list(all_menus))

transactions, unique_menus = load_and_preprocess_data(DATA_URL)

# 2. 연관 규칙 분석 (One-hot Encoding & Apriori)
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 최소 지지도(min_support)를 설정하여 빈발 항목집합 도출 (최대 2개 항목 쌍 기준)
frequent_itemsets = apriori(df_encoded, min_support=2/len(transactions), use_colnames=True)

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

# 정렬 기준 선택
sort_by = st.sidebar.radio("정렬 기준", ["향상도 순", "신뢰도 순", "동시 순"])
sort_column_map = {"향상도 순": "향상도", "신뢰도 순": "신뢰도", "동시 순": "동시"}

# 4. 데이터 필터링 및 정렬
filtered_rules = rules_df.copy()

if selected_menu != "전체":
    filtered_rules = filtered_rules[
        (filtered_rules["조건"] == selected_menu) | (filtered_rules["결과"] == selected_menu)
    ]

filtered_rules = filtered_rules.sort_values(by=sort_column_map[sort_by], ascending=False)

# 5. 메인 화면 - 통계 요약 (한 줄 출력)
total_days = len(transactions)
total_menu_types = len(unique_menus)
total_pairs = len(filtered_rules)

st.markdown(f"**급식 일수:** {total_days}일 | **메뉴 종류 수:** {total_menu_types}개 | **함께 나온 적이 있는 메뉴 쌍의 수:** {total_pairs}개")
st.divider()

# 6. 향상도 상위 10개 가로 막대그래프 (Plotly)
st.subheader("📊 향상도 상위 10개 메뉴 쌍")

if not filtered_rules.empty:
    # 향상도 기준 상위 10개 추출
    top10_rules = filtered_rules.sort_values(by="향상도", ascending=False).head(10).copy()
    top10_rules["규칙"] = top10_rules["조건"] + " ➔ " + top10_rules["결과"]
    
    # 가로 막대그래프 생성을 위해 y축 순서 정렬
    top10_rules = top10_rules.sort_values(by="향상도", ascending=True)

    fig = px.bar(
        top10_rules,
        x="향상도",
        y="규칙",
        orientation="h",
        text="향상도",
        title=f"향상도 TOP 10 {'(' + selected_menu + ' 선택됨)' if selected_menu != '전체' else ''}",
        labels={"향상도": "향상도 (Lift)", "규칙": "메뉴 규칙 (조건 ➔ 결과)"}
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("선택한 메뉴에 대한 연관 규칙 데이터가 없습니다.")

# 7. 데이터 표 출력
st.subheader("📋 메뉴 연관 규칙 표")
st.dataframe(filtered_rules, use_container_width=True, hide_index=True)
