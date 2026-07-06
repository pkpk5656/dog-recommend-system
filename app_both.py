import pandas as pd
import numpy as np
import re
import urllib.parse
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="犬種推薦システム比較",
    layout="wide"
)

st.title("🐶 犬種推薦システム比較")
st.write("同じ入力条件で、2つの推薦手法を同時に比較します。")

# ==============================================================================
# 1. データ読み込み
# ==============================================================================

@st.cache_data
def load_data():
    return pd.read_csv("dog_breeds_japanese_3.csv")

df = load_data()

# ==============================================================================
# 2. descriptionを行ごとに分解
# ==============================================================================

def split_description(desc):
    lines = [
        line.strip()
        for line in str(desc).splitlines()
        if line.strip()
    ]

    return {
        "breed_text": lines[0] if len(lines) > 0 else "不明",
        "height_text": lines[1] if len(lines) > 1 else "不明",
        "weight_text": lines[2] if len(lines) > 2 else "不明",
        "life_text": lines[3] if len(lines) > 3 else "不明",
        "trait_text": lines[4] if len(lines) > 4 else "不明",
        "energy_text": lines[5] if len(lines) > 5 else "不明",
        "trainability_text": lines[6] if len(lines) > 6 else "不明",
        "temperament_text": lines[7] if len(lines) > 7 else "不明",
        "grooming_text": lines[8] if len(lines) > 8 else "不明",
        "shedding_text": lines[9] if len(lines) > 9 else "不明"
    }

split_df = df["description"].apply(split_description).apply(pd.Series)
df = pd.concat([df, split_df], axis=1)

# ==============================================================================
# 3. サイズ・運動量の分類
# ==============================================================================

def extract_weight_kg(text):
    nums = re.findall(r'(\d+(?:\.\d+)?)kg', str(text))

    if not nums:
        return None

    nums = [float(n) for n in nums]
    return sum(nums) / len(nums)


def classify_size(weight_kg):
    if weight_kg is None:
        return "不明"

    if weight_kg < 10:
        return "小型犬"
    elif weight_kg < 25:
        return "中型犬"
    else:
        return "大型犬"


def classify_energy(text):
    text = str(text)

    if "少なく" in text or "ゆったり" in text:
        return "少なめ"

    if "非常に活発" in text or "多くの運動" in text:
        return "多め"

    if "適度な運動" in text:
        return "普通"

    return "不明"


def category_score(user_value, dog_value):
    if dog_value == "不明":
        return 0.5

    if user_value == dog_value:
        return 1.0

    if (
        user_value == "小型犬" and dog_value == "大型犬"
    ) or (
        user_value == "大型犬" and dog_value == "小型犬"
    ):
        return 0.0

    return 0.3


df["weight_kg"] = df["weight_text"].apply(extract_weight_kg)
df["size_category"] = df["weight_kg"].apply(classify_size)
df["energy_category"] = df["energy_text"].apply(classify_energy)

# ==============================================================================
# 4. 項目別Sentence-BERT用テキスト
# ==============================================================================

df["beginner_text"] = df["trainability_text"]
df["personality_text"] = df["trait_text"] + " " + df["temperament_text"]
df["barking_text"] = df["trait_text"] + " " + df["temperament_text"]
df["care_text"] = df["grooming_text"] + " " + df["shedding_text"]
df["lifestyle_text"] = df["trait_text"] + " " + df["energy_text"]

feature_columns = [
    "beginner_text",
    "personality_text",
    "barking_text",
    "care_text",
    "lifestyle_text"
]

weights = {
    "size": 1,
    "energy": 1,
    "beginner_text": 1,
    "personality_text": 1,
    "barking_text": 1,
    "care_text": 1,
    "lifestyle_text": 1
}

# ==============================================================================
# 5. モデル読み込み
# ==============================================================================

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

model = load_model()

# ==============================================================================
# 6. ベクトル作成
# ==============================================================================

@st.cache_data
def encode_texts(texts):
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

df["description"] = df["description"].fillna("")

description_embeddings = encode_texts(
    df["description"].astype(str).tolist()
)

item_embeddings = {}

for col in feature_columns:
    item_embeddings[col] = encode_texts(
        df[col].fillna("不明").astype(str).tolist()
    )

# ==============================================================================
# 7. 入力フォーム
# ==============================================================================

st.subheader("希望条件を入力してください")

col_input1, col_input2 = st.columns(2)

with col_input1:
    size_choice = st.selectbox(
        "希望サイズ",
        ["小型犬", "中型犬", "大型犬"]
    )

    energy_choice = st.selectbox(
        "希望する運動量",
        ["少なめ", "普通", "多め"]
    )

    experience = st.text_input(
        "飼育経験",
        "初心者でも飼いやすい"
    )

    personality = st.text_input(
        "理想の性格",
        "人懐っこくて穏やか"
    )

with col_input2:
    barking = st.text_input(
        "吠えやすさの許容範囲",
        "あまり吠えない"
    )

    care = st.text_input(
        "手入れ・抜け毛の希望",
        "手入れが少なく、抜け毛が少ない"
    )

    lifestyle = st.text_input(
        "犬との過ごし方",
        "室内で一緒に過ごし、散歩も楽しみたい"
    )

# ==============================================================================
# 8. 推薦関数：全文Sentence-BERT
# ==============================================================================

def recommend_full_text(
    size_choice,
    energy_choice,
    experience,
    personality,
    barking,
    care,
    lifestyle
):
    user_text = f"""
    希望サイズは{size_choice}です。
    希望する運動量は{energy_choice}です。
    飼育経験は{experience}です。
    理想の性格は{personality}です。
    吠えやすさの許容範囲は{barking}です。
    手入れ・抜け毛の希望は{care}です。
    犬との過ごし方は{lifestyle}です。
    """

    user_vector = model.encode(
        user_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    results = []

    for i, row in df.iterrows():

        text_score = float(np.dot(user_vector, description_embeddings[i]))
        size_score = category_score(size_choice, row["size_category"])
        energy_score = category_score(energy_choice, row["energy_category"])

        final_score = (
            text_score * 0.7
            + size_score * 0.15
            + energy_score * 0.15
        )

        results.append({
            "breed": row["BREED"],
            "score": final_score,
            "text_score": text_score,
            "size_score": size_score,
            "energy_score": energy_score,
            "description": row["description"],
            "size_category": row["size_category"],
            "energy_category": row["energy_category"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ==============================================================================
# 9. 推薦関数：項目別Sentence-BERT
# ==============================================================================

def recommend_item_based(
    size_choice,
    energy_choice,
    experience,
    personality,
    barking,
    care,
    lifestyle
):
    user_inputs = {
        "beginner_text": experience,
        "personality_text": personality,
        "barking_text": barking,
        "care_text": care,
        "lifestyle_text": lifestyle
    }

    user_vectors = {}

    for col in feature_columns:
        user_vectors[col] = model.encode(
            user_inputs[col],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    results = []

    for i, row in df.iterrows():

        total_score = 0
        total_weight = 0
        detail_scores = {}

        size_score = category_score(size_choice, row["size_category"])
        energy_score = category_score(energy_choice, row["energy_category"])

        total_score += size_score * weights["size"]
        total_weight += weights["size"]
        detail_scores["size"] = size_score

        total_score += energy_score * weights["energy"]
        total_weight += weights["energy"]
        detail_scores["energy"] = energy_score

        for col in feature_columns:
            score = float(np.dot(user_vectors[col], item_embeddings[col][i]))

            total_score += score * weights[col]
            total_weight += weights[col]
            detail_scores[col] = score

        final_score = total_score / total_weight

        results.append({
            "breed": row["BREED"],
            "score": final_score,
            "description": row["description"],
            "size_category": row["size_category"],
            "energy_category": row["energy_category"],
            "detail_scores": detail_scores
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ==============================================================================
# 10. 表示関数
# ==============================================================================

def show_result_card(result, method_name):
    st.subheader(result["breed"])
    st.write(f"総合スコア：{result['score']:.4f}")
    st.write(f"分類サイズ：{result['size_category']}")
    st.write(f"運動量分類：{result['energy_category']}")

    if method_name == "full":
        st.write(f"全文類似度：{result['text_score']:.4f}")
        st.write(f"サイズ補正：{result['size_score']:.4f}")
        st.write(f"運動量補正：{result['energy_score']:.4f}")

    if method_name == "item":
        score_df = pd.DataFrame({
            "項目": [
                "サイズ",
                "運動量",
                "飼育経験",
                "性格",
                "吠えやすさ",
                "手入れ・抜け毛",
                "過ごし方"
            ],
            "スコア": [
                result["detail_scores"]["size"],
                result["detail_scores"]["energy"],
                result["detail_scores"]["beginner_text"],
                result["detail_scores"]["personality_text"],
                result["detail_scores"]["barking_text"],
                result["detail_scores"]["care_text"],
                result["detail_scores"]["lifestyle_text"]
            ]
        })

        st.dataframe(score_df, use_container_width=True)

    with st.expander("犬種説明を見る"):
        st.write(result["description"])

    youtube_url = (
        "https://www.youtube.com/results?search_query="
        + urllib.parse.quote_plus(f"{result['breed']} dog")
    )

    st.link_button("YouTubeで見る", youtube_url)


# ==============================================================================
# 11. 実行
# ==============================================================================

if st.button("2つの手法でおすすめ犬種を検索"):

    full_results = recommend_full_text(
        size_choice,
        energy_choice,
        experience,
        personality,
        barking,
        care,
        lifestyle
    )

    item_results = recommend_item_based(
        size_choice,
        energy_choice,
        experience,
        personality,
        barking,
        care,
        lifestyle
    )

    st.header("推薦結果の比較")

    left, right = st.columns(2)

    with left:
        st.subheader("全文Sentence-BERT")
        for rank, result in enumerate(full_results[:3], start=1):
            st.markdown(f"### {rank}位")
            show_result_card(result, "full")
            st.divider()

    with right:
        st.subheader("項目別Sentence-BERT")
        for rank, result in enumerate(item_results[:3], start=1):
            st.markdown(f"### {rank}位")
            show_result_card(result, "item")
            st.divider()