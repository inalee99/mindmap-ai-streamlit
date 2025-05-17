import streamlit as st
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import networkx as nx
import matplotlib.pyplot as plt
import json

# === 初始化 KeyBERT 模型（多語言支援，包括中文） ===
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
kw_model = KeyBERT(model)

st.set_page_config(page_title="免費 AI 心智圖產生器", layout="wide")
st.title("🧠 免費 AI 輔助心智圖教學系統")

# === 使用者輸入文章 ===
st.subheader("步驟一：請輸入一段文章，系統將幫你找出主題與子概念")
article = st.text_area("貼上文章內容", height=250)

# === 產生心智圖 ===
if st.button("🔍 萃取主題並產生心智圖"):
    if not article:
        st.warning("請先輸入文章內容！")
    else:
        with st.spinner("AI 正在萃取關鍵主題並建立心智圖..."):
            try:
                # 使用 KeyBERT 萃取主題詞
                keywords = kw_model.extract_keywords(article, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=6)
                main_topic = keywords[0][0] if keywords else "主題"
                sub_topics = [kw[0] for kw in keywords[1:]]

                # 組成結構
                structure = {main_topic: sub_topics}
                st.subheader("系統產出的心智圖資料結構 (JSON)")
                st.code(json.dumps(structure, ensure_ascii=False, indent=2), language="json")

                # 畫心智圖
                G = nx.DiGraph()
                for main, subs in structure.items():
                    G.add_node(main)
                    for sub in subs:
                        G.add_node(sub)
                        G.add_edge(main, sub)

                pos = nx.spring_layout(G)
                fig, ax = plt.subplots(figsize=(10, 6))
                nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightgreen", font_size=12, font_weight="bold", edge_color="gray", ax=ax)
                st.subheader("🧠 自動生成的心智圖")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"⚠️ 發生錯誤：{e}")
