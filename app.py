import streamlit as st
from openai import OpenAI
import networkx as nx
import matplotlib.pyplot as plt
import io
from PIL import Image
import json

# === 設定 OpenAI API Key ===
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="AI 心智圖產生器", layout="wide")
st.title("🧠 AI 輔助心智圖學習系統")

# === 使用者輸入文章 ===
st.subheader("步驟一：請輸入一段文章，AI 將幫你生成心智圖")
article = st.text_area("貼上文章內容", height=250)

# === 按下按鈕觸發 GPT 呼叫 ===
if st.button("🔍 產生心智圖"):
    if not article:
        st.warning("請先輸入文章內容！")
    else:
        with st.spinner("AI 正在分析文章結構與主題，請稍候..."):
            prompt = f"請將以下文章內容萃取成心智圖的 JSON 結構格式，格式為：{{'主題': ['子主題1', '子主題2', ...]}}\n文章：{article}"
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=1000
                )
                raw_output = response.choices[0].message.content
                st.subheader("AI 回傳的心智圖結構 (JSON)")
                st.code(raw_output, language="json")

                # 嘗試解析 JSON 結構
                try:
                    structure = json.loads(raw_output)

                    # === 畫心智圖 ===
                    G = nx.DiGraph()
                    for main, subs in structure.items():
                        G.add_node(main)
                        for sub in subs:
                            G.add_node(sub)
                            G.add_edge(main, sub)

                    pos = nx.spring_layout(G)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=12, font_weight="bold", edge_color="gray", ax=ax)
                    st.subheader("🧠 自動生成的心智圖")
                    st.pyplot(fig)

                except json.JSONDecodeError:
                    st.error("⚠️ AI 回傳的 JSON 格式有誤，請嘗試重新生成。")

            except Exception as e:
                st.error(f"⚠️ 發生錯誤：{e}")
