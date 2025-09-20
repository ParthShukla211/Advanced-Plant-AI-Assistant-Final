# page_modules/models.py
# Advanced, interactive model management page

import os
import streamlit as st
import pandas as pd
from models.adapters import CTRANS_CHOICES, HF_T2T_CHOICES

# --- Augmented Model Data ---
# This dictionary now contains all the rich metadata for the page.
ALL_MODELS = {
    "🟢 Local Mistral (GGUF)": {
        "emoji": "💨",
        "family": "Mistral (by Mistral AI, distilled by DeepSeek)",
        "size": "8B",
        "quantization": "Q4_0",
        "type": "Local File",
        "speed": 9,
        "quality": 8.5,
        "use_case": "High-quality chat and complex instruction following.",
        "specialization": "A powerful, general-purpose model that excels at reasoning and following nuanced instructions. It's a top-tier choice for a local model when quality is a priority.",
        "description": "This model is a quantized version of a fine-tuned Mistral model, optimized for performance on local machines. It provides a great balance of speed and high-quality responses, making it suitable for a wide variety of tasks from simple Q&A to more complex, multi-turn conversations. Its instruction-following capabilities are particularly strong."
    },
    "🟡 Mistral-7B-Instruct (Q4_K_M, auto)": {
        "emoji": "🌬️",
        "family": "Mistral",
        "size": "7B",
        "quantization": "Q4_K_M",
        "type": "Auto-Download",
        "speed": 8,
        "quality": 8,
        "use_case": "A great balance of high performance and reasonable resource usage.",
        "specialization": "Instruction-following and chat. It's one of the most popular open-source models for a reason.",
        "description": "The original instruction-tuned model from Mistral AI. It set a new standard for open-source models in its size class and remains a very strong performer. The Q4_K_M quantization offers a good compromise between model size and performance."
    },
    "🟠 Local Phi-2 (Q5_0)": {
        "emoji": "💡",
        "family": "Phi (by Microsoft)",
        "size": "2.7B",
        "quantization": "Q5_0",
        "type": "Local File",
        "speed": 10,
        "quality": 7,
        "use_case": "Excellent for logical reasoning and code-related tasks in a compact size.",
        "specialization": "Common-sense reasoning, logical puzzles, and code generation.",
        "description": "Phi-2 is a Small Language Model (SLM) from Microsoft that punches well above its weight class. It was trained on 'textbook-quality' data, which gives it surprisingly strong reasoning abilities. It's an excellent choice when you need a fast, lightweight model for tasks that require logic."
    },
    "🟣 Llama-2-7B-Chat (Q3_K_S, auto)": {
        "emoji": "🦙",
        "family": "Llama (by Meta)",
        "size": "7B",
        "quantization": "Q3_K_S",
        "type": "Auto-Download",
        "speed": 7,
        "quality": 7,
        "use_case": "A reliable and safe choice for general-purpose conversational AI.",
        "specialization": "Safe and helpful conversations. It has been fine-tuned to avoid generating harmful content.",
        "description": "Meta's Llama 2 is one of the most well-known open-source models. The chat-tuned version is designed to be a helpful assistant. While it may be slightly less performant than newer models like Mistral, it is a very solid and reliable choice."
    },
    "🟤 Qwen2-7B-Instruct (Q4_K_M, auto)": {
        "emoji": "🐉",
        "family": "Qwen (by Alibaba Cloud)",
        "size": "7B",
        "quantization": "Q4_K_M",
        "type": "Auto-Download",
        "speed": 8,
        "quality": 8.2,
        "use_case": "Excels in multilingual tasks and following complex instructions.",
        "specialization": "Multilingual capabilities and strong instruction-following.",
        "description": "Qwen2 is a powerful series of models from Alibaba Cloud. The 7B instruction-tuned version is highly capable and competitive with other top models in its class. It is particularly noted for its strong performance in languages other than English."
    },
    "🔵 Flan-T5 Small (HF)": {
        "emoji": "🍮",
        "family": "Flan-T5 (by Google)",
        "size": "80M",
        "quantization": "N/A",
        "type": "Hugging Face (T2T)",
        "speed": 10,
        "quality": 5,
        "use_case": "Extremely fast for simple tasks like classification and summarization.",
        "specialization": "Text-to-text tasks, such as translation, summarization, and question answering.",
        "description": "Flan-T5 is a powerful text-to-text model from Google. The 'small' version is incredibly fast and lightweight, making it perfect for simple, structured NLP tasks. It is not designed for conversational chat."
    },
    "🔵 Flan-T5 Base (HF)": {
        "emoji": "🍮",
        "family": "Flan-T5 (by Google)",
        "size": "250M",
        "quantization": "N/A",
        "type": "Hugging Face (T2T)",
        "speed": 8,
        "quality": 6,
        "use_case": "A solid baseline for a variety of standard natural language processing tasks.",
        "specialization": "A good middle-ground for text-to-text tasks.",
        "description": "The 'base' version of Flan-T5 offers a significant step up in quality from the small version, while remaining relatively fast. It's a great all-around choice for many NLP benchmarks."
    },
    "🔵 Flan-T5 Large (HF)": {
        "emoji": "🍮",
        "family": "Flan-T5 (by Google)",
        "size": "780M",
        "quantization": "N/A",
        "type": "Hugging Face (T2T)",
        "speed": 6,
        "quality": 7,
        "use_case": "Higher quality responses for more nuanced text-to-text tasks.",
        "specialization": "High-quality text generation for summarization, translation, and other text-to-text tasks.",
        "description": "The 'large' version of Flan-T5 provides the highest quality responses in the family, at the cost of speed and resource usage. It is an excellent choice for offline processing tasks where quality is the top priority."
    }
}

def page_models():
    st.header("🧠 Model Dashboard")
    st.markdown("Explore, compare, and select the AI model that best fits your needs for speed, quality, and task specialization.")
    st.markdown("---")

    # --- 1. Currently Selected Model ---
    selected_model_name = st.session_state.get("current_model_kind")
    selected_model_info = ALL_MODELS.get(selected_model_name, {})
    
    st.markdown("### ⭐ Currently Selected Model")
    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"<p style='font-size: 52px; text-align: center;'>{selected_model_info.get('emoji', '🧠')}</p>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"#### {selected_model_name}")
            st.markdown(f"**Family:** `{selected_model_info.get('family', 'N/A')}` | **Size:** `{selected_model_info.get('size', 'N/A')}` | **Type:** `{selected_model_info.get('type', 'N/A')}`")
            st.markdown(selected_model_info.get('desc', 'No description available.'))

    st.markdown("---")

    # --- 2. Model Comparison Tool ---
    st.markdown("### ⚖️ Model Comparison Tool")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        model_names = list(ALL_MODELS.keys())
        
        with c1:
            st.markdown("##### Model A")
            model_a_name = st.selectbox("Select Model A", model_names, index=0, key="compare_a")
            model_a_info = ALL_MODELS[model_a_name]
            st.markdown(f"**Speed:** {'⚡' * model_a_info['speed']}")
            st.markdown(f"**Quality:** {'🎯' * int(model_a_info['quality'])}")
            st.markdown(f"**Best for:** *{model_a_info['use_case']}*")

        with c2:
            st.markdown("##### Model B")
            model_b_name = st.selectbox("Select Model B", model_names, index=1, key="compare_b")
            model_b_info = ALL_MODELS[model_b_name]
            st.markdown(f"**Speed:** {'⚡' * model_b_info['speed']}")
            st.markdown(f"**Quality:** {'🎯' * int(model_b_info['quality'])}")
            st.markdown(f"**Best for:** *{model_b_info['use_case']}*")

    st.markdown("---")

    # --- 3. Performance Overview ---
    st.markdown("### 📊 Performance At-a-Glance")
    with st.container(border=True):
        df_data = []
        for name, info in ALL_MODELS.items():
            df_data.append({
                "Model": name, "Speed": info["speed"], "Quality": info["quality"], "Size (Params)": info["size"]
            })
        df = pd.DataFrame(df_data)
        st.dataframe(df.set_index("Model"), use_container_width=True)

    st.markdown("---")

    # --- 4. Detailed Model Library ---
    st.markdown("### 📇 Detailed Model Library")
    for name, info in ALL_MODELS.items():
        with st.container(border=True):
            st.markdown(f"### {info['emoji']} {name}")
            
            # Key Info Badges
            param_html = f"<span class='badge'>{info['type']}</span> <span class='badge'>{info['family']}</span> <span class='badge'>{info['size']} Params</span> <span class='badge'>{info['quantization']}</span>"
            st.markdown(param_html, unsafe_allow_html=True)
            
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Specialization:** {info['specialization']}")
            
            # Action Button
            if name == selected_model_name:
                st.button("Currently Selected ⭐", key=f"select_{name}", use_container_width=True, disabled=True)
            else:
                if st.button("Select this Model", key=f"select_{name}", use_container_width=True):
                    st.session_state["current_model_kind"] = name
                    st.rerun()