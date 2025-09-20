# page_modules/analytics.py
# The enhanced analytics dashboard for the application

import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

def page_analytics():
    """ Renders the enhanced analytics dashboard. """
    st.header('📊 Analytics Dashboard')
    st.markdown("---")

    # Ensure logs exist and are not empty
    if "logs" not in st.session_state or not st.session_state["logs"]:
        st.warning("No logs recorded yet. Ask some questions in the Chat page to see analytics.")
        return

    # --- 1. Data Preparation ---
    # Convert the list of log dictionaries to a Pandas DataFrame for easy analysis
    try:
        df = pd.DataFrame(st.session_state["logs"])
        df['time'] = pd.to_datetime(df['time'])
    except Exception as e:
        st.error(f"An error occurred while preparing the data: {e}")
        return

    # --- 2. Key Metrics Dashboard ---
    st.markdown("### 📈 Key Metrics")
    
    total_queries = len(df)
    avg_response_time = df['response_time'].mean()
    total_docs_retrieved = df['docs_retrieved'].sum()

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(
        label="Total Queries",
        value=total_queries,
    )
    kpi2.metric(
        label="Avg. Response Time (s)",
        value=f"{avg_response_time:.2f}",
    )
    kpi3.metric(
        label="Total Docs Retrieved",
        value=total_docs_retrieved,
    )

    st.markdown("---")

    # --- 3. Usage Breakdown ---
    st.markdown("### 🛠️ Usage Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model Usage Pie Chart
        model_counts = df['model'].value_counts()
        fig_model = px.pie(
            model_counts, 
            values=model_counts.values, 
            names=model_counts.index, 
            title='Model Usage Distribution'
        )
        st.plotly_chart(fig_model, use_container_width=True)

    with col2:
        # Answer Mode Usage Pie Chart
        mode_counts = df['mode'].value_counts()
        fig_mode = px.pie(
            mode_counts, 
            values=mode_counts.values, 
            names=mode_counts.index, 
            title='Answer Mode Usage Distribution'
        )
        st.plotly_chart(fig_mode, use_container_width=True)

    st.markdown("---")

    # --- 4. Performance & Source Analysis ---
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Query Volume Over Time
        st.markdown("### 🗓️ Query Volume Over Time")
        queries_per_day = df.set_index('time').resample('D').size().reset_index(name='count')
        fig_timeline = px.line(
            queries_per_day, 
            x='time', 
            y='count', 
            title='Daily Query Volume',
            labels={'time': 'Date', 'count': 'Number of Queries'}
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    with col4:
        # Top 10 Most Cited Sources
        st.markdown("### 📚 Top Document Sources")
        # Flatten the list of sources, handling cases where sources might be empty
        all_sources = [source for sublist in df['sources'] for source in sublist if sublist]
        if all_sources:
            source_counts = Counter(all_sources)
            top_sources = pd.DataFrame(source_counts.most_common(10), columns=['Source', 'Count'])
            fig_sources = px.bar(
                top_sources, 
                x='Count', 
                y='Source', 
                orientation='h', 
                title='Top 10 Most Cited Sources'
            )
            fig_sources.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_sources, use_container_width=True)
        else:
            st.info("No sources have been cited yet in RAG mode.")

    st.markdown("---")

    # --- 5. Raw Data Logs ---
    with st.expander("📂 View Raw Logs"):
        st.dataframe(df, use_container_width=True)