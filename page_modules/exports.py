# pages/exports.py
# Exports page

import streamlit as st
from datetime import datetime

def page_exports():
    st.header("📦 Exports & Reports")
    st.markdown("---")
    if st.session_state["logs"]:
        if st.button("📄 Generate Chat History Report (Markdown)"):
            report = "# Plant AI Assistant – Chat History Report\n\n"
            report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report += f"Total Queries: {len(st.session_state['logs'])}\n\n"
            for i, log in enumerate(st.session_state["logs"], 1):
                report += f"## Query {i}\n"
                report += f"**Time:** {log['time']}\n\n"
                report += f"**Question:** {log['q']}\n\n"
                report += f"**Answer:**\n{log['a']}\n\n"
                if log['sources']:
                    report += f"**Sources:** {', '.join(log['sources'])}\n\n"
                report += f"**Model:** {log['model']}\n\n"
                report += f"**Response Time:** {log.get('response_time','N/A')}s\n\n"
            st.download_button(
                "⬇️ Download Report",
                data=report,
                file_name="plant_ai_report.md",
                mime="text/markdown",
            )
    else:
        st.info("No chat history to export yet.")

    analytics = st.session_state["analytics"]
    if analytics["total_queries"] > 0:
        summary = f"""**Session Analytics**
- Total Queries: {analytics['total_queries']}
- Average Response Time: {analytics['avg_response_time']:.2f}s
- Documents in Index: {len(st.session_state.get('all_chunks', []))}
"""
        st.markdown(summary)
    else:
        st.info("No analytics data available yet.")