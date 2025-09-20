# pages/logs.py
# Logs page

import streamlit as st

def page_logs():
    st.header("🗒️ System Logs")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh"):
            try:
                st.rerun()
            except Exception:
                pass
            
    with col2:
        clear_logs = st.button("🗑️ Clear Logs")
    
    # Handle clear logs action outside columns for full-screen message
    if clear_logs:
        st.session_state["logs"] = []
        st.success("Logs cleared!")
    
    if not st.session_state["logs"]:
        st.info("No logs available yet. Have some chat!")
        return

    for i, log in enumerate(reversed(st.session_state["logs"])):
        with st.expander(f"Query {len(st.session_state['logs']) - i}: {log['q'][:60]}..."):
            st.markdown(f"**⏰ Time:** {log['time']}")
            st.markdown(f"**❓ Question:** {log['q']}")
            st.markdown(f"**💡 Answer:**\n\n{log['a']}")  # Simple text, no styling
            if log['sources']:
                st.markdown(f"**📂 Sources:** {', '.join(log['sources'])}")
            st.markdown(f"**🧠 Model:** {log['model']} &nbsp;&nbsp; **⚙️ Mode:** {log['mode']}")
            st.markdown(f"**📄 Documents Retrieved:** {log.get('docs_retrieved', 'N/A')}")
            st.markdown(f"**⏱️ Response Time:** {log.get('response_time', 'N/A')} sec")
