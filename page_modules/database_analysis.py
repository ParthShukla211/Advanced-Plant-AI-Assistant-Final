# pages/database_analysis.py
# Database analysis page

import streamlit as st
from rag.document_processing import pdf_meta_scan

def page_database_analysis():
    st.header("🗃️ Database Analysis")
    st.markdown("---")
    pdf_folder = "Database"
    df = pdf_meta_scan(pdf_folder)
    if df.empty:
        st.info("No files found in the `Database` folder. Please add PDFs and refresh.")
        return

    total_files = len(df)
    pdf_files = df[df["ext"] == "PDF"]
    total_pdfs = len(pdf_files)
    total_pages = int(pdf_files["pages"].fillna(0).sum())
    total_size_mb = df["size_kb"].sum() / 1024.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Files", total_files)
    c2.metric("PDF Files", total_pdfs)
    c3.metric("Total Pages (PDF)", total_pages)
    c4.metric("Total Size (MB)", f"{total_size_mb:.2f}")

    st.markdown("#### File Type Distribution")
    type_counts = df["ext"].value_counts().reset_index()
    type_counts.columns = ["Type", "Count"]
    st.bar_chart(type_counts.set_index("Type"))

    st.markdown("#### Top 15 Largest Files (MB)")
    df_largest = df.sort_values(by="size_kb", ascending=False).head(15).copy()
    df_largest["size_mb"] = df_largest["size_kb"] / 1024.0
    st.dataframe(
        df_largest[["file", "ext", "size_mb", "pages", "modified"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### All Files")
    st.dataframe(
        df[["file", "ext", "size_kb", "pages", "modified", "path"]],
        use_container_width=True,
        hide_index=True,
    )

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download File Inventory (CSV)",
        data=csv_data,
        file_name="database_inventory.csv",
        mime="text/csv",
    )