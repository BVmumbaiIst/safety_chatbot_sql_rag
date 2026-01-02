import streamlit as st

st.set_page_config(
    page_title="Safety Optimise Dashboard",
    layout="wide"
)

pg = st.navigation([
    st.Page("pages/overview.py", title="Overview", icon="🏠", default=True),
    st.Page("pages/safety_chatbot_sql_rag_app.py", title="Raw_overview", icon="🏠", default=True),
    st.Page("pages/template_2mfms.py", title="2MFMS", icon="📊"),
    st.Page("pages/template_toolbox.py", title="Toolbox", icon="🧰"),
])

pg.run()
