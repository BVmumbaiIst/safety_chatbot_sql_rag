"""
safety_chatbot_sql_rag_app.py
Corrected, memory-optimized Streamlit app for Safety Culture Chatbot (SQL + optional RAG)

Features:
- Lazy DB metadata load (no full table in memory at startup)
- On-demand SQL queries when user clicks "Run Query"
- Optional RAG (FAISS) built only when requested and limited in size
- Cached LLM initialization (ChatOpenAI) with fallback if no API key
- Safe SQL IN formatting and robust date handling
- Temp DB cleanup and resource-friendly behavior for Streamlit Cloud
"""

import os
import sqlite3
import tempfile
import atexit
import time
import traceback
import uuid
import gc

import pandas as pd
import numpy as np
import streamlit as st
import json
from datetime import datetime
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import boto3

# Optional imports (guarded)
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import create_sql_agent
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain.chains.retrieval import RetrievalQA
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# visualization libs
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards

# ============================================================
# Basic page config - must be set before ANY st.* call
# ============================================================
st.set_page_config(page_title="💬 Safety Chatbot (SQL + Optional RAG)", layout="wide")
st.title("💬 Safety Chatbot — SQL + Optional RAG (Memory-Optimized)")

# ============================================================
# Load environment and keys
# ============================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# AWS S3 config (if you store DBs in S3)
BUCKET_NAME = "iauditorsafetydata"
S3_KEYS = {
    "items": "BVPI_Safety_Optimise/safety_Chat_bot_db/inspection_employee_schedule_items.db",
    "users": "BVPI_Safety_Optimise/safety_Chat_bot_db/inspection_employee_schedule_users.db"
}

# Create S3 client once (will use env credentials if set)
s3 = boto3.client("s3")

# ============================================================
# Helper: cleanup old temp DBs (>24 hours) and register atexit
# ============================================================
def cleanup_old_dbs(tmp_dir=tempfile.gettempdir(), hours=24):
    cutoff = time.time() - hours * 3600
    removed = 0
    for file in os.listdir(tmp_dir):
        if file.endswith(".db"):
            path = os.path.join(tmp_dir, file)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except Exception:
                pass
    if removed:
        st.info(f"Cleaned up {removed} old temp .db files from {tmp_dir}")

# run immediately (Streamlit reruns will call this often but it's cheap)
cleanup_old_dbs()
atexit.register(lambda: cleanup_old_dbs())

# ============================================================
# S3 helper: download DB if not present locally & validate
# (if you keep DBs in S3). If you store DBs in repo, adapt accordingly.
# ============================================================
def load_sqlite_from_s3_cached(s3_key: str):
    """
    Download an S3 object to /tmp if not present already.
    Returns local path to .db
    """
    base_name = os.path.basename(s3_key).replace("/", "_")
    local_path = os.path.join(tempfile.gettempdir(), base_name)

    # if exists and appears valid, reuse
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
        return local_path

    # Attempt download
    try:
        s3.download_file(BUCKET_NAME, s3_key, local_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download {s3_key} from S3: {e}")

    # Validate quickly
    try:
        conn = sqlite3.connect(local_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        conn.close()
        if tables.empty:
            raise RuntimeError(f"No tables found in downloaded DB: {local_path}")
    except Exception as e:
        # remove broken file
        try:
            os.remove(local_path)
        except Exception:
            pass
        raise RuntimeError(f"Downloaded DB validation failed: {e}")

    # ensure cleanup on exit
    atexit.register(lambda: os.path.exists(local_path) and os.remove(local_path))
    return local_path

# ============================================================
# Cached: Get DB local paths (from S3)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_db_paths():
    """
    Returns tuple (items_db_path, users_db_path)
    If S3 is unavailable this raises.
    """
    items_path = load_sqlite_from_s3_cached(S3_KEYS["items"])
    users_path = load_sqlite_from_s3_cached(S3_KEYS["users"])
    return items_path, users_path

# Try to load paths now; if fails, stop the app gracefully
try:
    DB_PATH_ITEMS, DB_PATH_USERS = get_db_paths()
except Exception as e:
    st.error("❌ Could not load database files from S3 or local path.")
    st.exception(e)
    st.stop()

# ============================================================
# Utility functions: DB connections & quick SQL run
# ============================================================
def get_connection_items():
    return sqlite3.connect(DB_PATH_ITEMS)

def get_connection_users():
    return sqlite3.connect(DB_PATH_USERS)

def run_sql_query(db_path, sql, params=None, limit_rows=None):
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        if limit_rows:
            # naive wrapper: add LIMIT if not present (safe only for our controlled queries)
            sql_l = sql.rstrip().rstrip(";")
            sql_l = f"{sql_l} LIMIT {limit_rows};"
            df = pd.read_sql(sql_l, conn, params=params)
        else:
            df = pd.read_sql(sql, conn, params=params)
        return df
    finally:
        conn.close()

# ============================================================
# Metadata loader (small queries only) - cached
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_db_metadata(db_path, table_hint=None):
    """
    Loads small metadata (min/max dates and distinct values for filters)
    Returns dict with keys: table, date_min, date_max, distincts
    """
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        # detect first table
        tbl_df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;", conn)
        table_name = tbl_df.iloc[0, 0] if not tbl_df.empty else (table_hint or "")
        meta = {}
        try:
            meta_df = pd.read_sql(f'SELECT MIN("date completed") as date_min, MAX("date completed") as date_max FROM "{table_name}";', conn)
            meta["date_min"] = meta_df["date_min"].iloc[0] if not meta_df.empty else None
            meta["date_max"] = meta_df["date_max"].iloc[0] if not meta_df.empty else None
        except Exception:
            meta["date_min"], meta["date_max"] = None, None

        distincts = {}
        # columns we want to use as filters (tunable)
        cols = ["region", "TemplateNames", "owner name", "assignee status", "employee status", "email"]
        for c in cols:
            try:
                q = f'SELECT DISTINCT "{c}" as val FROM "{table_name}" WHERE "{c}" IS NOT NULL LIMIT 2000;'
                vals = pd.read_sql(q, conn)["val"].dropna().tolist()
                distincts[c] = sorted(vals)
            except Exception:
                distincts[c] = []
        return {"table": table_name, "meta": meta, "distincts": distincts}
    finally:
        conn.close()

# load metadata for items and users
items_meta = load_db_metadata(DB_PATH_ITEMS)
users_meta = load_db_metadata(DB_PATH_USERS)

# ============================================================
# LLM setup (cached). If OPENAI_API_KEY missing, llm will be None
# ============================================================
@st.cache_resource(show_spinner=False)
def setup_llm():
    if not OPENAI_API_KEY:
        return None
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
        return llm
    except Exception:
        return None

llm = setup_llm()
if llm is None:
    st.info("⚠️ No OpenAI API key found or LLM init failed. LLM features will be disabled.")

# ============================================================
# Safe helpers for building SQL strings
# ============================================================
def sql_list(values):
    """Escape single quotes and return SQL IN formatted string"""
    safe = [str(v).replace("'", "''") for v in values]
    return ",".join([f"'{s}'" for s in safe])

def build_where_clause(selected_filters, date_range_tuple=None):
    """
    selected_filters: dict with keys mapping to SQL column names and list of values
    date_range_tuple: (start_date, end_date) or None
    """
    filters = []
    for col, vals in selected_filters.items():
        if vals:
            filters.append(f'"{col}" IN ({sql_list(vals)})')
    if date_range_tuple:
        start = pd.to_datetime(date_range_tuple[0]).strftime("%Y-%m-%d")
        end = pd.to_datetime(date_range_tuple[1]).strftime("%Y-%m-%d")
        filters.append(f'"date completed" BETWEEN "{start}" AND "{end}"')
    return " AND ".join(filters) if filters else "1=1"

# ============================================================
# Sidebar: login and filters (use metadata to populate options)
# ============================================================
with st.sidebar:
    st.header("🔑 Login")
    entered_email = st.text_input("Enter your Email", key="login_email")
    if st.button("Login"):
        emails_list = users_meta["distincts"].get("email", [])
        if entered_email:
            if entered_email in emails_list:
                st.session_state["logged_in"] = True
                st.session_state["email"] = entered_email
                st.success(f"✅ Logged in as: {entered_email}")
            else:
                st.session_state["logged_in"] = False
                st.error("❌ Access denied. Email not found.")
        else:
            st.warning("Please enter an email.")

# require login to proceed
if not st.session_state.get("logged_in", False):
    st.warning("🔒 Please log in to access filters and data.")
    st.stop()

st.sidebar.header("🔎 Filters (use Run Query to apply)")
# date inputs robust handling
date_min = items_meta["meta"].get("date_min")
date_max = items_meta["meta"].get("date_max")

if date_min is not None and date_max is not None:
    try:
        default_dates = [pd.to_datetime(date_min).date(), pd.to_datetime(date_max).date()]
    except Exception:
        default_dates = None
else:
    default_dates = None

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=default_dates,
    min_value=pd.to_datetime(date_min).date() if date_min is not None else None,
    max_value=pd.to_datetime(date_max).date() if date_max is not None else None
)

# normalize date_range
if isinstance(date_range, (list, tuple)):
    if len(date_range) == 0:
        date_range = None
    elif len(date_range) == 1:
        date_range = (date_range[0], date_range[0])
else:
    # single date returned sometimes - convert to tuple
    try:
        date_range = (date_range, date_range)
    except Exception:
        date_range = None

# filter pickers use limited distinct lists already loaded
regions = st.sidebar.multiselect("Select Regions", items_meta["distincts"].get("region", []))
templates = st.sidebar.multiselect("Select Templates", items_meta["distincts"].get("TemplateNames", []))
employees = st.sidebar.multiselect("Select Employees (owner name)", items_meta["distincts"].get("owner name", []))
statuses = st.sidebar.multiselect("Select Assignee Status", items_meta["distincts"].get("assignee status", []))
employee_status = st.sidebar.multiselect("Select Employee Status", items_meta["distincts"].get("employee status", []))
row_limit = st.sidebar.slider("Limit rows to preview (saves memory)", min_value=10, max_value=5000, value=500, step=10)

# ============================================================
# Build SQL query (but do NOT run it automatically)
# ============================================================
selected_filters = {
    "region": regions,
    "TemplateNames": templates,
    "owner name": employees,
    "assignee status": statuses,
    "employee status": employee_status
}
where_clause = build_where_clause(selected_filters, date_range_tuple=date_range)
items_table_name = items_meta["table"]
default_query = f'SELECT * FROM "{items_table_name}" WHERE {where_clause} ;'

st.sidebar.markdown("### SQL preview (edit if needed)")
sql_query_text = st.sidebar.text_area("Edit SQL", value=default_query, height=160)
st.sidebar.code(sql_query_text, language="sql")

# ============================================================
# Run Query button - query runs on demand and result is limited
# We store the SQL in session_state (not the full df). We store
# only a small preview (N rows) for display and visual generation,
# to avoid keeping huge DataFrames in memory.
# ============================================================
def store_filtered_preview(sql_text, preview_limit=500):
    try:
        df_preview = run_sql_query(DB_PATH_ITEMS, sql_text, limit_rows=preview_limit)
        # store small preview and count (metadata)
        st.session_state["filtered_preview"] = df_preview
        # store the SQL used (so we can re-run or export later)
        st.session_state["filtered_sql"] = sql_text
        st.success(f"Loaded preview with {len(df_preview)} rows (preview).")
        return True
    except Exception as e:
        st.error(f"SQL Error: {e}")
        st.text(traceback.format_exc())
        return False

if st.sidebar.button("Run Query"):
    ok = store_filtered_preview(sql_query_text, preview_limit=row_limit)
    # clear GC to free memory
    gc.collect()

# Provide a button to clear cached previews if needed
if st.sidebar.button("Clear Preview Cache"):
    st.session_state.pop("filtered_preview", None)
    st.session_state.pop("filtered_sql", None)
    st.success("Cleared preview from session state.")
    gc.collect()

# ============================================================
# Optionally enable RAG (build vectorstore lazily & limited)
# ============================================================
rag_enabled = st.sidebar.checkbox("Enable RAG (build vector store only on demand)")
if rag_enabled and not LANGCHAIN_AVAILABLE:
    st.sidebar.warning("LangChain/FAISS not available in environment. RAG disabled.")

# Build RAG only on user action to save memory/time
if rag_enabled and LANGCHAIN_AVAILABLE:
    if st.sidebar.button("Build RAG (may take a minute)"):
        with st.spinner("Building limited RAG (this may take a while)..."):
            try:
                # limit rows and columns to keep memory low
                limit_rows_for_rag = 2000
                engine = sqlite3.connect(DB_PATH_ITEMS)
                q = f'SELECT "audit id" as audit_id, "TemplateNames" as template, "region" as region, "owner name" as owner FROM "{items_table_name}" LIMIT {limit_rows_for_rag};'
                df_rag = pd.read_sql(q, engine)
                engine.close()

                docs = []
                for i, r in df_rag.iterrows():
                    texts = f"audit_id:{r.get('audit_id','')} | template:{r.get('template','')} | region:{r.get('region','')} | owner:{r.get('owner','')}"
                    docs.append(Document(page_content=texts, metadata={"row": i}))

                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vs = FAISS.from_documents(chunks, embeddings)
                retriever = vs.as_retriever(search_kwargs={"k": 5})
                # store retriever in session_state (cached)
                st.session_state["rag_retriever"] = retriever
                st.success("✅ RAG vectorstore built and cached (limited sample).")
            except Exception as e:
                st.error("❌ Failed to build RAG:")
                st.exception(e)
                gc.collect()

# ============================================================
# Main UI: left column for chat, right for visuals
# ============================================================
col_left, col_right = st.columns([1, 0.6])

with col_left:
    st.subheader("💬 Ask a Question About the Data")
    user_question = st.text_input("Enter your question:", key="user_question_input")

    # small helper: simple relevance detector (fallback)
    def detect_relevance_simple(q, df):
        ql = q.lower()
        for col in df.select_dtypes(include='object').columns:
            if col.lower() in ql:
                return True
            top_vals = df[col].dropna().astype(str).unique()[:5]
            if any(str(v).lower() in ql for v in top_vals):
                return True
        return False

    if st.button("Ask Chatbot (Analyze)"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                st.info("🤖 Understanding your question and generating query...")
    
                # STEP 1 — Convert natural query → SQL (use LLM)
                sql_prompt = f"""
    You are a SQL expert. Convert the following natural language question into an SQLite SQL query
    that can be run on a table named "{items_table_name}" with columns like:
    ['region', 'TemplateNames', 'owner name', 'assignee status', 'date completed', 'responses']
    
    Question: "{user_question}"
    
    Rules:
    - Only output the SQL query (no markdown or explanation).
    - Use correct column names if mentioned.
    - Include filtering by region, employee, template, or date if specified.
    - Limit results to 1000 rows for performance.
    """
                sql_query = ""
                try:
                    sql_out = llm.invoke(sql_prompt)
                    sql_query = sql_out.content.strip() if hasattr(sql_out, "content") else str(sql_out).strip()
                except Exception as e:
                    st.warning(f"⚠️ Could not generate SQL with LLM: {e}")
                    sql_query = ""
    
                if not sql_query or "SELECT" not in sql_query.upper():
                    st.warning("LLM did not generate a valid SQL query — running summary on full data instead.")
                    sql_query = f'SELECT * FROM "{items_table_name}" LIMIT 1000;'
    
                st.code(sql_query, language="sql")
    
                # STEP 2 — Run SQL safely
                df_result = run_sql_query(DB_PATH_ITEMS, sql_query)
                if df_result.empty:
                    st.warning("No data found for your query.")
                    st.stop()
    
                # Convert date columns safely
                if "date completed" in df_result.columns:
                    df_result["date completed"] = pd.to_datetime(df_result["date completed"], errors="coerce")
    
                # STEP 3 — KPIs
                st.markdown("### 📊 Key Metrics")
                total_records = len(df_result)
                unique_regions = df_result["region"].nunique() if "region" in df_result.columns else 0
                unique_templates = df_result["TemplateNames"].nunique() if "TemplateNames" in df_result.columns else 0
                unique_employees = df_result["owner name"].nunique() if "owner name" in df_result.columns else 0
                top_template = (
                    df_result["TemplateNames"].value_counts().idxmax()
                    if "TemplateNames" in df_result.columns and not df_result["TemplateNames"].empty
                    else "N/A"
                )
                top_employee = (
                    df_result["owner name"].value_counts().idxmax()
                    if "owner name" in df_result.columns and not df_result["owner name"].empty
                    else "N/A"
                )
    
                # Compute status counts
                completed_count = 0
                missed_count = 0
                late_count = 0
                if "assignee status" in df_result.columns:
                    status_counts = df_result["assignee status"].value_counts().to_dict()
                    completed_count = status_counts.get("Complete", 0)
                    missed_count = status_counts.get("Missed", 0)
                    late_count = status_counts.get("Late Complete", 0)
    
                col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
                col1.metric("Total Records", total_records)
                col2.metric("Regions", unique_regions)
                col3.metric("Templates", unique_templates)
                col4.metric("Employees", unique_employees)
                col5.metric("Top Template", top_template)
                col6.metric("Top Employee", top_employee)
                col7.metric("Completed", completed_count)
                col8.metric("Missed", missed_count)
    
                try:
                    style_metric_cards(
                        background_color="#f8f9fa",
                        border_color="#e0e0e0",
                        border_radius_px=12,
                    )
                except Exception:
                    pass
    
                # STEP 4 — Auto Visuals
                st.markdown("### 📈 Auto-Generated Visuals")
                vivid_colors = px.colors.qualitative.Vivid
    
                try:
                    if "region" in df_result.columns and "TemplateNames" in df_result.columns:
                        region_count = df_result.groupby("region")["TemplateNames"].count().reset_index(name="count")
                        fig = px.bar(
                            region_count, x="region", y="count", text="count",
                            color="region", color_discrete_sequence=vivid_colors,
                            title="Inspections by Region"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
                    if "TemplateNames" in df_result.columns:
                        template_count = df_result["TemplateNames"].value_counts().head(10).reset_index()
                        template_count.columns = ["TemplateNames", "count"]
                        fig = px.bar(
                            template_count, x="TemplateNames", y="count", text="count",
                            color="TemplateNames", title="Top 10 Templates by Count"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
                    if "owner name" in df_result.columns:
                        emp_count = df_result["owner name"].value_counts().head(10).reset_index()
                        emp_count.columns = ["owner name", "count"]
                        fig = px.bar(
                            emp_count, x="owner name", y="count", text="count",
                            color="owner name", title="Top 10 Employees by Inspection Count"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
                    if "assignee status" in df_result.columns:
                        status_count = df_result["assignee status"].value_counts().reset_index()
                        status_count.columns = ["assignee status", "count"]
                        fig = px.pie(
                            status_count, values="count", names="assignee status",
                            title="Distribution of Assignee Status",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
                except Exception as viz_err:
                    st.warning(f"⚠️ Could not render all visuals: {viz_err}")
    
                # STEP 5 — Analytical LLM Summary
                try:
                    summary_prompt = f"""
    You are a senior data analyst.
    Below is a preview of the dataset and its metrics.
    
    Sample rows:
    {df_result.head(10).to_json(orient="records")}
    
    KPIs:
    - Total Records: {total_records}
    - Unique Regions: {unique_regions}
    - Unique Templates: {unique_templates}
    - Top Template: {top_template}
    - Top Employee: {top_employee}
    - Completed Count: {completed_count}
    - Missed Count: {missed_count}
    - Late Count: {late_count}
    
    User Question: {user_question}
    
    Task:
    1. Summarize this data in context of the question.
    2. Highlight key KPIs, trends, and anomalies.
    3. Identify top/bottom performing regions, templates, or employees.
    4. Provide 2 actionable recommendations.
    5. Format the output as a markdown analytical report.
    """
                    analysis_out = llm.invoke(summary_prompt)
                    answer_text = analysis_out.content if hasattr(analysis_out, "content") else str(analysis_out)
                    st.markdown("### 🧠 Analytical Report")
                    st.markdown(answer_text)
                except Exception as e:
                    st.warning(f"⚠️ Could not generate AI analytical report: {e}")
    
                # STEP 6 — Show Data Preview
                st.markdown("### 📋 Filtered Data Preview")
                st.dataframe(df_result.head(50))
    
                # Cleanup
                del df_result
                gc.collect()
    
            except Exception as e:
                st.error(f"❌ Error running chatbot: {e}")
                st.text(traceback.format_exc())

    
    
    # ============================================================
    # Right Panel: Filtered Preview + Charts
    # ============================================================
    with col_right:
        st.subheader("📊 Filtered Data & Visualizations (Preview)")
    
        preview = st.session_state.get("filtered_preview")
        if preview is not None and not preview.empty:
            st.markdown("### 🔍 Preview (sample from Run Query)")
            st.dataframe(preview)
    
            # Completion by month chart
            try:
                if "date completed" in preview.columns:
                    preview2 = preview.assign(
                        completion_month=pd.to_datetime(
                            preview["date completed"], errors="coerce"
                        )
                        .dt.to_period("M")
                        .astype(str)
                    )
                    chart_df = (
                        preview2.groupby("completion_month")["TemplateNames"]
                        .count()
                        .reset_index(name="template_count")
                        .sort_values("completion_month")
                    )
                    st.markdown("### 📅 Inspections by Completion Month")
                    st.bar_chart(
                        chart_df.set_index("completion_month")["template_count"]
                    )
            except Exception:
                pass
    
            # Template Preview
            try:
                if "TemplateNames" in preview.columns:
                    template_count = (
                        preview["TemplateNames"].value_counts().head(10).reset_index()
                    )
                    template_count.columns = ["TemplateNames", "count"]
                    fig = px.bar(
                        template_count,
                        x="TemplateNames",
                        y="count",
                        text="count",
                        title="Top Templates (preview)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
        else:
            st.info("ℹ️ No preview available. Use sidebar filters and click 'Run Query'.")

# ============================================================
# Memory Debug (optional)
# ============================================================
if st.checkbox("Show memory usage (debug)", value=False):
    try:
        import psutil

        mem = psutil.virtual_memory()
        st.write(f"Memory used: {mem.used/1024**2:.1f} MB / {mem.total/1024**2:.1f} MB")
    except Exception:
        st.write("psutil not installed or accessible.")

# ============================================================
# End of file
# ============================================================

