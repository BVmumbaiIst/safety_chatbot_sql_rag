
import streamlit as st
# ============================================================
# Basic page config - must be set before ANY st.* call
# ============================================================
st.set_page_config(page_title="💬 Safety Chatbot (SQL + Optional RAG)", layout="wide")

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
# TITLE
# ============================================================

st.title("💬 Safety Chatbot — SQL + Optional RAG (Memory-Optimized)")
# ============================================================
# LOAD ENV
# ============================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================
# SESSION STATE INIT (VERY IMPORTANT)
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "email" not in st.session_state:
    st.session_state.email = None

if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False


# ============================================================
# S3 CONFIG
# ============================================================
BUCKET_NAME = "iauditorsafetydata"

S3_KEYS = {
    "items": "BVPI_Safety_Optimise/safety_Chat_bot_db/inspection_employee_schedule_items.db",
    "users": "BVPI_Safety_Optimise/safety_Chat_bot_db/inspection_employee_schedule_users.db"
}

s3 = boto3.client("s3")

# ============================================================
# CLEAN OLD TEMP FILES
# ============================================================
def cleanup_old_dbs(tmp_dir=tempfile.gettempdir(), hours=24):
    cutoff = time.time() - hours * 3600
    for file in os.listdir(tmp_dir):
        if file.endswith(".db"):
            path = os.path.join(tmp_dir, file)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except:
                pass

cleanup_old_dbs()
atexit.register(lambda: cleanup_old_dbs())

# ============================================================
# LAZY S3 DOWNLOAD
# ============================================================
@st.cache_data
def load_sqlite_from_s3(s3_key):
    filename = os.path.basename(s3_key)
    local_path = os.path.join(tempfile.gettempdir(), filename)

    if not os.path.exists(local_path):
        s3.download_file(BUCKET_NAME, s3_key, local_path)

    return local_path


@st.cache_data
def get_db_paths():
    return (
        load_sqlite_from_s3(S3_KEYS["items"]),
        load_sqlite_from_s3(S3_KEYS["users"])
    )
# ============================================================
# USE DB FROM SESSION
# ============================================================
DB_PATH_ITEMS = st.session_state.DB_PATH_ITEMS
items_meta = st.session_state.items_meta
# ============================================================
# LOAD METADATA SAFELY
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_db_metadata(db_path, s3_key=None):

    def _read_metadata(path):
        conn = sqlite3.connect(path)
        try:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';",
                conn
            )

            if tables.empty:
                return None

            table_name = tables.iloc[0]["name"]

            try:
                meta_df = pd.read_sql(
                    f'SELECT MIN("date completed") as date_min, MAX("date completed") as date_max FROM "{table_name}";',
                    conn
                )
                date_min = meta_df["date_min"].iloc[0]
                date_max = meta_df["date_max"].iloc[0]
            except Exception:
                date_min, date_max = None, None

            distincts = {}
            cols = [
                "region",
                "TemplateNames",
                "owner name",
                "assignee status",
                "employee status",
                "email",
            ]

            for c in cols:
                try:
                    q = f'SELECT DISTINCT "{c}" as val FROM "{table_name}" WHERE "{c}" IS NOT NULL LIMIT 2000;'
                    vals = pd.read_sql(q, conn)["val"].dropna().tolist()
                    distincts[c] = sorted(vals)
                except Exception:
                    distincts[c] = []

            return {
                "table": table_name,
                "meta": {"date_min": date_min, "date_max": date_max},
                "distincts": distincts,
            }

        finally:
            conn.close()

    # Try reading metadata
    meta = _read_metadata(db_path)

    if meta is not None:
        return meta

    # If DB corrupted or empty → redownload
    if s3_key:
        try:
            os.remove(db_path)
        except Exception:
            pass

        new_path = load_sqlite_from_s3(s3_key)

        meta = _read_metadata(new_path)

        if meta is not None:
            return meta

    raise RuntimeError("Database could not be loaded or contains no tables")



# ============================================================
# DOWNLOAD SQLITE FROM S3 (SAFE)
# ============================================================
def load_sqlite_from_s3(s3_key: str):

    filename = os.path.basename(s3_key)
    local_path = os.path.join(tempfile.gettempdir(), filename)

    # If file missing or corrupted → redownload
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 1024:
        if os.path.exists(local_path):
            os.remove(local_path)

        s3.download_file(BUCKET_NAME, s3_key, local_path)

    # Validate tables
    conn = sqlite3.connect(local_path)
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';",
        conn
    )
    conn.close()

    if tables.empty:
        os.remove(local_path)
        s3.download_file(BUCKET_NAME, s3_key, local_path)

    return local_path



# ============================================================
# SAFE SQL EXECUTION
# ============================================================
def run_sql_query(db_path, sql, params=None, limit_rows=None):
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        if limit_rows:
            sql = sql.rstrip().rstrip(";")
            sql = f"{sql} LIMIT {limit_rows};"
        df = pd.read_sql(sql, conn, params=params)
        return df
    finally:
        conn.close()


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
# LOGIN (SAFE + DYNAMIC TABLE)
# ============================================================
with st.sidebar:
    st.header("🔑 Login")

    email_input = st.text_input("Enter your Email")

    if st.button("Login"):

        if not email_input:
            st.warning("Enter email")
        else:
            try:
                DB_PATH_ITEMS, DB_PATH_USERS = get_db_paths()

                conn = sqlite3.connect(DB_PATH_USERS)

                # 🔥 get table safely
                table_df = pd.read_sql(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';",
                    conn
                )

                if table_df.empty:
                    st.error("❌ No tables found in USERS DB")
                    st.stop()

                tables = table_df["name"].tolist()

                # smart selection
                table_name = next((t for t in tables if "user" in t.lower()), tables[0])

                query = f'SELECT 1 FROM "{table_name}" WHERE LOWER(email)=? LIMIT 1'

                result = pd.read_sql(query, conn, params=[email_input.lower()])

                conn.close()

                if not result.empty:
                    st.session_state.logged_in = True
                    st.session_state.email = email_input
                    st.session_state.db_loaded = False
                    st.success("✅ Login successful")
                    st.experimental_rerun()
                else:
                    st.error("❌ Access denied")

            except Exception as e:
                st.error("❌ Login failed")
                st.exception(e)
# ============================================================
# STOP IF NOT LOGGED IN
# ============================================================
if not st.session_state.logged_in:
    st.warning("🔒 Please login to continue.")
    st.stop()

# ============================================================
# LAZY LOAD DB AFTER LOGIN
# ============================================================
if not st.session_state.db_loaded:

    with st.spinner("Loading database..."):
        try:
            DB_PATH_ITEMS, DB_PATH_USERS = get_db_paths()

            items_meta = load_db_metadata(DB_PATH_ITEMS, S3_KEYS["items"])

            # store in session
            st.session_state.DB_PATH_ITEMS = DB_PATH_ITEMS
            st.session_state.items_meta = items_meta

            st.session_state.db_loaded = True

        except Exception as e:
            st.error("❌ Failed to load DB")
            st.exception(e)
            st.stop()

# ============================================================
# FILTERS
# ============================================================
st.sidebar.header("🔎 Filters")

date_min = items_meta["meta"].get("date_min")
date_max = items_meta["meta"].get("date_max")

if date_min and date_max:
    date_range = st.sidebar.date_input(
        "Date Range",
        value=[
            pd.to_datetime(date_min).date(),
            pd.to_datetime(date_max).date()
        ]
    )
else:
    date_range = None

regions = st.sidebar.multiselect("Region", items_meta["distincts"]["region"])
templates = st.sidebar.multiselect("Template", items_meta["distincts"]["TemplateNames"])
employees = st.sidebar.multiselect("Owner", items_meta["distincts"]["owner name"])
statuses = st.sidebar.multiselect("Assignee Status", items_meta["distincts"]["assignee status"])
employee_status = st.sidebar.multiselect("Employee Status", items_meta["distincts"]["employee status"])

row_limit = st.sidebar.slider(
    "Preview Row Limit",
    10,
    5000,
    500
)


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
def sql_list(values):
    safe = [str(v).replace("'", "''") for v in values]
    return ",".join([f"'{s}'" for s in safe])


def build_where_clause(selected_filters, date_range_tuple=None):

    filters = []

    for col, vals in selected_filters.items():
        if vals:
            filters.append(f'"{col}" IN ({sql_list(vals)})')

    if date_range_tuple and isinstance(date_range_tuple, (list, tuple)) and len(date_range_tuple) == 2:
        start = pd.to_datetime(date_range_tuple[0]).strftime("%Y-%m-%d")
        end = pd.to_datetime(date_range_tuple[1]).strftime("%Y-%m-%d")
        filters.append(f'"date completed" BETWEEN "{start}" AND "{end}"')

    return " AND ".join(filters) if filters else "1=1"

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
        if not user_question or len(user_question.strip()) == 0:
            st.warning("Please enter a question.")
        else:
            try:
                # CASE A: If preview SQL exists, analyze filtered preview (or run limited query)
                if st.session_state.get("filtered_sql"):
                    # run a slightly larger limited query for analysis (bounded)
                    analysis_limit = min(2000, max(200, row_limit))
                    analysis_sql = st.session_state["filtered_sql"].rstrip().rstrip(";")
                    analysis_sql = f"{analysis_sql} LIMIT {analysis_limit};"
                    df_for_analysis = run_sql_query(DB_PATH_ITEMS, analysis_sql)

                    # Generat summary string for LLM fallback
                    try:
                        numeric_summary = (
                            df_for_analysis.describe(include=[np.number]).transpose().round(2)
                        )
                        categorical_summary = {
                            col: df_for_analysis[col].value_counts().head(5).to_dict()
                            for col in df_for_analysis.select_dtypes(include='object').columns
                        }
                        summary = f"Numerical Summary:\n{numeric_summary.to_string()}\n\nTop categories:\n{json.dumps(categorical_summary, indent=2)}"
                    except Exception:
                        summary = f"Rows: {len(df_for_analysis)}"

                    # ------------------------
                    # KPIs SECTION
                    # ------------------------
                    st.markdown("### 📊 Key Metrics")
                    total_records = len(df_for_analysis)
                    unique_regions = (
                        df_for_analysis["region"].nunique()
                        if "region" in df_for_analysis.columns
                        else 0
                    )
                    template_count = (
                        df_for_analysis["TemplateNames"].nunique()
                        if "TemplateNames" in df_for_analysis.columns
                        else 0
                    )
                    employee_count = (
                        df_for_analysis["owner name"].nunique()
                        if "owner name" in df_for_analysis.columns
                        else 0
                    )
                    top_template = (
                                df_for_analysis["TemplateNames"].value_counts().idxmax()
                                if "TemplateNames" in df_for_analysis.columns and not df_for_analysis["TemplateNames"].isna().all()
                                else "N/A"
                                )
                
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Records", total_records)
                    col2.metric("Unique Regions", unique_regions)
                    col3.metric("Template", template_count)
                    col4.metric("Employee", employee_count)

                    # Style metric cards (optional)
                    try:
                        style_metric_cards(
                            background_color="#f8f9fa",
                            border_color="#e0e0e0",
                            border_radius_px=12,
                        )
                    except Exception:
                        pass

                    # ------------------------
                    # Relevance detection
                    # ------------------------
                    relevance = False
                    if llm:
                        try:
                            preview_rows = df_for_analysis.head(5).to_dict(orient="records")
                            prompt = f"""You are an assistant. Preview rows: {json.dumps(preview_rows)}. Question: "{user_question}". Respond ONLY RELATED or UNRELATED."""
                            out = llm.invoke(prompt).strip().upper()
                            relevance = "RELATED" in out
                        except Exception:
                            relevance = detect_relevance_simple(user_question, df_for_analysis)
                    else:
                        relevance = detect_relevance_simple(user_question, df_for_analysis)

                    # ------------------------
                    # Build LLM analytical prompt (safe: kpi_text always defined)
                    # ------------------------
                    kpi_text = f"Total Records: {total_records}\nTop Template: {template_count}\nTop Employee: {employee_count}"

                    if llm:
                        task_prompt = f"""
You are a senior data analyst. Summary:
{summary}

KPIs:
{kpi_text}

User question: {user_question}

Requirements:
1) Answer the question clearly using the filtered data.
2) Show KPI counts and percentage breakdowns for TemplateNames (top few).
3) Highlight top/bottom regions, templates, responses, assignee status, employees (if available).
4) Suggest 2 actionable recommendations.
5) If possible, compare selected month vs previous month deviations.

Return a concise markdown-formatted report.
"""
                        try:
                            llm_out = llm.invoke(task_prompt)
                            answer_text = llm_out.content if hasattr(llm_out, "content") else str(llm_out)
                        except Exception as e:
                            answer_text = f"❌ LLM error: {e}\n\nFallback summary:\n{summary}"
                    else:
                        # fallback simple analysis when LLM not available
                        top_templates = df_for_analysis["TemplateNames"].value_counts().head(5).to_dict() if "TemplateNames" in df_for_analysis.columns else {}
                        answer_text = f"Fallback analysis (LLM not available).\nRows: {len(df_for_analysis)}\nTop templates: {top_templates}"

                    st.markdown("### 📋 Chatbot Response")
                    st.markdown(answer_text)

                    # ------------------------
                    # AUTO-GENERATED VISUALS (based on df_for_analysis)
                    # ------------------------
                    st.markdown("### 📈 Auto-Generated Visual Insights")
                    vivid_colors = px.colors.qualitative.Vivid
                    df_viz = df_for_analysis  # alias

                    try:
                        # Region bar
                        if "region" in df_viz.columns and "TemplateNames" in df_viz.columns:
                            region_count = df_viz.groupby("region")["TemplateNames"].count().reset_index(name="count")
                            fig = px.bar(region_count, x="region", y="count", text="count", color="region", color_discrete_sequence=vivid_colors, title="Inspections by Region")
                            st.plotly_chart(fig, width="stretch")

                        # Template distribution
                        if "TemplateNames" in df_viz.columns:
                            template_count = df_viz["TemplateNames"].value_counts().head(10).reset_index()
                            template_count.columns = ["TemplateNames", "count"]
                            fig = px.bar(template_count, x="TemplateNames", y="count", color="TemplateNames", text="count", title="Top 10 Templates by Inspection Count")
                            st.plotly_chart(fig, width="stretch")

                        # Employee distribution
                        if "owner name" in df_viz.columns:
                            emp_count = df_viz["owner name"].value_counts().head(10).reset_index()
                            emp_count.columns = ["owner name", "count"]
                            fig = px.bar(emp_count, x="owner name", y="count", color="owner name", text="count", title="Top 10 Employees by Inspection Count")
                            st.plotly_chart(fig, width="stretch")

                        # Assignee status pie
                        if "assignee status" in df_viz.columns:
                            status_count = df_viz["assignee status"].value_counts().reset_index()
                            status_count.columns = ["assignee status", "count"]
                            fig = px.pie(status_count, values="count", names="assignee status", title="Distribution of Assignee Status", color_discrete_sequence=px.colors.qualitative.Set3)
                            st.plotly_chart(fig, width="stretch")

                    except Exception as e:
                        st.warning(f"⚠️ Could not generate visuals automatically: {e}")

                    # cleanup
                    del df_for_analysis
                    gc.collect()

                else:
                    # CASE B: No filtered SQL — run a small aggregated query on full items
                    st.info("No filtered query found. Running a small aggregated analysis on the items table.")
                    agg_sql = f'SELECT "TemplateNames", COUNT(*) as cnt FROM "{items_table_name}" GROUP BY "TemplateNames" ORDER BY cnt DESC LIMIT 20;'
                    agg_df = run_sql_query(DB_PATH_ITEMS, agg_sql)
                    st.markdown("### Top Templates (aggregated)")
                    st.dataframe(agg_df)

            except Exception as outer_e:
                st.error(f"❌ Chatbot failed: {outer_e}")
                st.text(traceback.format_exc())
                gc.collect()
    
    
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
                    st.plotly_chart(fig, width="stretch")
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



