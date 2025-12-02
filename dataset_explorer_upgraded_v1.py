# streamlit run dataset_explorer_upgraded_v1.py
#  directory setup: cd C:\users\oakhtar\documents\pyprojs_local



# python_datahub_dataset_relationships_v200_AI_EDITION.py
# THE ULTIMATE BRIGHTSPACE DATASET EXPLORER — AI POWERED
# Local test: streamlit run python_datahub_dataset_relationships_v200_AI_EDITION.py

import pandas as pd
import re
import streamlit as st
import os
import logging
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import plotly.graph_objects as go
import json
import openai
from streamlit_plotly_events import plotly_events

# ========================= CONFIG & SECRETS =========================
logging.basicConfig(level=logging.INFO)
requests.packages.urllib3.disable_warnings()

# Secure API key loading (same pattern as your physics app)
def get_api_key(provider: str):
    if provider == "OpenAI":
        key = st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if key:
            return key, "https://api.openai.com/v1", "gpt-4o"
    elif provider == "xAI (Grok)":
        key = st.secrets.get("xai_api_key") or os.getenv("XAI_API_KEY")
        if key:
            return key, "https://api.x.ai/v1", "grok-beta"
    return None, None, None

# ========================= DEFAULT URLS =========================
DEFAULT_URLS = """https://community.d2l.com/brightspace/kb/articles/4752-accommodations-data-sets
https://community.d2l.com/brightspace/kb/articles/4712-activity-feed-data-sets
https://community.d2l.com/brightspace/kb/articles/4723-announcements-data-sets
https://community.d2l.com/brightspace/kb/articles/4767-assignments-data-sets
https://community.d2l.com/brightspace/kb/articles/4519-attendance-data-sets
https://community.d2l.com/brightspace/kb/articles/4520-awards-data-sets
https://community.d2l.com/brightspace/kb/articles/4521-calendar-data-sets
https://community.d2l.com/brightspace/kb/articles/4523-checklist-data-sets
https://community.d2l.com/brightspace/kb/articles/4754-competency-data-sets
https://community.d2l.com/brightspace/kb/articles/4713-content-data-sets
https://community.d2l.com/brightspace/kb/articles/22812-content-service-data-sets
https://community.d2l.com/brightspace/kb/articles/26020-continuous-professional-development-cpd-data-sets
https://community.d2l.com/brightspace/kb/articles/4725-course-copy-data-sets
https://community.d2l.com/brightspace/kb/articles/4524-course-publisher-data-sets
https://community.d2l.com/brightspace/kb/articles/26161-creator-data-sets
https://community.d2l.com/brightspace/kb/articles/4525-discussions-data-sets
https://community.d2l.com/brightspace/kb/articles/4526-exemptions-data-sets
https://community.d2l.com/brightspace/kb/articles/4527-grades-data-sets
https://community.d2l.com/brightspace/kb/articles/4528-intelligent-agents-data-sets
https://community.d2l.com/brightspace/kb/articles/5782-jit-provisioning-data-sets
https://community.d2l.com/brightspace/kb/articles/4714-local-authentication-data-sets
https://community.d2l.com/brightspace/kb/articles/4727-lti-data-sets
https://community.d2l.com/brightspace/kb/articles/4529-organizational-units-data-sets
https://community.d2l.com/brightspace/kb/articles/4796-outcomes-data-sets
https://community.d2l.com/brightspace/kb/articles/4530-portfolio-data-sets
https://community.d2l.com/brightspace/kb/articles/4531-questions-data-sets
https://community.d2l.com/brightspace/kb/articles/4532-quizzes-data-sets
https://community.d2l.com/brightspace/kb/articles/4533-release-conditions-data-sets
https://community.d2l.com/brightspace/kb/articles/33182-reoffer-course-data-sets
https://community.d2l.com/brightspace/kb/articles/4534-role-details-data-sets
https://community.d2l.com/brightspace/kb/articles/4535-rubrics-data-sets
https://community.d2l.com/brightspace/kb/articles/4536-scorm-data-sets
https://community.d2l.com/brightspace/kb/articles/4537-sessions-and-system-access-data-sets
https://community.d2l.com/brightspace/kb/articles/19147-sis-course-merge-data-sets
https://community.d2l.com/brightspace/kb/articles/4538-surveys-data-sets
https://community.d2l.com/brightspace/kb/articles/4540-tools-data-sets
https://community.d2l.com/brightspace/kb/articles/4740-users-data-sets
https://community.d2l.com/brightspace/kb/articles/4541-virtual-classroom-data-sets""".strip()

# ========================= SCRAPING FUNCTIONS (unchanged) =========================
def parse_urls_from_text_area(text_block):
    urls = [line.strip() for line in text_block.split('\n') if line.strip() and line.startswith('http')]
    return sorted(list(set(urls)))

def scrape_table(url, category_name):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        data = []
        current_dataset = category_name
        for element in soup.find_all(['h2', 'h3', 'table']):
            if element.name in ['h2', 'h3']:
                current_dataset = element.text.strip().lower()
            elif element.name == 'table':
                headers = [th.text.strip().lower().replace(' ', '_') for th in element.find_all('th')]
                if not headers: continue
                for row in element.find_all('tr'):
                    cols = row.find_all('td')
                    if len(cols) != len(headers): continue
                    entry = {headers[i]: cols[i].text.strip() for i in range(len(headers))}
                    entry = {('column_name' if k in ['field','name'] else 'data_type' if k=='type' else k): v for k,v in entry.items()}
                    if 'column_name' not in entry or not entry['column_name']: continue
                    entry['dataset_name'] = current_dataset
                    entry['category'] = category_name
                    data.append(entry)
        return pd.DataFrame(data).to_dict('records') if data else []
    except Exception as e:
        logging.error(f"Scrape failed {url}: {e}")
        return []

def scrape_and_save_from_list(url_list):
    all_data = []
    progress = st.progress(0)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for i, url in enumerate(url_list):
            cat = re.sub(r'^\d+\s*', '', os.path.basename(url).replace('-data-sets','').replace('-',' ')).lower()
            futures[executor.submit(scrape_table, url, cat)] = i
        for i, future in enumerate(futures):
            all_data.extend(future.result())
            progress.progress((i + 1) / len(futures))
    progress.empty()
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df.columns = [c.lower().replace(' ','_') for c in df.columns]
    for col in ['category','dataset_name','column_name','data_type','description','key','version','version_history','column_size','notes']:
        if col not in df.columns: df[col] = ''
    df = df.fillna('')
    df['is_primary_key'] = df['key'].str.contains('pk', case=False, na=False)
    df['is_foreign_key'] = df['key'].str.contains('fk', case=False, na=False)
    df.to_csv('dataset_metadata.csv', index=False)
    return df

@st.cache_data
def find_pk_fk_joins(df, selected_datasets):
    if df.empty or not selected_datasets: return pd.DataFrame()
    pks = df[df['is_primary_key']]
    fks = df[df['is_foreign_key'] & df['dataset_name'].isin(selected_datasets)]
    if pks.empty or fks.empty: return pd.DataFrame()
    merged = pd.merge(fks, pks, on='column_name', suffixes=('_fk', '_pk'))
    merged = merged[merged['dataset_name_fk'] != merged['dataset_name_pk']]
    result = merged[['dataset_name_fk', 'column_name', 'dataset_name_pk', 'category_pk']].copy()
    result.columns = ['Source Dataset', 'Join Column', 'Target Dataset', 'Target Category']
    return result.drop_duplicates().reset_index(drop=True)

# ========================= AI FUNCTIONS =========================
def call_llm(messages, api_key, base_url, model):
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        response_format={"type": "json_object"} if "json" in messages[0]["content"] else None
    )
    return response.choices[0].message.content

@st.cache_data(ttl=3600)
def ai_find_datasets(question: str, metadata_str: str, api_key, base_url, model):
    prompt = f"""You are a Brightspace Data Hub expert.
Given this schema and the user question, return a JSON object with:
{{"datasets": ["list of exact dataset names"], "columns": {{"dataset_name": ["relevant columns"]}}}}

Schema:
{metadata_str}

Question: {question}
"""
    try:
        resp = call_llm([{"role": "user", "content": prompt}], api_key, base_url, model)
        return json.loads(resp)
    except:
        return {"datasets": [], "columns": {}}

@st.cache_data(ttl=3600)
def ai_generate_sql(selected_datasets, goal, api_key, base_url, model):
    schema = "\n".join([f"{row['dataset_name']}: {row['column_name']} ({row['data_type']}) - {row['description']}"
                        for _, row in columns[columns['dataset_name'].isin(selected_datasets)].iterrows()])
    prompt = f"""Write a complete, valid SQL query for Brightspace Data Hub.
Selected tables: {', '.join(selected_datasets)}
Goal: {goal}

Schema:
{schema}

Return ONLY the SQL query. Use proper JOINs on known PK/FK columns."""
    try:
        return call_llm([{"role": "user", "content": prompt}], api_key, base_url, model)
    except:
        return "-- Failed to generate SQL"

# ========================= MAIN APP =========================
def main():
    st.set_page_config(page_title="Brightspace Dataset Explorer v200 AI", layout="wide", page_icon="🧠")
    st.title("🧠 Brightspace Dataset Explorer v200 — AI Edition")

    # ====================== SIDEBAR ======================
    with st.sidebar:
        st.header("⚙️ AI Provider")
        provider = st.radio("Choose Model", ["OpenAI (gpt-4o)", "xAI (Grok)"])
        api_key, base_url, model = get_api_key(provider.split()[0])
        if not api_key:
            st.error("API key not found. Add to secrets.toml or env vars.")
            st.stop()

        st.divider()
        st.header("📊 Data Controls")
        with st.expander("STEP 1: Load/Update Data", expanded=False):
            pasted = st.text_area("URLs (one per line)", height=200, value=DEFAULT_URLS)
            if st.button("Scrape & Update", type="primary"):
                urls = parse_urls_from_text_area(pasted)
                with st.spinner(f"Scraping {len(urls)} pages..."):
                    df = scrape_and_save_from_list(urls)
                    st.success("Data updated!")
                    st.rerun()

    # ====================== LOAD DATA ======================
    if not os.path.exists("dataset_metadata.csv"):
        st.warning("No data found. Use STEP 1 to scrape.")
        st.stop()
    columns = pd.read_csv("dataset_metadata.csv").fillna("")

    datasets = sorted(columns['dataset_name'].unique())
    categories = sorted(columns['category'].unique())

    # ====================== GLOBAL SEARCH ======================
    st.subheader("🔍 Global Search")
    search = st.text_input("Search any column, description, or key across all datasets", "")
    if search:
        mask = columns.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
        results = columns[mask][['dataset_name', 'column_name', 'description', 'key']].head(50)
        st.dataframe(results, use_container_width=True)

    # ====================== AI NATURAL LANGUAGE SEARCH ======================
    st.subheader("🧠 Ask AI: Find Datasets & Columns")
    question = st.text_input("e.g. late quiz submissions with grades and penalties")
    if question:
        with st.spinner("Thinking..."):
            metadata_preview = columns[['dataset_name','column_name','description']].drop_duplicates().to_csv(index=False)
            result = ai_find_datasets(question, metadata_preview, api_key, base_url, model)
            if result.get("datasets"):
                st.success(f"Found {len(result['datasets'])} relevant datasets")
                selected_datasets = st.multiselect("Auto-selected datasets", datasets, default=result["datasets"])
            else:
                st.info("No strong matches found.")

    # ====================== MANUAL SELECTION ======================
    selected_categories = st.multiselect("Filter by Category", categories, default=[])
    filtered = columns[columns['category'].isin(selected_categories)]['dataset_name'].unique() if selected_categories else datasets
    selected_datasets = st.multiselect("Select Datasets", filtered, default=selected_datasets if 'selected_datasets' in locals() else [])

    # ====================== MOST CONNECTED ======================
    if not columns.empty:
        joins = find_pk_fk_joins(columns, datasets)
        if not joins.empty:
            top = joins['Source Dataset'].value_counts().head(1)
            st.sidebar.metric("🔥 Most Connected Dataset", top.index[0], f"{top.iloc[0]} outgoing FKs")

    # ====================== GRAPH ======================
    st.subheader("Graph Explorer")
    graph_mode = st.radio("Mode", ("Focused (between selected)", "Discovery (outgoing)"), horizontal=True)

    if not selected_datasets:
        st.info("Select datasets above to begin.")
    else:
        join_data = find_pk_fk_joins(columns, selected_datasets)
        G = nx.DiGraph()

        for ds in selected_datasets:
            G.add_node(ds, type='focus')

        if graph_mode.startswith("Focused"):
            for _, row in join_data.iterrows():
                if row['Source Dataset'] in selected_datasets and row['Target Dataset'] in selected_datasets:
                    G.add_edge(row['Source Dataset'], row['Target Dataset'], label=row['Join Column'], sql=f"INNER JOIN {row['Target Dataset']} ON {row['Source Dataset']}.{row['Join Column']} = {row['Target Dataset']}.{row['Join Column']}")
        else:
            for _, row in join_data.iterrows():
                if row['Source Dataset'] in selected_datasets:
                    G.add_node(row['Target Dataset'], type='neighbor')
                    G.add_edge(row['Source Dataset'], row['Target Dataset'], label=row['Join Column'], sql=f"INNER JOIN {row['Target Dataset']} ON {row['Source Dataset']}.{row['Join Column']} = {row['Target Dataset']}.{row['Join Column']}")

        if G.nodes():
            pos = nx.spring_layout(G, k=1, iterations=50)
            edge_traces = []
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                sql = edge[2].get('sql', '')
                edge_trace = go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                        mode='lines', line=dict(color='#888', width=2),
                                        hoverinfo='text', hovertext=sql,
                                        customdata=[sql], showlegend=False)
                edge_traces.append(edge_trace)

            node_trace = go.Scatter(x=[], y=[], text=[], mode='markers+text', hoverinfo='text',
                                    marker=dict(size=40, color=[], line=dict(width=3, color=[])),
                                    textfont=dict(size=16, color='white'))

            cat_colors = {c: f"hsl({(hash(c)*137) % 360}, 70%, 60%)" for c in categories}
            for node in G.nodes():
                x, y = pos[node]
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])
                cat = columns[columns['dataset_name']==node]['category'].iloc[0] if not columns[columns['dataset_name']==node].empty else ''
                color = cat_colors.get(cat, '#ccc')
                node_trace['marker']['color'] += tuple([color])
                node_trace['marker']['line']['color'] += tuple(['white' if G.nodes[node].get('type')=='focus' else 'gray'])
                node_trace['text'] += tuple([f"<b>{node}</b>"])

            fig = go.Figure(data=edge_traces + [node_trace],
                            layout=go.Layout(showlegend=False, hovermode='closest',
                                             paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e',
                                             xaxis=dict(showgrid=False, zeroline=False),
                                             yaxis=dict(showgrid=False, zeroline=False),
                                             height=700))

            selected_points = plotly_events(fig, click_event=True, hover_event=False)
            if selected_points:
                point = selected_points[0]
                if 'customdata' in point and point['customdata']:
                    sql = point['customdata'][0]
                    st.code(sql, language='sql')
                    st.button("Copy SQL to clipboard", on_click=lambda: st.write(f"<script>navigator.clipboard.writeText(`{sql}`)</script>", unsafe_allow_html=True))

            st.plotly_chart(fig, use_container_width=True)

    # ====================== AI SQL GENERATOR ======================
    if selected_datasets:
        st.subheader("🛠️ Generate SQL Query")
        goal = st.text_input("What do you want to analyze?", placeholder="e.g. students with late submissions and grade penalties")
        if goal and st.button("Generate SQL"):
            with st.spinner("Writing query..."):
                sql = ai_generate_sql(selected_datasets, goal, api_key, base_url, model)
                st.code(sql, language='sql')

    # ====================== DATASET DETAILS + AI EXPLAIN ======================
    st.subheader("Dataset Details")
    for ds in selected_datasets:
        with st.expander(f"📋 {ds}", expanded=False):
            df = columns[columns['dataset_name']==ds]
            st.dataframe(df[['column_name','data_type','description','key']], use_container_width=True)
            if st.button(f"Explain {ds} in plain English", key=f"exp_{ds}"):
                with st.spinner("Explaining..."):
                    prompt = f"Explain this Brightspace dataset in simple terms for a non-technical department admin:\n{ds}\nColumns:\n{df[['column_name','description']].to_csv(index=False)}"
                    explanation = call_llm([{"role": "user", "content": prompt}], api_key, base_url, model)
                    st.write(explanation)

    # ====================== CHAT WITH YOUR DATA ======================
    st.subheader("💬 Chat with Brightspace Data")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask anything about Brightspace data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                metadata_snippet = columns[['dataset_name','column_name','description']].drop_duplicates().head(100).to_csv(index=False)
                full_prompt = f"Schema sample:\n{metadata_snippet}\n\nQuestion: {prompt}\nAnswer helpfully and concisely."
                response = call_llm([{"role": "user", "content": full_prompt}], api_key, base_url, model)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()