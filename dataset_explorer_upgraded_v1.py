# streamlit run dataset_explorer_upgraded_v1.py
#  directory setup: cd C:\users\oakhtar\documents\pyprojs_local
import streamlit as st
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import plotly.graph_objects as go
import openai
import re
import logging

# ========================= CONFIG & LOGGING =========================
st.set_page_config(page_title="Brightspace Explorer & AI", layout="wide", page_icon="🕸️")

logging.basicConfig(filename='scraper.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# ========================= PASSWORD PROTECTION =========================
def check_password():
    pwd = st.secrets.get("app_password")
    if not pwd: return True 
    if st.session_state.get("authenticated", False): return True
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 Login")
        with st.form("login_form"):
            user_input = st.text_input("Password", type="password", autocomplete="current-password")
            if st.form_submit_button("Submit"):
                if user_input == pwd:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password")
    st.stop()

check_password()

# ========================= RESTORED: SCRAPE SUCCESS MESSAGE =========================
if 'scrape_msg' in st.session_state:
    st.success(st.session_state['scrape_msg'])
    del st.session_state['scrape_msg']

# ========================= DEFAULT URLs =========================
DEFAULT_URLS = """
https://community.d2l.com/brightspace/kb/articles/4752-accommodations-data-sets
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
https://community.d2l.com/brightspace/kb/articles/4541-virtual-classroom-data-sets
""".strip()

# ========================= SCRAPING LOGIC =========================
def parse_urls_from_text_area(text_block):
    urls = [line.strip() for line in text_block.split('\n') if line.strip()]
    valid_urls = [url for url in urls if url.startswith('http')]
    return sorted(list(set(valid_urls)))

def scrape_table(url, category_name):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        if response.status_code != 200: return []
        soup = BeautifulSoup(response.content, 'html.parser')
        data = []
        elements = soup.find_all(['h2', 'h3', 'table'])
        current_dataset = category_name
        for element in elements:
            if element.name in ['h2', 'h3']: 
                text = element.text.strip()
                if len(text) > 3: current_dataset = text.lower()
            elif element.name == 'table':
                table_headers = [th.text.strip().lower().replace(' ', '_') for th in element.find_all('th')]
                if not table_headers or not any(x in table_headers for x in ['type', 'description', 'data_type']): continue
                for row in element.find_all('tr'):
                    columns_ = row.find_all('td')
                    if len(columns_) < len(table_headers): continue 
                    entry = {}
                    for i, header in enumerate(table_headers):
                        if i < len(columns_): entry[header] = columns_[i].text.strip()
                    header_map = {'field': 'column_name', 'name': 'column_name', 'type': 'data_type'}
                    entry = {header_map.get(k, k): v for k, v in entry.items()}
                    if 'column_name' in entry and entry['column_name']:
                        entry['dataset_name'] = current_dataset
                        entry['category'] = category_name
                        data.append(entry)
        return data
    except Exception: return []

def scrape_and_save_from_list(url_list):
    all_data = []
    progress_bar = st.progress(0, "Initializing Scraper...")
    
    def get_category_from_url(url):
        return re.sub(r'^\d+\s*', '', os.path.basename(url).split('?')[0].replace('-data-sets', '').replace('-', ' ')).lower()

    with ThreadPoolExecutor(max_workers=10) as executor:
        args = [(url, get_category_from_url(url)) for url in url_list]
        future_to_url = {executor.submit(scrape_table, *arg): arg[0] for arg in args}
        for i, future in enumerate(future_to_url):
            try:
                result = future.result()
                all_data.extend(result)
            except Exception: pass
            progress_bar.progress((i + 1) / len(url_list), f"Scraping {i+1}/{len(url_list)}...")
    progress_bar.empty()
    if not all_data: return pd.DataFrame()

    df = pd.DataFrame(all_data)
    expected_cols = ['category', 'dataset_name', 'column_name', 'data_type', 'description', 'key']
    for col in expected_cols:
        if col not in df.columns: df[col] = ''
    df = df.fillna('')
    df['dataset_name'] = df['dataset_name'].astype(str).str.title()
    df['category'] = df['category'].astype(str).str.title()
    df['is_primary_key'] = df['key'].astype(str).str.contains(r'\bpk\b', case=False, regex=True)
    df['is_foreign_key'] = df['key'].astype(str).str.contains(r'\bfk\b', case=False, regex=True)
    df.to_csv('dataset_metadata.csv', index=False)
    return df

@st.cache_data
def find_pk_fk_joins(df, selected_datasets):
    if df.empty or not selected_datasets: return pd.DataFrame()
    pks = df[df['is_primary_key'] == True]
    fks = df[(df['is_foreign_key'] == True) & (df['dataset_name'].isin(selected_datasets))]
    if pks.empty or fks.empty: return pd.DataFrame()
    merged = pd.merge(fks, pks, on='column_name', suffixes=('_fk', '_pk'))
    joins = merged[merged['dataset_name_fk'] != merged['dataset_name_pk']]
    if joins.empty: return pd.DataFrame()
    result = joins[['dataset_name_fk', 'column_name', 'dataset_name_pk', 'category_pk']]
    result.columns = ['Source Dataset', 'Join Column', 'Target Dataset', 'Target Category']
    return result.drop_duplicates().reset_index(drop=True)

# ========================= SIDEBAR & MAIN LOGIC =========================
if os.path.exists('dataset_metadata.csv'):
    df = pd.read_csv('dataset_metadata.csv').fillna('')
else:
    df = pd.DataFrame()

with st.sidebar:
    st.title("Brightspace Explorer")
    
    # --- AI Section ---
    with st.expander("🤖 AI Settings", expanded=False):
        ai_provider = st.radio("Provider", ["OpenAI (GPT-4o)", "xAI (Grok)"])
        if "OpenAI" in ai_provider:
            api_key_name = "openai_api_key"
            base_url = None 
            model_name = "gpt-4o"
        else:
            api_key_name = "xai_api_key"
            base_url = "https://api.x.ai/v1"
            model_name = "grok-beta"
        api_key = st.secrets.get(api_key_name)
        if not api_key: api_key = st.text_input(f"Enter {api_key_name}", type="password")

    # --- SEARCH (NEW FEATURE) ---
    st.divider()
    st.header("1. Search & Select")
    
    if not df.empty:
        # GLOBAL COLUMN SEARCH
        with st.expander("🔍 Find Datasets by Column Name", expanded=True):
            col_search = st.text_input("Enter column (e.g. OrgUnitId)", placeholder="Type field name...")
            if col_search:
                # Case-insensitive search
                matches = df[df['column_name'].astype(str).str.contains(col_search, case=False)]
                if not matches.empty:
                    found_datasets = sorted(matches['dataset_name'].unique())
                    st.success(f"Found in **{len(found_datasets)}** datasets:")
                    st.dataframe(found_datasets, hide_index=True, use_container_width=True)
                else:
                    st.warning("No matching columns found.")

        # DATASET SELECTOR
        st.subheader("Select Datasets")
        select_mode = st.radio("Method:", ["Category (Grouped)", "List All"], horizontal=True, label_visibility="collapsed")

        selected_datasets = []
        
        if select_mode == "Category (Grouped)":
            all_cats = sorted(df['category'].unique())
            selected_cats = st.multiselect("Filter Categories:", all_cats, default=[])
            if selected_cats:
                for cat in selected_cats:
                    cat_ds = sorted(df[df['category'] == cat]['dataset_name'].unique())
                    s = st.multiselect(f"📦 {cat}", cat_ds, key=f"sel_{cat}")
                    selected_datasets.extend(s)
        else: 
            all_ds = sorted(df['dataset_name'].unique())
            selected_datasets = st.multiselect("Search All:", all_ds, key="global_search")

    # --- Scraper Section ---
    st.divider()
    with st.expander("⚠️ Update Data (Scraper)", expanded=False):
        pasted_text = st.text_area("URLs", height=100, value=DEFAULT_URLS)
        if st.button("Scrape URLs", type="primary"):
            url_list = parse_urls_from_text_area(pasted_text)
            with st.spinner(f"Scraping {len(url_list)} pages..."):
                df_new = scrape_and_save_from_list(url_list)
                stats_msg = f"**Scrape Success:** {df_new['dataset_name'].nunique()} Datasets / {len(df_new):,} Columns"
                st.session_state['scrape_msg'] = stats_msg
                st.rerun()

# ========================= MAIN PAGE CONTENT =========================
if df.empty:
    st.warning("Please use the sidebar to scrape data first.")
    st.stop()

if selected_datasets:
    col_title, col_clear = st.columns([5, 1])
    with col_title:
        st.title(f"Analyzing {len(selected_datasets)} Dataset(s)")
    with col_clear:
        if st.button("Clear All", type="primary"):
            for key in st.session_state.keys():
                if key.startswith("sel_") or key == "global_search":
                    st.session_state[key] = []
            st.rerun()
else:
    st.title("Dataset & Relationship Explorer")
    st.info("👈 Use the **'Find Datasets by Column'** tool or select datasets to begin.")
    st.stop()

# 3. Data Preview
with st.expander("📋 View Schema Details", expanded=False):
    subset = df[df['dataset_name'].isin(selected_datasets)]
    st.dataframe(subset[['dataset_name', 'column_name', 'data_type', 'description', 'key']], use_container_width=True, hide_index=True)

# 4. Graph Visualization
col_header, col_controls = st.columns([1, 1])
with col_header:
    st.subheader("Connection Graph")
with col_controls:
    graph_mode = st.radio("Graph Mode", options=['Between selected (Focused)', 'From selected (Discovery)'], horizontal=True)

join_data = find_pk_fk_joins(df, selected_datasets)
G = nx.DiGraph()

if graph_mode == 'Between selected (Focused)':
    for ds in selected_datasets: G.add_node(ds, type='focus')
    if not join_data.empty:
        for _, row in join_data.iterrows():
            s, t = row['Source Dataset'], row['Target Dataset']
            if s in selected_datasets and t in selected_datasets:
                G.add_edge(s, t, label=row['Join Column'])
else:
    for ds in selected_datasets: G.add_node(ds, type='focus')
    if not join_data.empty:
        for _, row in join_data.iterrows():
            s, t = row['Source Dataset'], row['Target Dataset']
            if s in selected_datasets:
                if not G.has_node(t): G.add_node(t, type='neighbor')
                G.add_edge(s, t, label=row['Join Column'])

if G.number_of_nodes() > 0:
    pos = nx.spring_layout(G, k=0.6, iterations=50)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#666'), hoverinfo='none', mode='lines')
    
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        n_type = G.nodes[node].get('type', 'focus')
        node_color.append('#FF4B4B' if n_type == 'focus' else '#1F77B4')
        node_size.append(25 if n_type == 'focus' else 15)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition="top center",
        marker=dict(color=node_color, size=node_size, line=dict(width=2, color='white'))
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=550, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    ))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No relationships found.")

# 5. Auto-SQL
st.divider()
st.subheader("⚡ Instant SQL Builder")
col_sql1, col_sql2 = st.columns([2,1])
with col_sql1:
    st.caption("Programmatic JOIN generation based on Metadata.")

if len(selected_datasets) < 2:
    st.warning("Select 2+ datasets to generate SQL.")
elif G.number_of_edges() == 0 and graph_mode == 'Between selected (Focused)':
    st.warning("No direct PK/FK relationships found.")
else:
    base_table = selected_datasets[0]
    aliases = {name: f"t{i+1}" for i, name in enumerate(selected_datasets)}
    sql_lines = [f"SELECT TOP 100", f"    {aliases[base_table]}.*"]
    sql_lines.append(f"FROM {base_table} {aliases[base_table]}")
    joined_tables = {base_table}
    relevant_edges = [e for e in G.edges(data=True) if e[0] in selected_datasets and e[1] in selected_datasets]
    
    for u, v, data in relevant_edges:
        col = data.get('label')
        if v not in joined_tables and u in joined_tables:
            sql_lines.append(f"LEFT JOIN {v} {aliases[v]} ON {aliases[u]}.{col} = {aliases[v]}.{col}")
            joined_tables.add(v)
        elif u not in joined_tables and v in joined_tables:
            sql_lines.append(f"LEFT JOIN {u} {aliases[u]} ON {aliases[v]}.{col} = {aliases[u]}.{col}")
            joined_tables.add(u)
            
    st.code("\n".join(sql_lines), language="sql")

# 6. AI Chat
st.divider()
st.subheader(f"Ask {ai_provider.split(' ')[0]} about your data")

if "messages" not in st.session_state: st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if prompt := st.chat_input("e.g., Explain these columns..."):
    if not api_key:
        st.error("Please enter API Key in sidebar.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                context_df = df[df['dataset_name'].isin(selected_datasets)] if selected_datasets else df.head(50)
                
                # UPDATED SYSTEM PROMPT TO PREVENT HALLUCINATION
                system_msg = f"""
                You are an expert SQL Data Architect for Brightspace (D2L).
                
                IMPORTANT:
                You have been provided schema context ONLY for the {len(selected_datasets)} datasets the user has explicitly selected.
                There are ~140 total datasets in the system, but you cannot see them right now.
                
                DO NOT make absolute statements like "The total number of datasets with column X is Y" based on this subset.
                Always clarify: "In the selected datasets, X appears in..."
                
                Selected Schema Context:
                {context_df.to_csv(index=False)[:12000]}
                """
                
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Error: {e}")
