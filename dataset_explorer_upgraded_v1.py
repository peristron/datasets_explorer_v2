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

# Logging Setup
logging.basicConfig(filename='scraper.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# ========================= PASSWORD PROTECTION =========================
def check_password():
    """Simple password protection."""
    pwd = st.secrets.get("app_password")
    if not pwd: return True # Allow if no password set in secrets
    
    if st.session_state.get("authenticated", False): return True
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("Login")
        user_input = st.text_input("Password", type="password", key="pw_input")
        if st.button("Submit", type="primary"):
            if user_input == pwd:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
    st.stop()

check_password()

# ========================= DEFAULT URLs (FULL LIST) =========================
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

# ========================= SCRAPING FUNCTIONS =========================
def parse_urls_from_text_area(text_block):
    urls = [line.strip() for line in text_block.split('\n') if line.strip()]
    valid_urls = [url for url in urls if url.startswith('http')]
    return sorted(list(set(valid_urls)))

def scrape_table(url, category_name):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
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
                if len(text) > 3: # Filter out empty/tiny headers
                    current_dataset = text.lower()
            elif element.name == 'table':
                table_headers = [th.text.strip().lower().replace(' ', '_') for th in element.find_all('th')]
                if not table_headers: continue
                
                # Heuristic: Only scrape tables that look like data definitions
                if not any(x in table_headers for x in ['type', 'description', 'data_type']):
                    continue

                for row in element.find_all('tr'):
                    columns_ = row.find_all('td')
                    if len(columns_) < len(table_headers): continue # Skip malformed rows
                    
                    entry = {}
                    for i, header in enumerate(table_headers):
                        if i < len(columns_):
                            entry[header] = columns_[i].text.strip()
                            
                    # Standardize keys
                    header_map = {'field': 'column_name', 'name': 'column_name', 'type': 'data_type'}
                    entry = {header_map.get(k, k): v for k, v in entry.items()}
                    
                    if 'column_name' not in entry or not entry['column_name']: continue
                    
                    entry['dataset_name'] = current_dataset
                    entry['category'] = category_name
                    data.append(entry)
        return data
    except Exception as e:
        logging.error(f"Error scraping page {url}: {e}")
        return []

def scrape_and_save_from_list(url_list):
    all_data = []
    progress_bar = st.progress(0, "Scraping dataset pages...")
    
    def get_category_from_url(url):
        return re.sub(r'^\d+\s*', '', os.path.basename(url).split('?')[0].replace('-data-sets', '').replace('-', ' ')).lower()

    with ThreadPoolExecutor(max_workers=10) as executor:
        args = [(url, get_category_from_url(url)) for url in url_list]
        future_to_url = {executor.submit(scrape_table, *arg): arg[0] for arg in args}
        for i, future in enumerate(future_to_url):
            try:
                result = future.result()
                all_data.extend(result)
            except Exception:
                pass
            progress_bar.progress((i + 1) / len(url_list), f"Scraping... {i+1}/{len(url_list)}")
    
    progress_bar.empty()
    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    
    # Ensure columns exist
    expected_cols = ['category', 'dataset_name', 'column_name', 'data_type', 'description', 'key']
    for col in expected_cols:
        if col not in df.columns: df[col] = ''
        
    df = df.fillna('')
    # Normalize dataset names
    df['dataset_name'] = df['dataset_name'].astype(str).str.title()
    df['category'] = df['category'].astype(str).str.title()
    
    df['is_primary_key'] = df['key'].astype(str).str.contains(r'\bpk\b', case=False, regex=True)
    df['is_foreign_key'] = df['key'].astype(str).str.contains(r'\bfk\b', case=False, regex=True)
    
    df.to_csv('dataset_metadata.csv', index=False)
    return df

@st.cache_data
def find_pk_fk_joins(df, selected_datasets):
    if df.empty or not selected_datasets: return pd.DataFrame()
    
    # Exact match logic
    pks = df[df['is_primary_key'] == True]
    fks = df[(df['is_foreign_key'] == True) & (df['dataset_name'].isin(selected_datasets))]
    
    if pks.empty or fks.empty: return pd.DataFrame()
    
    # Join on column name
    merged = pd.merge(fks, pks, on='column_name', suffixes=('_fk', '_pk'))
    joins = merged[merged['dataset_name_fk'] != merged['dataset_name_pk']]
    
    if joins.empty: return pd.DataFrame()
    
    result = joins[['dataset_name_fk', 'column_name', 'dataset_name_pk', 'category_pk']]
    result.columns = ['Source Dataset', 'Join Column', 'Target Dataset', 'Target Category']
    return result.drop_duplicates().reset_index(drop=True)

# ========================= SIDEBAR =========================
with st.sidebar:
    st.title("Brightspace Explorer")
    
    # --- AI Section ---
    with st.expander("🤖 AI Configuration", expanded=True):
        ai_provider = st.radio("AI Provider", ["OpenAI (GPT-4o)", "xAI (Grok)"])
        
        # Set Keys and Base URLs based on selection
        if "OpenAI" in ai_provider:
            api_key_name = "openai_api_key"
            base_url = None # Default for OpenAI
            model_name = "gpt-4o"
        else:
            api_key_name = "xai_api_key"
            base_url = "https://api.x.ai/v1"
            model_name = "grok-beta"
            
        api_key = st.secrets.get(api_key_name)
        if not api_key:
            api_key = st.text_input(f"Enter {api_key_name}", type="password")

    # --- Scraper Section ---
    with st.expander("⚠️ Update Data Source", expanded=False):
        st.info("Add specific KB article URLs below.")
        pasted_text = st.text_area("URLs", height=150, value=DEFAULT_URLS)
        if st.button("Scrape All URLs", type="primary"):
            url_list = parse_urls_from_text_area(pasted_text)
            with st.spinner(f"Scraping {len(url_list)} pages..."):
                scrape_and_save_from_list(url_list)
                st.success("Done! Reloading...")
                st.rerun()

# ========================= MAIN UI =========================

# 1. Load Data
if os.path.exists('dataset_metadata.csv'):
    df = pd.read_csv('dataset_metadata.csv').fillna('')
else:
    st.warning("No data found. Please click 'Scrape All URLs' in the sidebar.")
    st.stop()

# 2. Filtering
st.sidebar.subheader("Dataset Selection")
categories = sorted(df['category'].unique())
selected_categories = st.sidebar.multiselect("Filter Category", categories)

available_datasets = df[df['category'].isin(selected_categories)]['dataset_name'].unique() if selected_categories else df['dataset_name'].unique()
selected_datasets = st.sidebar.multiselect("Select Datasets", sorted(available_datasets))

st.title("Dataset & Relationship Explorer")

# 3. Display Data Details
if selected_datasets:
    with st.expander("Dataset Schema Details", expanded=False):
        subset = df[df['dataset_name'].isin(selected_datasets)]
        # Prioritize columns
        cols = [c for c in ['dataset_name', 'column_name', 'data_type', 'description', 'key'] if c in subset.columns]
        st.dataframe(subset[cols], use_container_width=True, hide_index=True)
else:
    st.info("Select datasets from the sidebar to begin.")

# 4. Graph Visualization
if selected_datasets:
    st.subheader("Connection Graph")
    
    graph_mode = st.radio(
        "Graph Mode:",
        ('Between selected (Focused)', 'From selected (Discovery)'),
        horizontal=True
    )
    
    # Graph Styling Controls
    col1, col2 = st.columns(2)
    with col1:
        node_separation = st.slider("Spread", 0.1, 2.0, 0.5)
    with col2:
        show_labels = st.checkbox("Show Edge Labels (Slower)", False)

    join_data = find_pk_fk_joins(df, selected_datasets)
    G = nx.DiGraph()
    
    # Logic to build nodes/edges
    if graph_mode == 'Between selected (Focused)':
        for ds in selected_datasets:
            G.add_node(ds, type='focus')
        if not join_data.empty:
            for _, row in join_data.iterrows():
                s, t = row['Source Dataset'], row['Target Dataset']
                if s in selected_datasets and t in selected_datasets:
                    G.add_edge(s, t, label=row['Join Column'])
    else:
        for ds in selected_datasets:
            G.add_node(ds, type='focus')
        if not join_data.empty:
            for _, row in join_data.iterrows():
                s, t = row['Source Dataset'], row['Target Dataset']
                if s in selected_datasets:
                    if not G.has_node(t): G.add_node(t, type='neighbor')
                    G.add_edge(s, t, label=row['Join Column'])

    # Render
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, k=node_separation, iterations=50)
        
        # 1. Edges (Optimized: Single Trace)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, 
            line=dict(width=1, color='#666'), 
            hoverinfo='none', 
            mode='lines'
        )
        
        traces = [edge_trace]

        # 2. Edge Labels (Optional - slower)
        if show_labels:
            label_x, label_y, label_text = [], [], []
            for u, v, data in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                label_x.append((x0+x1)/2)
                label_y.append((y0+y1)/2)
                label_text.append(data.get('label',''))
            
            label_trace = go.Scatter(
                x=label_x, y=label_y, mode='text', text=label_text,
                textfont=dict(color='cyan', size=10), hoverinfo='none'
            )
            traces.append(label_trace)

        # 3. Nodes
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            n_type = G.nodes[node].get('type', 'focus')
            node_color.append('#FF4B4B' if n_type == 'focus' else '#1F77B4')
            node_size.append(30 if n_type == 'focus' else 15)

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            text=node_text, textposition="top center",
            marker=dict(color=node_color, size=node_size, line=dict(width=2, color='white'))
        )
        traces.append(node_trace)

        fig = go.Figure(data=traces, layout=go.Layout(
            showlegend=False, hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No relationships found.")

# ========================= AI CHAT INTERFACE =========================
st.divider()
st.subheader(f"Ask {ai_provider.split(' ')[0]} about your data")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How do I join users and grades?"):
    if not api_key:
        st.error(f"Please enter your {ai_provider} API Key in the sidebar.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare Context
                context_df = df[df['dataset_name'].isin(selected_datasets)] if selected_datasets else df.head(50)
                csv_context = context_df.to_csv(index=False)
                
                system_msg = f"""
                You are an expert SQL Data Architect for Brightspace (D2L).
                
                CONTEXT:
                The user is asking about the following datasets: {', '.join(selected_datasets) if selected_datasets else 'general datasets'}.
                Here is the schema definition (CSV format):
                {csv_context[:12000]}
                
                INSTRUCTIONS:
                1. Suggest SQL joins based on the 'key' columns (PK/FK).
                2. Explain what the columns mean based on 'description'.
                3. If writing SQL, use standard SQL syntax compatible with Brightspace Data Sets (BDS).
                """

                client = openai.OpenAI(api_key=api_key, base_url=base_url)

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                reply = response.choices[0].message.content
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
            except Exception as e:
                st.error(f"API Error: {e}")
