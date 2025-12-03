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

# ========================= CONFIG & PASSWORD =========================
st.set_page_config(page_title="Brightspace Explorer", layout="wide")

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

st.title("Brightspace Dataset Explorer & AI Assistant")

# ========================= HELPER FUNCTIONS =========================
def standardize_columns(df):
    """Cleans up scraped column headers."""
    if df.empty: return df
    rename_map = {
        "field": "column_name", "name": "column_name", "column": "column_name",
        "type": "data_type", "data_type": "data_type",
        "description": "description", "desc": "description", "details": "description",
        "key": "key"
    }
    # Normalize headers to lower case for matching
    df.columns = [c.strip() for c in df.columns]
    actual_rename = {k: v for k, v in rename_map.items() if k in [c.lower() for c in df.columns]}
    
    # Apply renaming based on case-insensitive match
    new_cols = {}
    for col in df.columns:
        if col.lower() in rename_map:
            new_cols[col] = rename_map[col.lower()]
    
    df = df.rename(columns=new_cols)
    
    # Ensure essential columns exist
    required = ["column_name", "data_type", "description", "key"]
    for r in required:
        if r not in df.columns:
            df[r] = ""
            
    df["column_name"] = df["column_name"].astype(str).str.strip()
    return df.fillna("")

DEFAULT_URLS = [
    "https://community.d2l.com/brightspace/kb/articles/4740-users-data-sets",
    "https://community.d2l.com/brightspace/kb/articles/4527-grades-data-sets",
    "https://community.d2l.com/brightspace/kb/articles/4713-content-data-sets",
    "https://community.d2l.com/brightspace/kb/articles/4519-attendance-data-sets",
    "https://community.d2l.com/brightspace/kb/articles/4520-awards-data-sets",
    "https://community.d2l.com/brightspace/kb/articles/4767-assignments-data-sets"
]

# ========================= SCRAPER =========================
def scrape_single_url(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code != 200: return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract clean category name from URL
        category = os.path.basename(url).split('?')[0]
        category = re.sub(r'^\d+[-_]', '', category) # Remove leading numbers
        category = category.replace('-', ' ').replace('data sets', '').title().strip()
        
        data = []
        current_dataset = category # Default fallback
        
        # Find all headers and tables in order
        elements = soup.find_all(['h2', 'h3', 'h4', 'table'])
        
        for elm in elements:
            if elm.name in ['h2', 'h3', 'h4']:
                text = elm.get_text(strip=True)
                # Heuristic: Ignore generic headers
                if len(text) > 3 and "About" not in text and "History" not in text:
                    current_dataset = text
            
            elif elm.name == 'table':
                # Basic table parsing
                rows = elm.find_all('tr')
                if not rows: continue
                
                # Find header row
                headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]
                
                # Skip tables that don't look like data definitions (must have Type or Description)
                header_str = " ".join(headers).lower()
                if "type" not in header_str and "description" not in header_str:
                    continue

                for row in rows[1:]:
                    cols = [td.get_text(strip=True) for td in row.find_all('td')]
                    # Handle colspan or missing cells roughly
                    if len(cols) == len(headers):
                        entry = dict(zip(headers, cols))
                        entry['dataset_name'] = current_dataset
                        entry['category'] = category
                        data.append(entry)
        return data
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

def run_scraper(urls):
    all_data = []
    with st.spinner(f"Scraping {len(urls)} pages..."):
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(scrape_single_url, urls)
            for res in results:
                all_data.extend(res)
    
    if not all_data:
        st.error("No data found.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data)
    df = standardize_columns(df)
    
    # Identify Keys
    df['is_primary_key'] = df['key'].astype(str).str.contains(r'\bpk\b', case=False, regex=True)
    df['is_foreign_key'] = df['key'].astype(str).str.contains(r'\bfk\b', case=False, regex=True)
    
    df.to_csv("dataset_metadata.csv", index=False)
    return df

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Configuration")
    provider = st.selectbox("AI Model", ["OpenAI (gpt-4o)", "xAI (Grok)"])
    
    api_key_name = "openai_api_key" if "OpenAI" in provider else "xai_api_key"
    api_key = st.secrets.get(api_key_name)
    
    if not api_key:
        api_key = st.text_input(f"Enter {api_key_name}", type="password")
    
    with st.expander("Manage Data Source"):
        urls_input = st.text_area("URLs", "\n".join(DEFAULT_URLS), height=150)
        if st.button("Scrape & Update"):
            url_list = [u.strip() for u in urls_input.split('\n') if u.strip()]
            run_scraper(url_list)
            st.rerun()

# ========================= MAIN UI =========================

# 1. Load Data
if os.path.exists("dataset_metadata.csv"):
    df = pd.read_csv("dataset_metadata.csv")
else:
    st.info("Dataset metadata not found. Please click 'Scrape & Update' in the sidebar.")
    st.stop()

# 2. Filtering
categories = sorted(df['category'].astype(str).unique())
selected_cat = st.multiselect("Filter Category", categories)

if selected_cat:
    filtered_df = df[df['category'].isin(selected_cat)]
else:
    filtered_df = df

datasets = sorted(filtered_df['dataset_name'].astype(str).unique())
selected_datasets = st.multiselect("Select Datasets to Visualize", datasets)

# 3. Graph Visualization (Optimized)
if selected_datasets:
    st.subheader("Relationship Graph")
    
    # Build Graph
    G = nx.DiGraph()
    subset = df[df['dataset_name'].isin(selected_datasets)]
    
    # Add Nodes
    for ds in selected_datasets:
        G.add_node(ds)
        
    # Add Edges (Logic: FK matches PK name)
    # Optimized lookups
    pk_map = df[df['is_primary_key']].groupby('column_name')['dataset_name'].apply(list).to_dict()
    
    fk_rows = subset[subset['is_foreign_key']]
    
    relationships = []
    
    for _, row in fk_rows.iterrows():
        col = row['column_name']
        origin = row['dataset_name']
        
        if col in pk_map:
            targets = pk_map[col]
            for target in targets:
                # Only draw if target is selected OR we want to show external links
                if target in selected_datasets and target != origin:
                    G.add_edge(origin, target, col=col)
                    relationships.append(f"{origin}.{col} -> {target}.{col}")

    if G.number_of_nodes() > 0:
        pos = nx.shell_layout(G) if len(G.nodes) < 10 else nx.spring_layout(G, k=0.5)
        
        # Optimized Edge Trace (Single trace for all lines)
        edge_x = []
        edge_y = []
        edge_text = []
        
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            # Note: Hover text on lines in single trace is tricky in Plotly, usually applied to nodes or middle points
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = []
        node_adjacencies = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_adjacencies.append(len(G[node]))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                color=node_adjacencies,
                size=20,
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600
                     ))
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No direct relationships found between selected datasets.")

    # 4. Data Preview
    with st.expander("View Schema Details"):
        st.dataframe(subset[['dataset_name', 'column_name', 'data_type', 'key', 'description']], use_container_width=True)

# ========================= AI CHAT INTERFACE =========================
st.divider()
st.subheader("AI Data Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Logic
if prompt := st.chat_input("Ask about SQL joins or dataset details..."):
    if not api_key:
        st.error("Please enter an API Key in the sidebar.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Context preparation
                context_df = df[df['dataset_name'].isin(selected_datasets)] if selected_datasets else df.head(50)
                csv_context = context_df.to_csv(index=False)
                
                system_prompt = f"""
                You are an expert on Brightspace (D2L) datasets.
                Here is the schema context for the user's selected datasets:
                {csv_context[:10000]} 
                (Context truncated to first 10k chars)
                
                Answer the user's question regarding SQL queries, relationships, or column definitions.
                """
                
                if "OpenAI" in provider:
                    client = openai.OpenAI(api_key=api_key)
                    model = "gpt-4o"
                else:
                    # xAI configuration (assuming generic OpenAI compatible endpoint)
                    client = openai.OpenAI(
                        api_key=api_key, 
                        base_url="https://api.x.ai/v1"
                    )
                    model = "grok-beta"

                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                ai_reply = response.choices[0].message.content
                st.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            except Exception as e:
                st.error(f"AI Error: {str(e)}")
