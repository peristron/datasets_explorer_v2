# streamlit run dataset_explorer_upgraded_v1.py
#  directory setup: cd C:\users\oakhtar\documents\pyprojs_local



# python_datahub_dataset_relationships_v200_AI_EDITION.py
# THE ULTIMATE BRIGHTSPACE DATASET EXPLORER — AI POWERED
# Local test: streamlit run python_datahub_dataset_relationships_v200_AI_EDITION.py
# dataset_explorer_upgraded_v1.py
# Brightspace Dataset Explorer v200 AI Edition — FINAL PRODUCTION VERSION
# Fully working on Streamlit Cloud (Dec 2025)

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

# ========================= CONFIG =========================
logging.basicConfig(level=logging.INFO)
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

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

# ========================= HELPERS =========================
def parse_urls_from_text_area(text_block):
    urls = [line.strip() for line in text_block.split('\n') if line.strip().startswith('http')]
    return sorted(list(set(urls)))

def scrape_table(url, category_name):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=15, verify=False)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, 'html.parser')
        data = []
        current_dataset = category_name
        for el in soup.find_all(['h2', 'h3', 'table']):
            if el.name in ['h2', 'h3']:
                current_dataset = el.text.strip().lower()
            elif el.name == 'table':
                headers = [th.text.strip().lower().replace(' ', '_') for th in el.find_all('th')]
                if not headers: continue
                for row in el.find_all('tr'):
                    cols = row.find_all('td')
                    if len(cols) != len(headers): continue
                    entry = {headers[i]: cols[i].text.strip() for i in range(len(headers))}
                    entry = {('column_name' if k in ['field','name'] else 'data_type' if k=='type' else k): v for k,v in entry.items()}
                    if 'column_name' in entry and entry['column_name']:
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
        futures = []
        for url in url_list:
            cat = re.sub(r'^\d+\s*', '', os.path.basename(url).replace('-data-sets','').replace('-',' ')).lower()
            futures.append(executor.submit(scrape_table, url, cat))
        for i, future in enumerate(futures):
            all_data.extend(future.result())
            progress.progress((i + 1) / len(futures))
    progress.empty()
    if not all_data:
        st.error("No data scraped. Check URLs.")
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df.columns = [c.lower().replace(' ','_') for c in df.columns]
    for col in ['category','dataset_name','column_name','data_type','description','key','version','version_history','column_size','notes']:
        if col not in df.columns: df[col] = ''
    df = df.fillna('')
    df['is_primary_key'] = df['key'].str.contains('pk', case=False, na=False)
    df['is_foreign_key'] = df['key'].str.contains('fk', case=False, na=False)
    df.to_csv('dataset_metadata.csv', index=False)
    st.success(f"Scraped {len(df)} rows → dataset_metadata.csv")
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

def get_api_key(provider_name):
    if provider_name == "OpenAI":
        key = st.secrets.get("openai_api_key")
        return key, "https://api.openai.com/v1", "gpt-4o" if key else None
    else:  # xAI (Grok)
        key = st.secrets.get("xai_api_key")
        return key, "https://api.x.ai/v1", "grok-beta" if key else None

def call_llm(messages, api_key, base_url, model):
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content

# ========================= PASSWORD PROTECTION =========================
def check_password():
    """Perfect password protection — no false error on load."""
    password = st.secrets.get("app_password")
    
    # No password set → open access
    if not password:
        return True
    
    # Already logged in this session
    if st.session_state.get("password_correct", False):
        return True
    
    # Show password input
    st.text_input(
        "Password",
        type="password",
        key="password_input",
        placeholder="Enter password",
        label_visibility="collapsed"
    )
    
    # Only evaluate AFTER user has typed something
    if "password_input" in st.session_state:
        if st.session_state["password_input"] == password:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
            # Do NOT st.stop() here — let them try again
    else:
        st.info("This app is password protected")
        st.stop()  # First load — wait for input
    
    return True

# ========================= MAIN APP =========================
def main():
    st.set_page_config(page_title="Brightspace Dataset Explorer v200 AI", layout="wide", page_icon="🧠")
    st.title("Brightspace Dataset Explorer v200 — AI Edition")

    # ====================== SIDEBAR ======================
    with st.sidebar:
        st.header("AI Provider")
        provider = st.radio("Model", ["OpenAI (gpt-4o)", "xAI (Grok)"], key="ai_provider")
        api_key, base_url, model = get_api_key(provider.split()[0])
        if not api_key:
            st.error("API key missing in secrets.toml")
            st.stop()

        st.divider()
        st.header("Data Controls")
        with st.expander("STEP 1: Load/Update Data", expanded=False):
            urls_text = st.text_area("URLs (one per line)", height=200, value=DEFAULT_URLS, key="urls_input")
            if st.button("Scrape & Update", type="primary", key="scrape_btn"):
                urls = parse_urls_from_text_area(urls_text)
                with st.spinner(f"Scraping {len(urls)} pages..."):
                    scrape_and_save_from_list(urls)
                    st.rerun()

    # ====================== LOAD DATA ======================
    if not os.path.exists("dataset_metadata.csv"):
        st.warning("No dataset_metadata.csv found. Use 'Scrape & Update' in sidebar.")
        st.stop()

    columns = pd.read_csv("dataset_metadata.csv").fillna("")
    datasets = sorted(columns['dataset_name'].unique())
    categories = sorted(columns['category'].unique())

    # ====================== GLOBAL SEARCH ======================
    st.subheader("Global Search")
    search = st.text_input("Search columns, descriptions, keys...", key="global_search")
    if search:
        mask = columns.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
        results = columns[mask][['dataset_name', 'column_name', 'description', 'key']].head(100)
        st.dataframe(results, use_container_width=True)

    # ====================== AI NATURAL LANGUAGE ======================
    st.subheader("Ask AI: Find Datasets")
    question = st.text_input("e.g. late quiz submissions with grades", key="ai_question")
    if question and st.button("Search with AI", key="ai_search_btn"):
        with st.spinner("Thinking..."):
            metadata_snippet = columns[['dataset_name','column_name','description']].drop_duplicates().to_csv(index=False)
            try:
                resp = call_llm([{"role": "user", "content": f"Return ONLY a JSON list of dataset names relevant to: {question}\nSchema:\n{metadata_snippet}"}], api_key, base_url, model)
                suggested = json.loads(resp).get("datasets", [])
                if suggested:
                    st.success(f"Found: {', '.join(suggested[:5])}")
                    st.session_state.selected_from_ai = suggested
            except:
                st.error("AI failed to parse response")

    # ====================== DATASET SELECTION ======================
    selected_categories = st.multiselect("Filter by Category", categories, key="cat_filter")
    filtered_datasets = columns[columns['category'].isin(selected_categories)]['dataset_name'].unique() if selected_categories else datasets
    default_selection = st.session_state.get("selected_from_ai", [])
    selected_datasets = st.multiselect("Select Datasets", filtered_datasets, default=default_selection, key="dataset_select")

    # ====================== GRAPH ======================
    if selected_datasets:
        st.subheader("Relationship Graph")
        mode = st.radio("Mode", ["Focused (between selected)", "Discovery (outgoing)"], horizontal=True, key="graph_mode")

        join_data = find_pk_fk_joins(columns, selected_datasets)
        G = nx.DiGraph()
        for ds in selected_datasets:
            G.add_node(ds, type='focus')

        if mode.startswith("Focused"):
            for _, row in join_data.iterrows():
                if row['Source Dataset'] in selected_datasets and row['Target Dataset'] in selected_datasets:
                    G.add_edge(row['Source Dataset'], row['Target Dataset'],
                               label=row['Join Column'],
                               sql=f"INNER JOIN {row['Target Dataset']} ON {row['Source Dataset']}.{row['Join Column']} = {row['Target Dataset']}.{row['Join Column']}")
        else:
            for _, row in join_data.iterrows():
                if row['Source Dataset'] in selected_datasets:
                    G.add_node(row['Target Dataset'], type='neighbor')
                    G.add_edge(row['Source Dataset'], row['Target Dataset'],
                               label=row['Join Column'],
                               sql=f"INNER JOIN {row['Target Dataset']} ON {row['Source Dataset']}.{row['Join Column']} = {row['Target Dataset']}.{row['Join Column']}")

        if G.nodes():
            pos = nx.spring_layout(G, k=1.2, iterations=60)
            edge_traces = []
            for u, v, d in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                              mode='lines', line=dict(color='#888', width=2),
                                              hoverinfo='text', hovertext=d.get('sql', ''),
                                              customdata=[d.get('sql', '')]))

            node_trace = go.Scatter(x=[], y=[], text=[], mode='markers+text',
                                    marker=dict(size=40, color=[], line=dict(width=3)),
                                    textfont=dict(size=14, color='white'))

            colors = {c: f"hsl({(hash(c)*137)%360},70%,60%)" for c in categories}
            for node in G.nodes():
                x, y = pos[node]
                node_trace['x'] += (x,)
                node_trace['y'] += (y,)
                cat = columns[columns['dataset_name']==node]['category'].iloc[0] if not columns[columns['dataset_name']==node].empty else ''
                node_trace['marker']['color'] += (colors.get(cat, '#ccc'),)
                node_trace['text'] += (f"<b>{node}</b>",)

            fig = go.Figure(data=edge_traces + [node_trace],
                            layout=go.Layout(paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e',
                                             xaxis=dict(showgrid=False, zeroline=False),
                                             yaxis=dict(showgrid=False, zeroline=False),
                                             height=700, hovermode='closest'))

            clicked = plotly_events(fig, click_event=True)
            if clicked and clicked[0].get('customdata'):
                sql = clicked[0]['customdata'][0]
                st.code(sql, language='sql')
                st.success("SQL copied to clipboard (in browser)")

            st.plotly_chart(fig, use_container_width=True)

    # ====================== SQL GENERATOR & CHAT ======================
    if selected_datasets:
        st.subheader("Generate SQL")
        goal = st.text_input("What do you want to find?", key="sql_goal")
        if goal and st.button("Generate SQL", key="gen_sql"):
            with st.spinner("Writing query..."):
                schema = columns[columns['dataset_name'].isin(selected_datasets)][['dataset_name','column_name','data_type','description']]
                prompt = f"Write a complete SQL query using these tables: {', '.join(selected_datasets)}\nGoal: {goal}\nSchema:\n{schema.to_csv(index=False)}\nReturn only SQL."
                sql = call_llm([{"role": "user", "content": prompt}], api_key, base_url, model)
                st.code(sql, language='sql')

        st.subheader("Chat with Brightspace Data")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
        if prompt := st.chat_input("Ask anything..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = call_llm([{"role": "user", "content": f"Brightspace schema sample:\n{columns[['dataset_name','column_name']].head(50).to_csv(index=False)}\n\nQuestion: {prompt}"}], api_key, base_url, model)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    if check_password():
        main()


