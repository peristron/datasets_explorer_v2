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



# ========================= INITIALIZE SESSION STATE =========================

if 'total_cost' not in st.session_state: st.session_state['total_cost'] = 0.0

if 'total_tokens' not in st.session_state: st.session_state['total_tokens'] = 0



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



# ========================= HELPER: CLEAR CALLBACK =========================

def clear_all_selections():

    for key in list(st.session_state.keys()):

        if key.startswith("sel_") or key == "global_search":

            st.session_state[key] = []



# ========================= SCRAPE SUCCESS MESSAGE =========================

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


                        # === ADDED: Save URL to data ===


                        entry['url'] = url 

                        data.append(entry)

        return data

    except Exception: return []

@@ -152,7 +154,7 @@

    if not all_data: return pd.DataFrame()



    df = pd.DataFrame(all_data)


    expected_cols = ['category', 'dataset_name', 'column_name', 'data_type', 'description', 'key']


    expected_cols = ['category', 'dataset_name', 'column_name', 'data_type', 'description', 'key', 'url']

    for col in expected_cols:

        if col not in df.columns: df[col] = ''

    df = df.fillna('')

@@ -189,7 +191,7 @@

    with st.expander("❓ How to use this app", expanded=False):

        st.markdown("""

        **1. Load Data:** 


        Use 'Update Data' below to scrape the KB.


        Use 'Update Data' below to scrape the KB. **(Do this first to get URLs!)**

        

        **2. Find Datasets:**

        Search by column name or browse by Category.

@@ -223,9 +225,12 @@

    # --- COST ESTIMATOR ---

    with st.expander("💰 Cost Estimator", expanded=False):

        st.caption(f"Current Session ({model_name})")


        col_c1, col_c2 = st.columns(2)


        col_c1.metric("Tokens Used", f"{st.session_state['total_tokens']:,}")


        col_c2.metric("Est. Cost", f"${st.session_state['total_cost']:.4f}")


        


        # Using standard markdown/text to ensure precision isn't truncated


        c1, c2 = st.columns(2)


        c1.markdown(f"**Tokens:**\n{st.session_state['total_tokens']:,}")


        c2.markdown(f"**Cost:**\n`${st.session_state['total_cost']:.5f}`")


        

        if st.button("Reset Cost"):

            st.session_state['total_cost'] = 0.0

            st.session_state['total_tokens'] = 0

@@ -304,7 +309,10 @@

# 3. Data Preview

with st.expander("📋 View Schema Details", expanded=False):

    subset = df[df['dataset_name'].isin(selected_datasets)]


    st.dataframe(subset[['dataset_name', 'column_name', 'data_type', 'description', 'key']], use_container_width=True, hide_index=True)


    # Added 'url' to preview to confirm it exists


    cols_to_show = ['dataset_name', 'column_name', 'data_type', 'description', 'key']


    if 'url' in subset.columns: cols_to_show.append('url')


    st.dataframe(subset[cols_to_show], use_container_width=True, hide_index=True)



# 4. Graph Visualization

col_header, col_controls = st.columns([1, 1])

@@ -412,7 +420,6 @@



col_chat_opt, col_chat_msg = st.columns([1, 3])

with col_chat_opt:


    # === NEW: COST OPTIMIZATION TOGGLE ===

    use_full_context = st.checkbox("Include ALL Datasets", help="Sends the entire database schema to AI. Higher cost/token usage.", value=False)



if "messages" not in st.session_state: st.session_state.messages = []

@@ -429,19 +436,27 @@

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:


                # === COST OPTIMIZATION LOGIC ===

                if use_full_context:


                    # Drop heavy description column to save tokens


                    context_df = df[['dataset_name', 'column_name', 'data_type', 'key']]


                    scope_msg = "You are viewing the FULL database schema (Descriptions omitted to save space)."


                    # === CHANGED: Include 'url' in full context ===


                    context_df = df[['dataset_name', 'column_name', 'data_type', 'key', 'url']]


                    scope_msg = "You are viewing the FULL database schema."

                else:


                    # Use full details but only for selected datasets


                    context_df = df[df['dataset_name'].isin(selected_datasets)] if selected_datasets else df.head(50)


                    # === CHANGED: Include 'url' in subset context ===


                    cols_needed = ['dataset_name', 'column_name', 'data_type', 'description', 'key', 'url']


                    # Handle case where user hasn't re-scraped yet


                    if 'url' not in df.columns: cols_needed.remove('url')


                    


                    context_df = df[df['dataset_name'].isin(selected_datasets)][cols_needed] if selected_datasets else df.head(50)

                    scope_msg = f"You are viewing a SUBSET of {len(selected_datasets)} selected datasets."



                system_msg = f"""

                You are an expert SQL Data Architect for Brightspace (D2L).

                Context Scope: {scope_msg}


                


                INSTRUCTIONS:


                1. If you mention a dataset, you MUST provide its Source URL if available in the 'url' column.


                2. Format links as: [Dataset Name](URL)


                

                Schema Data:

                {context_df.to_csv(index=False)}

                """
