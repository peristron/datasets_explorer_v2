# 🧠 Brightspace Dataset Explorer  — AI Edition  


Live app → [https:// tbd
(Internal / password-protected, for now)

What this does

- Instantly see **every PK/FK relationship** across all 38+ Brightspace Data Hub datasets (and growing) 
- Beautiful interactive graph with **click-to-copy SQL JOINs** on every edge  
- Natural language search**: just type “late quiz submissions with grades and penalties” → AI returns the exact datasets & columns  
- 1-click full SQL generation: select tables + describe your goal → get a perfect, ready-to-run query  
- Explain any dataset in plain English for non-technical staff  
- Full chat-with-your-data-hub at the bottom, in a way — ask anything, get answers ~instantly  
- Global search across every column description in the entire schema  
- Zero crashes, even with single datasets or zero relationships  
- Works with OpenAI (gpt-4o) or Grok (private, no data leaves your org)

No longer just a schema browser (vs earlier version of this code-base)


Screenshot [PLACEHOLDER]

![v200 AI Edition in action](https://i.imgur.com/YOUR_FUTURE_SCREENSHOT_HERE.png)
*(replace with actual screenshot — it will blow minds)*

Quick Start (Local)

```bash
git clone https://github.com/[you]/brightspace-dataset-explorer.git
cd brightspace-dataset-explorer
pip install -r requirements.txt

# Add keys to .streamlit/secrets.toml
cp secrets.example.toml .streamlit/secrets.toml
# → edit with your OpenAI + xAI keys

streamlit run python_datahub_dataset_relationships_v200_AI_EDITION.py
