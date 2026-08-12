#!/usr/bin/env python
# coding: utf-8

import json
import os
import time
import urllib.parse
from datetime import datetime
from typing import Dict, List
import feedparser
from dotenv import load_dotenv
from openai import OpenAI

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

# ==========================================
# RECONSTRUCT CREDENTIALS (FOR CI/CD PLATFORMS)
# ==========================================
if not os.path.exists('token.json') and os.getenv("TOKEN_JSON"):
    with open('token.json', 'w') as f:
        f.write(os.getenv("TOKEN_JSON"))

if not os.path.exists('client_secret.json') and os.getenv("CLIENT_SECRET_JSON"):
    with open('client_secret.json', 'w') as f:
        f.write(os.getenv("CLIENT_SECRET_JSON"))

# ==========================================
# CONFIGURATION & INITIALIZATION
# ==========================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BLOGGER_BLOG_ID = os.getenv("BLOGGER_BLOG_ID")
MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b:free"

SCOPES = ['https://www.googleapis.com/auth/blogger']

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://localhost",
        "X-Title": "Biotech Catalyst Publisher",
    }
)

# ==========================================
# CATEGORIES SETUP
# ==========================================
ONCOLOGY_SUBCATEGORIES = {
    "General Oncology": ["oncology clinical trial results", "ASCO AACR Phase 3 readout"],
    "Antibody-Drug Conjugates (ADC)": ["ADC clinical trial readout", "antibody drug conjugate pipeline"],
    "Bispecific Antibodies": ["bispecific antibody clinical trial", "bispecific T-cell engager pipeline"],
    "RAS Pathway Inhibitors": ["KRAS inhibitor clinical trial", "pan-RAS inhibitor trial readout"],
    "Emerging Cancer Vaccines": ["cancer vaccine clinical trial", "mRNA cancer vaccine pipeline"],
    "CAR-T and Cell Therapy": ["CAR-T cell therapy clinical trial", "allogeneic cell therapy readout"]
}

NEUROSCIENCE_SUBCATEGORIES = {
    "General Neuroscience": ["neuroscience CNS clinical trial results", "Phase 3 readout neuroscience biotech"],
    "Schizophrenia and MDD": ["schizophrenia clinical trial readout", "depression drug pipeline catalyst"],
    "Alzheimer's and Neurodegenerative": ["Alzheimer clinical trial readout", "CTAD Alzheimer data readout"],
    "Rare Diseases and Movement Disorders": ["movement disorder trial readout", "orphan drug status neurological disease"]
}

IMMUNOLOGY_SUBCATEGORIES = {
    "General Immunology": ["immunology clinical trial results", "autoimmune Phase 3 readout"],
    "Biologics (mAbs)": ["monoclonal antibody autoimmune trial", "biologic autoimmune IL-23 IL-17"],
    "Calcineurin Inhibitors": ["calcineurin inhibitor clinical trial", "calcineurin inhibitor pipeline"],
    "Antimetabolites": ["antimetabolite immunosuppressive trial", "immunosuppressive drug trial results"],
    "JAK Inhibitors": ["JAK inhibitor clinical trial readout", "TYK2 inhibitor autoimmune trial"]
}

ANTI_DIABETICS_SUBCATEGORIES = {
    "General Anti-diabetics": ["diabetes clinical trial results", "Phase 3 readout diabetes drug"],
    "GLP-1 and Injectables": ["GLP-1 GIP clinical trial obesity", "GLP-1 diabetes trial readout"],
    "Oral Therapies": ["oral GLP-1 clinical trial readout", "oral diabetes drug pipeline catalyst"]
}

PRIMARY_CATEGORIES = {
    "Oncology": ONCOLOGY_SUBCATEGORIES,
    "Neuroscience": NEUROSCIENCE_SUBCATEGORIES, 
    "Immunology": IMMUNOLOGY_SUBCATEGORIES,
    "Anti-diabetics": ANTI_DIABETICS_SUBCATEGORIES,
    "AI Drug Discovery (AIDD)": [
        "AI drug discovery clinical trial",
        "AIDD candidate pipeline progress",
        "computational biology drug candidate trial"
    ]
}

SUBCATEGORY_PARENTS = ["Oncology", "Neuroscience", "Immunology", "Anti-diabetics"]


# ==========================================
# 1. RSS NEWS INGESTION
# ==========================================
def fetch_google_news_rss(query: str, max_results: int = 5) -> List[Dict]:
    clean_query = query.replace("(", "").replace(")", "").replace("/", " ").replace("-", " ").strip()
    time_bounded_query = f"{clean_query} when:3m"
    encoded_query = urllib.parse.quote(time_bounded_query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    feed = feedparser.parse(rss_url)
    articles = []
    
    for entry in feed.entries[:max_results]:
        source_title = getattr(entry, "source", {}).get("title", "Google News") if hasattr(entry, "source") else "Google News"
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": getattr(entry, "published", "N/A"),
            "summary": getattr(entry, "summary", ""),
            "source": source_title
        })
    return articles

def gather_category_data(category_label: str, queries: List[str]) -> List[Dict]:
    aggregated_articles = []
    seen_titles = set()
    
    for q in queries:
        articles = fetch_google_news_rss(q, max_results=5)
        for art in articles:
            if art["title"] not in seen_titles:
                seen_titles.add(art["title"])
                aggregated_articles.append(art)
                
    aggregated_articles.sort(key=lambda x: x["title"])
    return aggregated_articles


# ==========================================
# 2. LLM EXTRACTION
# ==========================================
SYSTEM_PROMPT = """You are a Senior Biotechnology Equity Research Analyst.
Your task is to analyze news and clinical data to identify, score, and rank the TOP 5 drug candidates in a given category.

Score each candidate (1 to 10 scale) using 6 criteria:
1. Market Size & Unmet Need
2. Market Competition & Saturation
3. Predicted Peak Sales
4. Valuation & Pricing Dynamics
5. Novelty, Moat & FDA Status
6. Catalysts & Sentiment

Always return strict valid JSON matching the requested schema.
"""

def analyze_catalysts_with_llm(category_label: str, news_data: List[Dict]) -> Dict:
    formatted_news = "\n".join([
        f"- Title: {item['title']}\n  Date: {item['published']}\n  Source: {item['source']}\n  Snippet: {item['summary']}\n"
        for item in news_data
    ]) if news_data else "No RSS news retrieved. Rely on biopharma domain expertise."

    user_prompt = f"""
Category / Subcategory to Analyze: {category_label}

Recent News & Data:
{formatted_news}

Identify and rank the TOP 5 DRUG CANDIDATES in {category_label}.
Return valid JSON adhering to this schema:

{{
  "category": "{category_label}",
  "total_candidates_found": 5,
  "ranked_drugs": [
    {{
      "rank": 1,
      "drug_name": "Drug Name / Code",
      "company_ticker": "Company (TICKER)",
      "mechanism_of_action": "MoA description",
      "attention_driver": "Key catalyst in last 1-3 months",
      "historical_track_record_and_risks": "Past trial failures or safety history",
      "stock_market_context": {{
        "ticker": "NASDAQ: TICKER",
        "stock_price_range_est": "Trading price context",
        "analyst_rating": "Buy/Hold/Sell",
        "target_price_est": "$XX.XX",
        "stock_trend_expectation": "Upward/Downward/Volatile"
      }},
      "clinical_trials": {{
        "current_phase": "Phase 2 / Phase 3 / PDUFA",
        "recent_readout_data": "Key data readout summary",
        "predicted_finish_date": "Est. completion window",
        "pdufa_or_key_dates": "Target dates"
      }},
      "priority_scores": {{
        "market_size_unmet_need": 9,
        "competition_saturation_inverse": 8,
        "predicted_peak_sales_potential": 9,
        "valuation_attractiveness": 7,
        "novelty_and_fda_status": 9,
        "sentiment_and_market_noise": 8
      }},
      "investment_analysis":{{
        "opportunities": ["Opp 1", "Opp 2"],
        "risks": ["Risk 1", "Risk 2"]
      }},
      "score_justification": {{
        "fda_status": "Breakthrough / Fast Track / None",
        "peak_sales_estimate": "$B estimate",
        "valuation_commentary": "Valuation assessment"
      }},
      "weighted_priority_score_total": 8.3,
      "investment_thesis": "One-sentence investment outlook"
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        top_p=1.0,
        seed=42,
        response_format={"type": "json_object"}
    )
    
    raw_content = response.choices[0].message.content.strip()
    return json.loads(raw_content)


# ==========================================
# 3. HTML BLOG BUILDER
# ==========================================
def build_html_blog_post(analysis: Dict) -> str:
    category = analysis.get("category", "Biotech").title()
    ranked_drugs = analysis.get("ranked_drugs", [])
    
    html = f"<h2>Daily Catalyst Intelligence Report: {category}</h2>"
    html += f"<p><em>Automated market overview compiled on {datetime.now().strftime('%B %d, %Y')}.</em></p>"
    
    # Leaderboard Table
    html += "<h3>Top 5 Ranked Candidates</h3>"
    html += "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse: collapse; width: 100%;'>"
    html += "<tr style='background-color: #f2f2f2;'><th>Rank</th><th>Candidate</th><th>Ticker</th><th>Phase</th><th>Score</th></tr>"
    for drug in ranked_drugs:
        html += f"<tr>"
        html += f"<td><b>#{drug.get('rank')}</b></td>"
        html += f"<td>{drug.get('drug_name')}</td>"
        html += f"<td>{drug.get('company_ticker')}</td>"
        html += f"<td>{drug.get('clinical_trials', {}).get('current_phase')}</td>"
        html += f"<td><b>{drug.get('weighted_priority_score_total')}/10</b></td>"
        html += f"</tr>"
    html += "</table><br/>"
    
    # Candidate Profiles
    html += "<h3>Detailed Pipeline Breakdown</h3>"
    for drug in ranked_drugs:
        trials = drug.get("clinical_trials", {})
        stock = drug.get("stock_market_context", {})
        scores = drug.get("priority_scores", {})
        inv = drug.get("investment_analysis", {})
        
        html += f"<div style='border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 5px;'>"
        html += f"<h4>#{drug.get('rank')} - {drug.get('drug_name')} ({drug.get('company_ticker')})</h4>"
        html += f"<p><b>Mechanism of Action:</b> {drug.get('mechanism_of_action')}</p>"
        html += f"<p><b>Attention Driver:</b> {drug.get('attention_driver')}</p>"
        
        html += "<h5>Clinical Status & Catalyst Window</h5>"
        html += f"<ul>"
        html += f"<li><b>Current Phase:</b> {trials.get('current_phase')}</li>"
        html += f"<li><b>Readout Data:</b> {trials.get('recent_readout_data')}</li>"
        html += f"<li><b>PDUFA / Key Target Dates:</b> {trials.get('pdufa_or_key_dates')}</li>"
        html += f"</ul>"
        
        html += "<h5>Equity & Financial Context</h5>"
        html += f"<ul>"
        html += f"<li><b>Price Context:</b> {stock.get('stock_price_range_est')}</li>"
        html += f"<li><b>Analyst Consensus:</b> {stock.get('analyst_rating')} (Target: {stock.get('target_price_est')})</li>"
        html += f"</ul>"

        html += "<h5>Opportunities vs Risks</h5>"
        html += f"<p><b>Opportunities:</b> {', '.join(inv.get('opportunities', []))}</p>"
        html += f"<p><b>Risks:</b> {', '.join(inv.get('risks', []))}</p>"

        html += f"<p><b>Investment Thesis:</b> <em>{drug.get('investment_thesis')}</em></p>"
        html += f"</div>"
        
    return html


# ==========================================
# 4. GOOGLE BLOGGER PUBLISHER
# ==========================================
def get_blogger_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        with open('token.json', 'w') as token_file:
            token_file.write(creds.to_json())

    return build('blogger', 'v3', credentials=creds)

def publish_to_blogger(service, title: str, content_html: str, labels: List[str]):
    if not BLOGGER_BLOG_ID:
        print("[!] BLOGGER_BLOG_ID missing.")
        return

    body = {
        "kind": "blogger#post",
        "title": title,
        "content": content_html,
        "labels": labels
    }
    
    try:
        posts = service.posts()
        result = posts.insert(blogId=BLOGGER_BLOG_ID, body=body).execute()
        print(f"[+] Post published: {result.get('url')}")
    except Exception as e:
        print(f"[!] Error publishing to Blogger: {e}")


# ==========================================
# 5. BATCH AUTOMATION RUNNER
# ==========================================
def run_all_categories():
    print("="*80)
    print(f" STARTING BATCH BIOTECH ANALYSIS & PUBLISHING: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)

    try:
        blogger_service = get_blogger_service()
    except Exception as e:
        print(f"[!] Authentication Error: {e}")
        return

    category_queue = []

    for cat_name, content in PRIMARY_CATEGORIES.items():
        if cat_name in SUBCATEGORY_PARENTS:
            for sub_name, queries in content.items():
                category_queue.append((f"{cat_name} - {sub_name}", cat_name, queries))
        else:
            category_queue.append((cat_name, cat_name, content))

    for label, parent_cat, queries in category_queue:
        print(f"\n[>>>] Processing Category: {label}")
        news_items = gather_category_data(label, queries)
        
        try:
            report = analyze_catalysts_with_llm(label, news_items)
            post_html = build_html_blog_post(report)
            post_title = f"{label} Top 5 Candidates Report ({datetime.now().strftime('%b %d, %Y')})"
            labels = ["Biotech Intelligence", parent_cat]
            
            publish_to_blogger(blogger_service, post_title, post_html, labels)
        except Exception as e:
            print(f"[!] Error processing {label}: {e}")
            
        time.sleep(5)

if __name__ == "__main__":
    run_all_categories()
