#!/usr/bin/env python
# coding: utf-8

import json
import os
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

# Stable free model on OpenRouter
MODEL_NAME = "google/gemini-2.0-flash-lite-preview-02-05:free"

SCOPES = ['https://www.googleapis.com/auth/blogger']

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://localhost",
        "X-Title": "Biotech Catalyst Publisher",
    }
)

# Target strictly RAS Pathway Inhibitors
CATEGORY_LABEL = "Oncology - RAS Pathway Inhibitors"
RAS_QUERIES = [
    "KRAS inhibitor clinical trial", 
    "pan-RAS inhibitor trial readout",
    "KRAS G12C G12D inhibitor pipeline"
]

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

def gather_category_data(queries: List[str]) -> List[Dict]:
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
Category to Analyze: {category_label}

Recent News & Data:
{formatted_news}

Identify and rank the TOP 5 DRUG CANDIDATES in {category_label}.
Return valid JSON adhering strictly to this structure:

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
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    raw_content = response.choices[0].message.content.strip()
    return json.loads(raw_content)


# ==========================================
# 3. HTML BLOG BUILDER (WITH NEWS LINKS)
# ==========================================
def build_html_blog_post(analysis: Dict, news_data: List[Dict]) -> str:
    if not isinstance(analysis, dict):
        raise ValueError("Invalid LLM response received: output is not a dictionary.")

    category = analysis.get("category", "Oncology - RAS Pathway Inhibitors").title()
    ranked_drugs = analysis.get("ranked_drugs", [])
    
    html = f"<h2>Daily Catalyst Intelligence Report: {category}</h2>"
    html += f"<p><em>Automated market overview compiled on {datetime.now().strftime('%B %d, %Y')}.</em></p>"
    
    # Leaderboard Table
    html += "<h3>Top 5 Ranked Candidates</h3>"
    html += "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse: collapse; width: 100%;'>"
    html += "<tr style='background-color: #f2f2f2;'><th>Rank</th><th>Candidate</th><th>Ticker</th><th>Phase</th><th>Score</th></tr>"
    for drug in ranked_drugs:
        html += f"<tr>"
        html += f"<td><b>#{drug.get('rank', 'N/A')}</b></td>"
        html += f"<td>{drug.get('drug_name', 'N/A')}</td>"
        html += f"<td>{drug.get('company_ticker', 'N/A')}</td>"
        html += f"<td>{drug.get('clinical_trials', {}).get('current_phase', 'N/A')}</td>"
        html += f"<td><b>{drug.get('weighted_priority_score_total', 'N/A')}/10</b></td>"
        html += f"</tr>"
    html += "</table><br/>"
    
    # Candidate Profiles
    html += "<h3>Detailed Pipeline Breakdown</h3>"
    for drug in ranked_drugs:
        trials = drug.get("clinical_trials", {})
        stock = drug.get("stock_market_context", {})
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
        
    # News Links & Sources Section
    if news_data:
        html += "<hr/><h3>Fetched News & Intelligence Sources</h3><ul>"
        for item in news_data:
            title = item.get("title", "News Source")
            link = item.get("link", "#")
            source = item.get("source", "Google News")
            published = item.get("published", "")
            
            html += f"<li><a href='{link}' target='_blank' rel='noopener noreferrer'><b>{title}</b></a> "
            html += f"<em>({source} - {published})</em></li>"
        html += "</ul>"
        
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
        print(f"[+] Successfully published to Blogger: {result.get('url')}")
    except Exception as e:
        print(f"[!] Error publishing to Blogger: {e}")


# ==========================================
# 5. SINGLE CATEGORY RUNNER
# ==========================================
def run_single_category():
    print("="*80)
    print(f" PROCESSING TARGET REPORT: {CATEGORY_LABEL}")
    print("="*80)

    try:
        blogger_service = get_blogger_service()
    except Exception as e:
        print(f"[!] Authentication Error: {e}")
        return

    # Step 1: Gather RSS news for RAS Oncology
    print(f"[1/3] Gathering RSS news for queries: {RAS_QUERIES}")
    news_items = gather_category_data(RAS_QUERIES)
    
    # Step 2: Run LLM Analysis
    print("[2/3] Analyzing top candidates with LLM...")
    try:
        report = analyze_catalysts_with_llm(CATEGORY_LABEL, news_items)
        
        # Pass news_items into the HTML builder to append news links
        post_html = build_html_blog_post(report, news_items)
        post_title = f"{CATEGORY_LABEL} Top 5 Candidates Report ({datetime.now().strftime('%b %d, %Y')})"
        labels = ["Biotech Intelligence", "Oncology", "RAS Inhibitors"]
        
        # Step 3: Publish post
        print("[3/3] Publishing report to Google Blogger...")
        publish_to_blogger(blogger_service, post_title, post_html, labels)
        
    except Exception as e:
        print(f"[!] Execution failed for {CATEGORY_LABEL}: {e}")

if __name__ == "__main__":
    run_single_category()
