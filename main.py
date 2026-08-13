#!/usr/bin/env python
# coding: utf-8

import json
import os
import time
import urllib.parse
from datetime import datetime
from typing import Dict, List
import feedparser
import requests
from dotenv import load_dotenv
from openai import OpenAI, APIError
from tqdm import tqdm

# Load environment variables
load_dotenv()

# ==========================================
# CONFIGURATION & INITIALIZATION
# ==========================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "google/gemini-2.0-flash-lite-preview-02-05:free"

WP_URL = os.getenv("WP_URL")  # e.g., "https://yourdomain.com"
WP_USERNAME = os.getenv("WP_USERNAME")
WP_APP_PASSWORD = os.getenv("WP_APP_PASSWORD")
PUBLISH_STATUS = os.getenv("PUBLISH_STATUS", "publish")  # 'publish' or 'draft'

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=120.0,
    default_headers={
        "HTTP-Referer": "https://localhost",
        "X-Title": "Biotech Catalyst Agent",
    }
)

# ==========================================
# CATEGORIES & SUBCATEGORIES
# ==========================================
ONCOLOGY_SUBCATEGORIES = {
    "General Oncology": [
        "oncology clinical trial results",
        "ASCO AACR Phase 3 readout",
        "FDA breakthrough designation cancer drug",
        "PDUFA oncology approval catalyst"
    ],
    "RAS Pathway Targeted Small Molecule Inhibitors": [
        "KRAS inhibitor clinical trial",
        "pan-RAS inhibitor trial readout",
        "KRAS G12C cancer pipeline",
        "FDA breakthrough KRAS inhibitor"
    ]
}

PRIMARY_CATEGORIES = {
    "Oncology": ONCOLOGY_SUBCATEGORIES
}

SUBCATEGORY_PARENTS = ["Oncology"]

# ==========================================
# 1. RSS NEWS INGESTION ENGINE
# ==========================================

def fetch_google_news_rss(query: str, max_results: int = 5) -> List[Dict]:
    encoded_query = urllib.parse.quote(query)
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

    print(f"\n[*] Starting RSS ingestion for: {category_label}")
    for q in tqdm(queries, desc="Fetching News Feeds", unit="query"):
        articles = fetch_google_news_rss(q, max_results=5)
        for art in articles:
            if art["title"] not in seen_titles:
                seen_titles.add(art["title"])
                aggregated_articles.append(art)

    print(f"[+] Ingestion complete. Total unique articles retrieved: {len(aggregated_articles)}")
    return aggregated_articles

# ==========================================
# 2. LLM EXTRACTION ENGINE
# ==========================================

SYSTEM_PROMPT = """You are a Senior Biotechnology Equity Research Analyst and Biotech Catalyst Specialist.
Your task is to analyze news, research summaries, and market signals to identify, score, and rank the TOP 5 drug candidates receiving the highest attention in a given category or subcategory.

Always respond with valid JSON formatting.
"""

def analyze_catalysts_with_llm(category_label: str, news_data: List[Dict]) -> Dict:
    formatted_news = "\n".join([
        f"- Title: {item['title']}\n  Date: {item['published']}\n  Source: {item.get('source', 'Google News')}\n  Snippet: {item['summary']}\n"
        for item in news_data
    ]) if news_data else "No recent RSS news items retrieved."

    user_prompt = f"""
Category / Subcategory to Analyze: {category_label}
News: {formatted_news}

Return the TOP 5 DRUG CANDIDATES formatted strictly as valid JSON adhering to this schema:
{{
  "category": "{category_label}",
  "total_candidates_found": 5,
  "ranked_drugs": [
    {{
      "rank": 1,
      "drug_name": "Drug Name",
      "company_ticker": "Company (TICKER)",
      "mechanism_of_action": "MoA description",
      "attention_driver": "Key reason for recent attention",
      "historical_track_record_and_risks": "Historical context",
      "stock_market_context": {{
        "ticker": "TICKER",
        "stock_price_range_est": "Range",
        "analyst_rating": "Rating",
        "target_price_est": "Target Price",
        "stock_trend_expectation": "Trend"
      }},
      "clinical_trials": {{
        "current_phase": "Phase",
        "recent_readout_data": "Readout data",
        "predicted_finish_date": "Finish date",
        "pdufa_or_key_dates": "Key dates"
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
        "opportunities": ["Opportunity 1"],
        "risks": ["Risk 1"]
      }},
      "score_justification": {{
        "fda_status": "Status",
        "peak_sales_estimate": "Estimate",
        "valuation_commentary": "Commentary"
      }},
      "weighted_priority_score_total": 8.3,
      "investment_thesis": "Thesis sentence"
    }}
  ]
}}
"""

    print(f"\n[*] Querying LLM ({MODEL_NAME})...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    cleaned_content = response.choices[0].message.content.strip()
    return json.loads(cleaned_content)

# ==========================================
# 3. HTML BUILDER & WORDPRESS PUBLISHER
# ==========================================

def build_blog_html(analysis: Dict, news_data: List[Dict]) -> str:
    ranked = analysis.get("ranked_drugs", [])[:5]
    cat = analysis.get("category", "Unknown Category")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = [f"<h2>Biotech Catalyst Intelligence Report</h2><p><b>Category:</b> {cat}<br><b>Generated:</b> {now}</p>"]
    html.append("<h3>Top 5 Leaderboard</h3><ol>")
    for d in ranked:
        html.append(f"<li><b>{d.get('drug_name','N/A')}</b> ({d.get('company_ticker','N/A')}) - Score: {d.get('weighted_priority_score_total','N/A')}/10</li>")
    html.append("</ol>")

    for d in ranked:
        html.append(f"<hr><h3>Rank #{d.get('rank','-')}: {d.get('drug_name','N/A')}</h3>")
        html.append(f"<p><b>Company:</b> {d.get('company_ticker','N/A')}<br><b>MoA:</b> {d.get('mechanism_of_action','N/A')}</p>")
        html.append(f"<p><b>Investment Thesis:</b> {d.get('investment_thesis','N/A')}</p>")

    html.append("<hr><h3>Resources</h3><ul>")
    for item in news_data:
        html.append(f'<li><a href="{item.get("link","#")}" target="_blank">{item.get("title","Link")}</a></li>')
    html.append("</ul>")

    return "\n".join(html)

def publish_to_wordpress(title: str, html_content: str) -> Dict:
    """Publishes a new post directly to WordPress via standard REST API."""
    if not WP_URL or not WP_USERNAME or not WP_APP_PASSWORD:
        raise ValueError("Missing WP_URL, WP_USERNAME, or WP_APP_PASSWORD in environment variables.")

    # Format the REST API post endpoint URL
    clean_url = WP_URL.rstrip('/')
    api_url = f"{clean_url}/wp-json/wp/v2/posts"

    # Define post body and status
    post_data = {
        "title": title,
        "content": html_content,
        "status": PUBLISH_STATUS.lower()  # 'publish' or 'draft'
    }

    # Send POST request using HTTP Basic Auth
    response = requests.post(
        api_url,
        auth=(WP_USERNAME, WP_APP_PASSWORD),
        json=post_data,
        timeout=30
    )

    if response.status_code not in [200, 201]:
        raise Exception(f"WordPress API Error ({response.status_code}): {response.text}")

    return response.json()

# ==========================================
# MAIN EXECUTION
# ==========================================

def run_agent():
    selected_category_name = os.getenv("CATEGORY_NAME", "Oncology")
    selected_subcat_name = os.getenv("SUBCATEGORY_NAME", "RAS Pathway Targeted Small Molecule Inhibitors")

    subcat_dict = PRIMARY_CATEGORIES[selected_category_name]
    queries_to_run = subcat_dict[selected_subcat_name]
    selected_label = f"{selected_category_name} - {selected_subcat_name}"

    news_items = gather_category_data(selected_label, queries_to_run)
    report = analyze_catalysts_with_llm(selected_label, news_items)

    post_title = f"Biotech Catalyst Report: {selected_label} ({datetime.utcnow().strftime('%Y-%m-%d')})"
    post_html = build_blog_html(report, news_items)
    
    published = publish_to_wordpress(post_title, post_html)
    print(f"[+] WordPress Post Published Successfully! Post URL: {published.get('link')}")

if __name__ == "__main__":
    run_agent()
