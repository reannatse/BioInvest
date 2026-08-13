#!/usr/bin/env python
# coding: utf-8

import json
import os
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import feedparser
from dotenv import load_dotenv
from openai import OpenAI, APIError
from tqdm import tqdm

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()

# ==========================================
# CONFIGURATION & INITIALIZATION
# ==========================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b:free"

BLOG_ID = os.getenv("BLOGGER_BLOG_ID") or os.getenv("BLOG_ID")
PUBLISH_STATUS = os.getenv("PUBLISH_STATUS", "LIVE")

TOKEN_FILE = "token.json"
CLIENT_SECRET_FILE = "client_secret.json"

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
    "Antibody-Drug Conjugates (ADC)": [
        "ADC clinical trial readout",
        "antibody drug conjugate pipeline",
        "ASCO ADC oncology trial results",
        "FDA fast track ADC cancer drug"
    ],
    "Bispecific and Multispecific Antibodies": [
        "bispecific antibody clinical trial",
        "multispecific antibody cancer readout",
        "bispecific T-cell engager pipeline",
        "FDA breakthrough bispecific antibody"
    ],
    "RAS Pathway Targeted Small Molecule Inhibitors": [
        "KRAS inhibitor clinical trial",
        "pan-RAS inhibitor trial readout",
        "KRAS G12C cancer pipeline",
        "FDA breakthrough KRAS inhibitor"
    ],
    "Emerging Cancer Vaccines": [
        "cancer vaccine clinical trial",
        "mRNA cancer vaccine pipeline",
        "therapeutic cancer vaccine readout",
        "FDA fast track cancer vaccine"
    ],
    "CAR-T and Cell Therapy": [
        "CAR-T cell therapy clinical trial",
        "allogeneic cell therapy readout",
        "ASCO ASH cell therapy data",
        "FDA fast track CAR-T"
    ]
}

NEUROSCIENCE_SUBCATEGORIES = {
    "General Neuroscience": [
        "neuroscience CNS clinical trial results",
        "Phase 3 readout neuroscience biotech",
        "FDA breakthrough designation CNS drug",
        "PDUFA neuroscience approval catalyst"
    ],
    "Schizophrenia and Major Depressive Disorder (MDD)": [
        "schizophrenia clinical trial readout",
        "depression drug pipeline catalyst",
        "muscarinic agonist schizophrenia trial",
        "FDA fast track psychiatric drug"
    ],
    "Alzheimer's and Neurodegenerative Disease": [
        "Alzheimer clinical trial readout",
        "Parkinson ALS pipeline catalyst",
        "CTAD Alzheimer data readout",
        "FDA approval neurodegenerative disease"
    ],
    "Rare Diseases and Movement Disorders": [
        "movement disorder trial readout",
        "Huntington SMA ataxia trial results",
        "orphan drug status neurological disease",
        "FDA fast track movement disorder drug"
    ]
}

IMMUNOLOGY_SUBCATEGORIES = {
    "General Immunology": [
        "immunology clinical trial results",
        "autoimmune Phase 3 readout",
        "FDA breakthrough immunology drug",
        "autoimmune drug approval catalyst"
    ],
    "Biologics (Monoclonal Antibodies)": [
        "monoclonal antibody autoimmune trial",
        "biologic autoimmune IL-23 IL-17",
        "monoclonal antibody immunology results",
        "biologic autoimmune clinical trial"
    ],
    "Calcineurin Inhibitors": [
        "calcineurin inhibitor clinical trial",
        "tacrolimus cyclosporine trial readout",
        "calcineurin autoimmune drug",
        "calcineurin inhibitor pipeline"
    ],
    "Antimetabolites and Cytotoxic Agents": [
        "antimetabolite immunosuppressive trial",
        "methotrexate mycophenolate pipeline",
        "cytotoxic autoimmune disease trial",
        "immunosuppressive drug trial results"
    ],
    "Janus Kinase (JAK) Inhibitors": [
        "JAK inhibitor clinical trial readout",
        "TYK2 inhibitor autoimmune trial",
        "JAK inhibitor trial results",
        "TYK2 JAK inhibitor biotech news"
    ]
}

ANTI_DIABETICS_SUBCATEGORIES = {
    "General Anti-diabetics": [
        "diabetes clinical trial results",
        "Phase 3 readout diabetes drug",
        "FDA approval catalyst diabetes",
        "anti-diabetic pipeline trial results"
    ],
    "GLP-1 and Injectable Therapies": [
        "GLP-1 GIP clinical trial obesity",
        "GLP-1 diabetes trial readout",
        "obesity metabolic drug readout",
        "ADA conference GLP-1 results"
    ],
    "Oral Therapies": [
        "oral GLP-1 clinical trial readout",
        "SGLT2 DPP-4 inhibitor trial",
        "oral diabetes drug pipeline catalyst",
        "FDA fast track oral anti-diabetic"
    ]
}

PRIMARY_CATEGORIES = {
    "Oncology": ONCOLOGY_SUBCATEGORIES,
    "Neuroscience": NEUROSCIENCE_SUBCATEGORIES,
    "Immunology": IMMUNOLOGY_SUBCATEGORIES,
    "Anti-diabetics": ANTI_DIABETICS_SUBCATEGORIES,
    "AI Drug Discovery(AIDD)": [
        "AI drug discovery clinical trial",
        "AIDD candidate pipeline progress",
        "computational biology drug candidate trial",
        "AIDD candidate pipeline FDA status"
    ]
}

SUBCATEGORY_PARENTS = ["Oncology", "Neuroscience", "Immunology", "Anti-diabetics"]


# ==========================================
# 1. RSS NEWS & CATALYST INGESTION ENGINE
# ==========================================

def fetch_google_news_rss(query: str, max_results: int = 5) -> List[Dict]:
    """Fetches news items from Google News RSS."""
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
    """Collects news data across queries using tqdm to render a progress bar."""
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

def display_retrieved_sources(news_data: List[Dict]):
    """Displays a list of fetched news articles and links for user review."""
    print("\n" + "="*85)
    print(" RETRIEVED NEWS SOURCES & ARTICLES")
    print("="*85)

    if not news_data:
        print(" [!] No RSS news articles were retrieved for review.")
        print("="*85)
        return

    for idx, item in enumerate(news_data, 1):
        print(f"\n [{idx}] {item['title']}")
        print(f" • Source: {item.get('source', 'Google News')}")
        print(f" • Date : {item['published']}")
        print(f" • URL : {item['link']}")
        print("="*85)

# ==========================================
# 2. LLM EXTRACTION & RANKING ENGINE
# ==========================================

SYSTEM_PROMPT = """You are a Senior Biotechnology Equity Research Analyst and Biotech Catalyst Specialist.
Your task is to analyze news, research summaries, and market signals to identify, score, and rank the TOP 5 drug candidates receiving the highest attention in a given category or subcategory.

You must score each candidate (1 to 10 scale) using 6 strict dimensions:
1. Market Size & Unmet Need
2. Market Competition & Saturation (10 = low saturation/strong moat)
3. Predicted Peak Sales & Revenue Potential
4. Valuation & Stock Pricing Dynamics (10 = attractive risk-reward)
5. Novelty, Moat & FDA Special Status (Breakthrough, Fast Track, Orphan)
6. Catalysts, Readout Dates & Sentiment/Market Noise

Strict Rules:
1. Prioritize candidates driven by recent news within the last 1-3 months.
2. Contextualize candidates by documenting any past trial failures, clinical holds, or safety concerns.
3. For public companies, evaluate stock performance, Wall Street analyst consensus, price targets, and ratings (e.g., Strong Buy, Hold, Sell).
4. Always respond with valid JSON formatting.
"""

def analyze_catalysts_with_llm(category_label: str, news_data: List[Dict]) -> Dict:
    if news_data:
        formatted_news = "\n".join([
            f"- Title: {item['title']}\n  Date: {item['published']}\n  Source: {item.get('source', 'Google News')}\n  Snippet: {item['summary']}\n"
            for item in news_data
        ])
    else:
        formatted_news = "No recent RSS news items retrieved. Rely on your domain knowledge of recent clinical trials and catalysts."

    user_prompt = f"""
Category / Subcategory to Analyze: {category_label}

Here is the compiled news, conference updates, published papers and market reports from the past 1-3 months:
{formatted_news}

Based on the news provided and your biopharma domain expertise:
1. Identify the TOP 5 DRUG CANDIDATES currently driving attention or clinical catalysts specifically in {category_label}.
2. Extract clinical trial readouts, key timeline dates, and company tickers for each candidate.
3. Score each drug across the 6 priority criteria (1-10 scale) and calculate `weighted_priority_score_total` (average of the 6 scores).
4. SORT the list in DESCENDING order (Rank #1 = highest overall catalyst score).

Analyze the data and return the TOP 5 DRUG CANDIDATES formatted strictly as valid JSON adhering to this schema:

{{
  "category": "{category_label}",
  "total_candidates_found": 5,
  "ranked_drugs": [
    {{
      "rank": 1,
      "drug_name": "Drug Name / Code",
      "company_ticker": "Company (TICKER) or Private",
      "mechanism_of_action": "Brief MoA description",
      "attention_driver": "Key reason for recent market attention. Specific catalyst/news event occurring in the last 1-3 months.",
      "historical_track_record_and_risks": "Past trial failures, CRLs, clinical holds, or adverse safety history to keep in mind",
      "stock_market_context": {{
        "ticker": "NASDAQ: TICKER or N/A",
        "stock_price_range_est": "Current trading price / 52-week context",
        "analyst_rating": "Strong Buy / Buy / Hold / Sell",
        "target_price_est": "Consensus 12-month Price Target (e.g., $45.00)",
        "stock_trend_expectation": "Generally Upward / Downward / Volatile near catalyst"
      }},
      "clinical_trials": {{
        "current_phase": "Phase 2 / Phase 3 / PDUFA",
        "recent_readout_data": "Key data readout summary",
        "predicted_finish_date": "Est. completion window",
        "pdufa_or_key_dates": "Specific target dates or quarter"
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
        "opportunities": [
          "First-in-class / best-in-class potential in large indication",
          "High M&A / licensing buyout target for large pharma",
          "Favorable FDA designation accelerates market entry"
        ],
        "risks": [
          "Off-target toxicity or adverse safety profile in Phase 3",
          "Commercial overcrowding and pricing pressure from rivals",
          "Risk of trial failure or delayed primary endpoint data"
        ]
      }},
      "score_justification": {{
        "fda_status": "Fast Track / Breakthrough / None",
        "peak_sales_estimate": "$B estimate",
        "valuation_commentary": "Brief valuation risk assessment"
      }},
      "weighted_priority_score_total": 8.3,
      "investment_thesis": "One-sentence bottom-line investment/catalyst outlook"
    }}
  ]
}}
"""

    print(f"\n[*] Querying LLM ({MODEL_NAME})...")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                stream=True
            )
            break
        except APIError as e:
            if attempt == max_retries:
                raise e
            print(f"[!] Server busy. Retrying in {attempt * 3} seconds...")
            time.sleep(attempt * 3)

    raw_content = ""
    ESTIMATED_RESPONSE_LENGTH = 5000

    with tqdm(
        total=ESTIMATED_RESPONSE_LENGTH,
        desc="LLM Analysis Progress",
        bar_format="{l_bar}{bar:30}| {percentage:3.0f}% [{elapsed}<{remaining}]",
        leave=True,
        mininterval=0.1
    ) as pbar:
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            if content:
                raw_content += content
                increment = min(len(content), max(0, ESTIMATED_RESPONSE_LENGTH - pbar.n))
                pbar.update(increment)

        pbar.update(max(0, ESTIMATED_RESPONSE_LENGTH - pbar.n))

    print("\n[+] Analysis complete. Parsing structured report...")

    cleaned_content = raw_content.strip()
    if cleaned_content.startswith("```json"):
        cleaned_content = cleaned_content.replace("```json", "", 1)
    if cleaned_content.startswith("```"):
        cleaned_content = cleaned_content.replace("```", "", 1)
    if cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[:-3]

    return json.loads(cleaned_content.strip())


# ==========================================
# 3. HTML BLOG BUILDER & DISPLAY ENGINE
# ==========================================

def display_report(analysis: Dict):
    """Renders a Top 5 Leaderboard followed by drug breakdowns to console."""
    ranked_list = analysis.get("ranked_drugs", [])[:5]
    cat = analysis.get("category", "").upper()

    print("\n" + "="*85)
    print(f" BIOTECH CATALYST INTELLIGENCE REPORT | TARGET: {cat}")
    print("="*85)

    print("\n [ TOP 5 RANKED CANDIDATES LEADERBOARD ]\n")
    print(f"{'Rank':<5} | {'Drug Candidate':<18} | {'Ticker':<12} | {'Phase':<10} | {'Catalyst Score':<14}")
    print("-" * 70)
    for drug in ranked_list:
        trials = drug.get("clinical_trials", {})
        print(f"{drug.get('rank', '-'):<5} | {drug.get('drug_name', 'N/A')[:18]:<18} | {drug.get('company_ticker', 'N/A')[:12]:<12} | {trials.get('current_phase', 'N/A')[:10]:<10} | {drug.get('weighted_priority_score_total', 0):.1f} / 10")
    print("-" * 70)

def build_blog_html(analysis: Dict, news_data: List[Dict]) -> str:
    """Builds HTML formatted string for Google Blogger including Resources section."""
    ranked = analysis.get("ranked_drugs", [])[:5]
    cat = analysis.get("category", "Unknown Category")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = []
    html.append(f"<h2>Biotech Catalyst Intelligence Report</h2>")
    html.append(f"<p><b>Category:</b> {cat}<br><b>Generated:</b> {now}</p>")

    html.append("<h3>Top 5 Leaderboard</h3>")
    html.append("<ol>")
    for d in ranked:
        html.append(
            f"<li><b>{d.get('drug_name','N/A')}</b> ({d.get('company_ticker','N/A')}) "
            f"- Score: {d.get('weighted_priority_score_total','N/A')}/10, "
            f"Phase: {d.get('clinical_trials',{}).get('current_phase','N/A')}</li>"
        )
    html.append("</ol>")

    for d in ranked:
        trials = d.get("clinical_trials", {})
        stock = d.get("stock_market_context", {})
        scores = d.get("priority_scores", {})
        just = d.get("score_justification", {})
        inv = d.get("investment_analysis", {})

        html.append(f"<hr><h3>Rank #{d.get('rank','-')}: {d.get('drug_name','N/A')}</h3>")
        html.append(f"<p><b>Company:</b> {d.get('company_ticker','N/A')}<br>")
        html.append(f"<b>MoA:</b> {d.get('mechanism_of_action','N/A')}<br>")
        html.append(f"<b>Attention Driver:</b> {d.get('attention_driver','N/A')}</p>")

        html.append(f"<p><b>Historical Risks:</b> {d.get('historical_track_record_and_risks','N/A')}</p>")

        html.append("<p><b>Stock Context</b><br>")
        html.append(f"Ticker: {stock.get('ticker','N/A')}<br>")
        html.append(f"Price Range: {stock.get('stock_price_range_est','N/A')}<br>")
        html.append(f"Analyst Rating: {stock.get('analyst_rating','N/A')}<br>")
        html.append(f"Target Price: {stock.get('target_price_est','N/A')}<br>")
        html.append(f"Trend: {stock.get('stock_trend_expectation','N/A')}</p>")

        html.append("<p><b>Clinical Timeline</b><br>")
        html.append(f"Phase: {trials.get('current_phase','N/A')}<br>")
        html.append(f"Readout: {trials.get('recent_readout_data','N/A')}<br>")
        html.append(f"Finish: {trials.get('predicted_finish_date','N/A')}<br>")
        html.append(f"Key Dates: {trials.get('pdufa_or_key_dates','N/A')}</p>")

        html.append("<p><b>Scoring</b><br>")
        html.append(f"Market Size: {scores.get('market_size_unmet_need','N/A')}, ")
        html.append(f"Competition(Inverse): {scores.get('competition_saturation_inverse','N/A')}, ")
        html.append(f"Peak Sales: {scores.get('predicted_peak_sales_potential','N/A')}, ")
        html.append(f"Valuation: {scores.get('valuation_attractiveness','N/A')}, ")
        html.append(f"Novelty/FDA: {scores.get('novelty_and_fda_status','N/A')}, ")
        html.append(f"Sentiment: {scores.get('sentiment_and_market_noise','N/A')}</p>")

        html.append("<p><b>Opportunities</b></p><ul>")
        for o in inv.get("opportunities", []):
            html.append(f"<li>{o}</li>")
        html.append("</ul><p><b>Risks</b></p><ul>")
        for r in inv.get("risks", []):
            html.append(f"<li>{r}</li>")
        html.append("</ul>")

        html.append(f"<p><b>FDA Status:</b> {just.get('fda_status','N/A')}<br>")
        html.append(f"<b>Peak Sales Estimate:</b> {just.get('peak_sales_estimate','N/A')}<br>")
        html.append(f"<b>Valuation Commentary:</b> {just.get('valuation_commentary','N/A')}</p>")

        html.append(f"<p><b>Investment Thesis:</b> {d.get('investment_thesis','N/A')}</p>")

    # Resources section
    html.append("<hr><h3>Resources</h3>")
    if news_data:
        html.append("<ul>")
        for item in news_data:
            title = item.get("title", "Untitled")
            link = item.get("link", "#")
            pub = item.get("published", "N/A")
            html.append(f'<li><a href="{link}" target="_blank">{title}</a> <i>({pub})</i></li>')
        html.append("</ul>")
    else:
        html.append("<p>No source links retrieved from RSS for this run.</p>")

    return "\n".join(html)


# ==========================================
# 4. GOOGLE BLOGGER PUBLISHER
# ==========================================

def get_blogger_credentials() -> Credentials:
    """Loads credentials from token.json and automatically refreshes them if expired."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, ["[https://www.googleapis.com/auth/blogger](https://www.googleapis.com/auth/blogger)"])

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    if not creds or not creds.valid:
        raise ValueError("Valid Google OAuth credentials could not be loaded from token.json.")

    return creds

def publish_to_blogger(title: str, html_content: str) -> Dict:
    if not BLOG_ID:
        raise ValueError("Missing BLOGGER_BLOG_ID in environment variables.")

    creds = get_blogger_credentials()
    service = build("blogger", "v3", credentials=creds, cache_discovery=False)

    body = {
        "kind": "blogger#post",
        "title": title,
        "content": html_content
    }

    is_draft = PUBLISH_STATUS.upper() != "LIVE"
    post = service.posts().insert(blogId=BLOG_ID, body=body, isDraft=is_draft).execute()
    return post


# ==========================================
# MAIN EXECUTION
# ==========================================

def run_agent():
    print("="*80)
    print(" BIOTECH CATALYST INTELLIGENCE AGENT")
    print("="*80)

    # Non-interactive CI-friendly selection
    selected_category_name = os.getenv("CATEGORY_NAME", "Oncology")
    selected_subcat_name = os.getenv("SUBCATEGORY_NAME", "RAS Pathway Targeted Small Molecule Inhibitors")

    if selected_category_name not in PRIMARY_CATEGORIES:
        raise ValueError(f"Invalid CATEGORY_NAME: {selected_category_name}")

    if selected_category_name in SUBCATEGORY_PARENTS:
        subcat_dict = PRIMARY_CATEGORIES[selected_category_name]
        if not selected_subcat_name:
            selected_subcat_name = list(subcat_dict.keys())[0]
        if selected_subcat_name not in subcat_dict:
            raise ValueError(f"Invalid SUBCATEGORY_NAME for {selected_category_name}: {selected_subcat_name}")
        selected_label = f"{selected_category_name} - {selected_subcat_name}"
        queries_to_run = subcat_dict[selected_subcat_name]
    else:
        selected_label = selected_category_name
        queries_to_run = PRIMARY_CATEGORIES[selected_category_name]

    # Execute Pipeline
    news_items = gather_category_data(selected_label, queries_to_run)
    display_retrieved_sources(news_items)

    if not news_items:
        print("[!] Warning: No news items returned from RSS queries. Proceeding to LLM analysis with fallback query mode...")

    try:
        report = analyze_catalysts_with_llm(selected_label, news_items)
        display_report(report)

        # Build and publish blog post
        post_title = f"Biotech Catalyst Report: {selected_label} ({datetime.utcnow().strftime('%Y-%m-%d')})"
        post_html = build_blog_html(report, news_items)
        published = publish_to_blogger(post_title, post_html)
        print(f"[+] Blogger post published: {published.get('url', 'No URL returned')}")
    except Exception as e:
        print(f"\n[!] Error during execution or publishing: {e}")

if __name__ == "__main__":
    run_agent()
