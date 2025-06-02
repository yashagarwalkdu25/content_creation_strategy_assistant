# data_ingestion_service.py
# Fetches, processes, and stores trending topic data.
# Designed to be run periodically (e.g., cron job).

import os
import json
import time
import re
from datetime import datetime, timezone
import uuid
from collections import Counter
from dotenv import load_dotenv
# Load the .env file
load_dotenv()
# For Colab user data (secrets) - for local execution, ensure env vars are set
try:
    from google.colab import userdata
    USE_COLAB_USERDATA = True
except ImportError:
    USE_COLAB_USERDATA = False 

# For Google Trends (using trendspy)
from trendspy import Trends 

# For Web Scraping
import requests
from bs4 import BeautifulSoup

# For Text Chunking (using Langchain's)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# For Pinecone Vector Store
from pinecone import Pinecone, ServerlessSpec 

# Langchain specific imports (only for embeddings here)
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings 

# LangSmith specific imports
from langsmith import Client as LangSmithClient
from langsmith import traceable

# --- Configuration ---
def load_config_value(colab_key, env_key, default_value):
    if USE_COLAB_USERDATA:
        val = userdata.get(colab_key)
        if val: return val
    val = os.environ.get(env_key)
    if val: return val
    print(f"Warning: Configuration for {colab_key}/{env_key} not found. Using default: {default_value}")
    return default_value

OPENAI_API_KEY = load_config_value('OPENAI_API_KEY', 'OPENAI_API_KEY', "YOUR_OPENAI_API_KEY_MANUAL") # Needed if summarization is ever moved here
PINECONE_API_KEY = load_config_value('PINECONE_API_KEY', 'PINECONE_API_KEY', "YOUR_PINECONE_API_KEY_MANUAL")
PINECONE_ENVIRONMENT = load_config_value('PINECONE_ENVIRONMENT', 'PINECONE_ENVIRONMENT', "YOUR_PINECONE_ENVIRONMENT_MANUAL")
LANGCHAIN_API_KEY = load_config_value('LANGCHAIN_API_KEY', 'LANGCHAIN_API_KEY', "YOUR_LANGSMITH_API_KEY_MANUAL")
LANGCHAIN_PROJECT = load_config_value('LANGCHAIN_PROJECT', 'LANGCHAIN_PROJECT', "Trending_Topics_Ingestion")


if LANGCHAIN_API_KEY and LANGCHAIN_API_KEY != "YOUR_LANGSMITH_API_KEY_MANUAL":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    if LANGCHAIN_PROJECT: os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    print(f"LangSmith tracing configured for project: {os.environ.get('LANGCHAIN_PROJECT')}")
else:
    print("Warning: LangSmith API Key not found. Tracing will be disabled for ingestion service.")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

PINECONE_INDEX_NAME = "trending-topics-phase1" 
embedding_model_name = 'all-MiniLM-L6-v2' 
EMBEDDING_DIMENSION = 384 

pc_client = None 
pinecone_index_client = None 
lc_embeddings = None # Embedding model instance

# --- Helper Functions ---
@traceable(name="IngestService_GetCurrentTimestamp")
def get_current_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") 

@traceable(name="IngestService_CleanText")
def clean_text(text):
    if not text: return ""
    text = re.sub(r'\s+', ' ', text); text = text.replace('â\x80\x99', "'"); return text.strip()

@traceable(name="IngestService_NormalizeTitle")
def normalize_title(title):
    if not title: return ""
    cleaned_title = title.replace('#', '').lower()
    cleaned_title = re.sub(r'[^a-z0-9\s]', '', cleaned_title)
    return " ".join(cleaned_title.split()) 

# --- Pinecone Initialization ---
@traceable(name="IngestService_InitPineconeClient")
def init_pinecone_client_if_needed():
    global pc_client, pinecone_index_client
    if pinecone_index_client: return True
    if not PINECONE_API_KEY or PINECONE_API_KEY == "YOUR_PINECONE_API_KEY_MANUAL": 
        print("Pinecone API Key not set. Cannot initialize Pinecone."); return False
    try:
        print("Initializing Pinecone client...")
        pc_client = Pinecone(api_key=PINECONE_API_KEY)
        if 'PINECONE_API_KEY' not in os.environ and PINECONE_API_KEY: os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
        
        region = PINECONE_ENVIRONMENT if (PINECONE_ENVIRONMENT and PINECONE_ENVIRONMENT != "YOUR_PINECONE_ENVIRONMENT_MANUAL") else 'us-east-1'
        if 'PINECONE_ENVIRONMENT' not in os.environ: os.environ['PINECONE_ENVIRONMENT'] = region
        
        if PINECONE_INDEX_NAME not in [idx.name for idx in pc_client.list_indexes()]:
            print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' in region '{region}'...")
            pc_client.create_index(name=PINECONE_INDEX_NAME, dimension=EMBEDDING_DIMENSION, metric='cosine',
                                   spec=ServerlessSpec(cloud='aws', region=region))
            start_time = time.time()
            while not pc_client.describe_index(PINECONE_INDEX_NAME).status['ready']:
                if time.time() - start_time > 300: print("Timeout waiting for index."); return False
                time.sleep(30); print("Waiting for index...")
            print("Index created and ready.")
        else: print(f"Using existing Pinecone index: '{PINECONE_INDEX_NAME}'")
        pinecone_index_client = pc_client.Index(PINECONE_INDEX_NAME)
        print("Pinecone client and Index object initialized."); return True
    except Exception as e: print(f"Pinecone init error: {e}"); return False

# --- Embedding Model Initialization ---
@traceable(name="IngestService_InitEmbeddingModel")
def init_embedding_model_if_needed():
    global lc_embeddings
    if lc_embeddings: return True
    try:
        lc_embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        print(f"Langchain SentenceTransformerEmbeddings loaded: {embedding_model_name}")
        return True
    except Exception as e:
        print(f"CRITICAL: Failed to load embeddings model: {e}"); lc_embeddings = None; return False

# --- Data Fetching Functions ---
@traceable(name="IngestService_FetchGoogleTrends")
def fetch_google_trends(region='IN', count=20): 
    print(f"Fetching Google Trends for {region} via trendspy...")
    time.sleep(3) 
    try:
        tr = Trends(); regional_trends = tr.trending_now_by_rss(geo=region.upper()) 
        topics = []
        if regional_trends:
            for i, item in enumerate(regional_trends[:count]): 
                news_articles = [{'title':a.title,'source':a.source,'url':a.url} for a in item.news[:3]] if item.news else []
                info = f"Google Trend: '{item.keyword}'. Related: {news_articles[0]['title'] if news_articles else 'N/A'}"
                topics.append({"title": item.keyword, "info_desc": info, "source_name": "Google Trends (trendspy)", 
                               "content_type": "Search Trend", "geo_region": region, 
                               "meta_data": {"rank":i+1, "query":item.keyword, "collected_at":get_current_timestamp(), "related_news":news_articles}})
        print(f"Fetched {len(topics)} Google Trends for {region}."); return topics
    except Exception as e: print(f"Google Trends (trendspy) error for {region}: {e}"); return []

@traceable(name="IngestService_FetchNewsTOI")
def fetch_news_toi(count=20): 
    print("Fetching Times of India news...")
    url = "https://timesofindia.indiatimes.com/etimes/trending"
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
    topics = []
    try:
        r = requests.get(url, headers=headers, timeout=30); r.raise_for_status()
        soup = BeautifulSoup(r.content, 'html.parser')
        links = soup.select('div.listing4 a[href*="/articleshow/"], div.top-newslist a[href*="/articleshow/"]') # More general selectors
        added_urls = set()
        for link in links[:count*2]: # Fetch more to filter
            if len(topics) >= count: break
            href = link['href']
            if not href.startswith('http'): href = "https://timesofindia.indiatimes.com" + href
            if href in added_urls: continue
            title = clean_text(link.get('title', link.get_text()))
            if title and len(title) > 15:
                topics.append({"title":title, "info_desc":title, "source_name":"Times of India (ETimes Trending)",
                               "content_type":"Article", "geo_region":"India", "meta_data":{"url":href, "collected_at":get_current_timestamp()}})
                added_urls.add(href)
        print(f"Fetched {len(topics)} TOI articles."); return topics
    except Exception as e: print(f"TOI scraper error: {e}"); return []

REGION_MAPPING_TWITTER = { "in": "india", "us": "united-states", "gb": "united-kingdom" } 

@traceable(name="IngestService_GetTwitterTrendsFromSite")
def get_twitter_trends_from_site(region_code):
    region_name_path = REGION_MAPPING_TWITTER.get(region_code.lower(), region_code.lower())
    url = f"https://www.twitter-trending.com/{region_name_path}/en"
    print(f"🔍 Scraping Twitter trends from {url}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
    scraped_trends = []
    try:
        r = requests.get(url, headers=headers, timeout=25); r.raise_for_status()
        soup = BeautifulSoup(r.content, 'html.parser')
        def process_section(section_id, period_name):
            data = []
            sec_div = soup.find('div', id=section_id)
            if sec_div:
                for item_div in sec_div.find_all('div', class_='one_cikan88'):
                    a = item_div.find('a')
                    if a and a.has_attr('href'):
                        title = clean_text(a.get_text(strip=True))
                        s_url = a['href']
                        now_utc = datetime.now(timezone.utc)
                        date_str = now_utc.strftime("%d/%m/%Y" if period_name == "Daily" else "%B %Y")
                        if title: data.append({'title':title, 'url_on_scraper_site':f"https://www.twitter-trending.com{s_url}" if s_url.startswith('/') else s_url,
                                               'scraper_list_date':date_str, 'period':period_name})
            return data
        raw_trends = process_section('gun_one_c', 'Daily') + process_section('hafta_one_c', 'Weekly') + process_section('ay_one_c', 'Monthly')
        processed_titles = set()
        for trend in raw_trends:
            norm_t = normalize_title(trend['title'])
            if norm_t not in processed_titles: scraped_trends.append(trend); processed_titles.add(norm_t)
        print(f"Parsed {len(scraped_trends)} unique Twitter trends from {url}."); return scraped_trends
    except Exception as e: print(f"Error scraping twitter-trending.com for {region_name_path}: {e}"); return []

@traceable(name="IngestService_FetchTwitterTrendsNewScraper")
def fetch_twitter_trends_new_scraper(region='IN', count=20):
    raw_scraped = get_twitter_trends_from_site(region_code=region.lower())
    topics = []
    if not raw_scraped: print(f"Twitter scraper found no trends for {region}."); return topics
    for i, trend_data in enumerate(raw_scraped[:count]):
        title = trend_data.get('title')
        if not title: continue
        topics.append({
            "title": title,
            "info_desc": f"{trend_data.get('period','')} Twitter trend in {REGION_MAPPING_TWITTER.get(region.lower(), region)}: {title}. Listed on {trend_data.get('scraper_list_date', 'N/A')}.",
            "source_name": "Twitter (via www.twitter-trending.com)", "content_type": "Social Trend", "geo_region": region,
            "meta_data": {"rank":i+1, "tweet_volume_display":"N/A", "collected_at":get_current_timestamp(), 
                          "scraper_list_date":trend_data.get('scraper_list_date'), "period":trend_data.get('period'),
                          "url":trend_data.get('url_on_scraper_site'), "platform":"Twitter",
                          "link_to_trend_list":f"https://www.twitter-trending.com/{REGION_MAPPING_TWITTER.get(region.lower(), region.lower())}/en",
                          "hashtags": [title] if title.startswith('#') else []}
        })
    print(f"Formatted {len(topics)} Twitter trends for {region}."); return topics

# --- Scoring and Content Processing ---
@traceable(name="IngestService_ScoreTopic")
def score_topic(topic, source_occurrence_bonus=0): 
    score = 0
    source_scores = {"Google Trends (trendspy)": 9, "Times of India (ETimes Trending)": 10, "Twitter (via www.twitter-trending.com)": 8}
    type_scores = {"Article": 5, "Social Trend": 4, "Search Trend": 4} 
    score += source_scores.get(topic.get('source_name'), 0) + type_scores.get(topic.get('content_type'), 0)
    collected_at_str = topic.get('meta_data', {}).get('collected_at')
    if collected_at_str:
        try:
            collected_dt = datetime.strptime(collected_at_str, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - collected_dt).total_seconds() / 3600
            if age_hours < 1: score += 7; 
            elif age_hours < 3: score += 5; 
            elif age_hours < 12: score += 3 
        except ValueError: pass 
    if topic.get('meta_data', {}).get('period') == 'Daily': score += 3
    elif topic.get('meta_data', {}).get('period') == 'Weekly': score += 1
    rank = topic.get('meta_data', {}).get('rank')
    if isinstance(rank, int): score += max(0, 15 - rank) 
    score += source_occurrence_bonus; return score

@traceable(name="IngestService_PickTopTopics")
def pick_top_topics(all_topics, topic_source_counts, min_score=5, top_n=30): 
    if not all_topics: return []
    scored_topics = []
    for topic in all_topics:
        bonus = (topic_source_counts.get(normalize_title(topic.get('title','')), 1) -1) * 8
        scored_topics.append({'score': score_topic(topic, source_occurrence_bonus=bonus), **topic})
    filtered = [t for t in scored_topics if t['score'] >= min_score] 
    return sorted(filtered, key=lambda x: (x['score'], -x.get('meta_data',{}).get('rank', float('-inf'))), reverse=True)[:top_n]

@traceable(name="IngestService_FetchContentFromURL")
def fetch_content_from_url(url): # Same as before
    print(f"Fetching content from URL: {url}...")
    if not url or not (url.startswith('http://') or url.startswith('https://')):
        print(f"Invalid or missing URL: {url}"); return None
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=25) 
        response.raise_for_status()
        print(f"Successfully fetched content from {url}.")
        return response.text
    except requests.exceptions.Timeout: print(f"Timeout fetching content from {url}"); return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error fetching content from {url}: {http_err.response.status_code} {http_err.response.reason}")
        return None
    except requests.exceptions.RequestException as e: print(f"Error fetching content from {url}: {e}"); return None
    except Exception as e: print(f"Unexpected error fetching content from {url}: {e}"); return None


@traceable(name="IngestService_ExtractTextFromHTML")
def extract_text_from_html(html_content): # Same as before
    if not html_content: return ""
    soup = BeautifulSoup(html_content, 'html.parser')
    for selector_to_remove in [
        "script", "style", "nav", "footer", "aside", "header", "form", "button", "input", 
        ".sidebar", "#sidebar", ".related-articles", ".comments-section", ".social-share", 
        ".breadcrumb", ".pagination", "figure.image > figcaption", ".ad-slot", "[class*='adbox']",
        "iframe", "noscript", "meta", "link", "[aria-hidden='true']" 
        ]: 
        for tag in soup.select(selector_to_remove): tag.decompose()
    text_parts = []
    main_content_selectors = [
        'div.story_details', 'div.article_content', 
        'div[class*="articlebody" i]', 'div[class*="storybody" i]', 'div[class*="article-content" i]', 
        'div.articletext', 'article[class*="article"]', 'article', 'main', 
        'div[class*="content" i]', 'div[id*="content" i]' 
    ]
    for selector in main_content_selectors:
        main_area = soup.select_one(selector)
        if main_area:
            text = main_area.get_text(separator=' ', strip=True)
            if text and len(text) > 300: 
                text_parts.append(text)
                if sum(len(p) for p in text_parts) > 15000: break 
    if not text_parts or sum(len(p) for p in text_parts) < 300 : 
        body_tag = soup.find('body')
        if body_tag: body_text = body_tag.get_text(separator=' ', strip=True)
        else: body_text = soup.get_text(separator=' ', strip=True)
        if body_text: text_parts.append(body_text)
    full_text = " ".join(text_parts)
    full_text = re.sub(r'(\n\s*)+\n', '\n', full_text) 
    full_text = re.sub(r'\s{3,}', '  ', full_text) 
    return clean_text(full_text)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, add_start_index=True) 

@traceable(name="IngestService_RecursiveChunkText")
def recursive_chunk_text(text): # Same as before
    if not text: return []
    # print(f"Chunking text (length: {len(text)} chars)...")
    chunks = text_splitter.split_text(text)
    # print(f"Text split into {len(chunks)} chunks.")
    return chunks

@traceable(name="IngestService_GetContentEmbeddings")
def get_content_embeddings(text_chunks): # Same as before
    global lc_embeddings
    if not lc_embeddings: print("Embeddings model not loaded."); return []
    if not text_chunks: return []
    # print(f"Generating embeddings for {len(text_chunks)} chunks...")
    try:
        embeddings = lc_embeddings.embed_documents(text_chunks)
        # print("Embeddings generated successfully."); 
        return embeddings 
    except Exception as e: print(f"Error generating embeddings: {e}"); return []

@traceable(name="IngestService_UpsertToPinecone")
def upsert_to_pinecone_client(topic_title, source_url, text_chunks, embeddings): # Same as before
    global pinecone_index_client
    if not pinecone_index_client: print("Pinecone index not available."); return 0
    if len(text_chunks) != len(embeddings): print("Mismatch: chunks & embeddings."); return 0
    vectors_to_upsert = []
    for i, (chunk, emb) in enumerate(zip(text_chunks, embeddings)):
        vectors_to_upsert.append({
            "id": str(uuid.uuid4()), "values": emb,
            "metadata": {"topic_title": str(topic_title)[:500], "source_url": str(source_url)[:500],
                         "text": str(chunk), "original_chunk_index": i}
        })
    if not vectors_to_upsert: print("No vectors to upsert."); return 0
    try:
        # print(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
        upsert_count = 0
        for i_batch in range(0, len(vectors_to_upsert), 100): 
            batch = vectors_to_upsert[i_batch:i_batch + 100]
            upsert_response = pinecone_index_client.upsert(vectors=batch)
            upsert_count += upsert_response.upserted_count 
        print(f"Successfully upserted {upsert_count} vectors for '{topic_title}'."); return upsert_count
    except Exception as e: print(f"Error upserting to Pinecone for '{topic_title}': {e}"); return 0

# --- Main Data Ingestion Function (Cron Job part) ---
@traceable(name="RunPeriodicDataIngestion")
def run_periodic_data_ingestion(manifest_filepath="topics_manifest.json"):
    print("--- Starting Periodic Data Ingestion & Processing ---")
    ingestion_start_time = get_current_timestamp()

    if not init_pinecone_client_if_needed(): 
        print("CRITICAL: Pinecone client failed to initialize. Halting ingestion.")
        return
    if not init_embedding_model_if_needed(): # Initialize embedding model here
        print(f"CRITICAL: Failed to load Langchain embeddings model. Halting ingestion.")
        return

    all_raw_topics = []
    print("\n--- Fetching Raw Topics ---")
    # Fetch more data by increasing counts
    all_raw_topics.extend(fetch_google_trends(region='IN', count=20))
    time.sleep(3) 
    all_raw_topics.extend(fetch_news_toi(count=20))
    time.sleep(3)
    all_raw_topics.extend(fetch_twitter_trends_new_scraper(region='in', count=20))

    if not all_raw_topics: print("\nCRITICAL: No topics collected from any source."); return 
    print(f"\nCollected a total of {len(all_raw_topics)} raw topics.")

    topic_title_counts = Counter(normalize_title(topic.get('title')) for topic in all_raw_topics if topic.get('title'))
    top_scored_topics = pick_top_topics(all_raw_topics, topic_title_counts, min_score=1, top_n=30) # Process more topics
    
    if not top_scored_topics: print("No topics met scoring criteria."); return
    print(f"\n--- Processing Top {len(top_scored_topics)} Scored Topics for Ingestion ---")
    
    processed_topics_for_manifest = []
    for topic_data in top_scored_topics:
        topic_title = topic_data.get('title', 'Untitled Topic')
        print(f"\nIngesting: {topic_title} from {topic_data.get('source_name')}")
        
        manifest_entry = {
            "title": topic_title,
            "initial_info_desc": topic_data.get('info_desc', ''),
            "source_name": topic_data.get('source_name'),
            "content_type": topic_data.get('content_type'),
            "geo_region": topic_data.get('geo_region', 'India'),
            "meta_data": topic_data.get('meta_data', {}),
            "full_text_content_for_llm": topic_data.get('info_desc', '') 
        }
        if 'collected_at' not in manifest_entry["meta_data"]: # Ensure collected_at is present
            manifest_entry["meta_data"]['collected_at'] = get_current_timestamp()


        content_url = topic_data.get('meta_data', {}).get('url')
        # For Google Trends from trendspy, use related_news URL if main URL for the trend itself is missing
        if not content_url and topic_data.get('source_name') == "Google Trends (trendspy)":
            related_news = topic_data.get('meta_data', {}).get('related_news', [])
            if related_news and related_news[0].get('url'):
                content_url = related_news[0]['url']
                manifest_entry['meta_data']['url'] = content_url 
                print(f"  Using related news URL for {topic_title}: {content_url}")
        
        # Fetch and process content only if a URL is available (from TOI or related news from trendspy)
        # Twitter trends from the new scraper usually won't have a direct article URL to fetch here.
        if content_url and topic_data.get('content_type') in ["Article", "Search Trend"]: # Only fetch for these types if URL exists
            raw_html = fetch_content_from_url(content_url)
            if raw_html:
                main_text = extract_text_from_html(raw_html)
                if main_text and len(main_text) > 200:
                    print(f"  Extracted {len(main_text)} chars from {content_url}")
                    manifest_entry['full_text_content_for_llm'] = main_text 
                    text_chunks = recursive_chunk_text(main_text)
                    if text_chunks and lc_embeddings and pinecone_index_client:
                        embeddings = get_content_embeddings(text_chunks) 
                        if embeddings:
                            upsert_to_pinecone_client(topic_title, content_url, text_chunks, embeddings)
                else: print(f"  No significant text from {content_url}")
            else: print(f"  Failed to fetch content from {content_url}")
        elif topic_data.get('content_type') == "Social Trend":
             manifest_entry['full_text_content_for_llm'] = f"Trending topic on {topic_data.get('meta_data',{}).get('platform','Social Media')} in {topic_data.get('geo_region','India')}: '{topic_title}'. Listed on {topic_data.get('meta_data',{}).get('scraper_list_date','N/A')} for the {topic_data.get('meta_data',{}).get('period','N/A')} period. Tweet volume: {topic_data.get('meta_data',{}).get('tweet_volume_display','N/A')}."
        
        processed_topics_for_manifest.append(manifest_entry)

    try:
        with open(manifest_filepath, 'w', encoding='utf-8') as f:
            json.dump(processed_topics_for_manifest, f, indent=2, ensure_ascii=False)
        print(f"\nProcessed topics manifest saved to {manifest_filepath}")
    except Exception as e:
        print(f"Error saving topics manifest: {e}")
    
    print(f"--- Data Ingestion & Processing Finished at {get_current_timestamp()} ---")

if __name__ == '__main__':
    print("This script is intended for data ingestion and processing (e.g., via cron).")
    print("To retrieve and format topics, or run the API, use the other respective scripts.")
    
    # Example of running the ingestion (for testing purposes)
    run_periodic_data_ingestion(manifest_filepath="topics_manifest.json")
