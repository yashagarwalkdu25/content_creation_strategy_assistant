# trend_retrieval_service.py
# Reads processed topic data and generates detailed, formatted trend information using RAG and LLM.

import os
import json
import time
import re
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# For Colab user data (secrets) - for local execution, ensure env vars are set
try:
    from google.colab import userdata
    USE_COLAB_USERDATA = True
except ImportError:
    USE_COLAB_USERDATA = False 

# Pinecone (only client needed for Langchain wrapper if keys are in env)
from pinecone import Pinecone # For type hinting if needed, not for direct init here

# Langchain specific imports
from langchain_openai import ChatOpenAI 
from langchain_pinecone import Pinecone as LangchainPineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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

OPENAI_API_KEY = load_config_value('OPENAI_API_KEY', 'OPENAI_API_KEY', "YOUR_OPENAI_API_KEY_MANUAL")
PINECONE_API_KEY = load_config_value('PINECONE_API_KEY', 'PINECONE_API_KEY', "YOUR_PINECONE_API_KEY_MANUAL") 
PINECONE_ENVIRONMENT = load_config_value('PINECONE_ENVIRONMENT', 'PINECONE_ENVIRONMENT', "YOUR_PINECONE_ENVIRONMENT_MANUAL") 
LANGCHAIN_API_KEY = load_config_value('LANGCHAIN_API_KEY', 'LANGCHAIN_API_KEY', "YOUR_LANGSMITH_API_KEY_MANUAL")
LANGCHAIN_PROJECT = load_config_value('LANGCHAIN_PROJECT', 'LANGCHAIN_PROJECT', "Trending_Topics_Retrieval")


if LANGCHAIN_API_KEY and LANGCHAIN_API_KEY != "YOUR_LANGSMITH_API_KEY_MANUAL":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    if LANGCHAIN_PROJECT: os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    print(f"LangSmith tracing configured for project: {os.environ.get('LANGCHAIN_PROJECT')}")
else:
    print("Warning: LangSmith API Key not found. Tracing will be disabled for retrieval service.")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

if PINECONE_API_KEY and PINECONE_API_KEY != "YOUR_PINECONE_API_KEY_MANUAL":
    if 'PINECONE_API_KEY' not in os.environ: os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
if PINECONE_ENVIRONMENT and PINECONE_ENVIRONMENT != "YOUR_PINECONE_ENVIRONMENT_MANUAL":
    if 'PINECONE_ENVIRONMENT' not in os.environ: os.environ['PINECONE_ENVIRONMENT'] = PINECONE_ENVIRONMENT
else: 
    if 'PINECONE_ENVIRONMENT' not in os.environ : os.environ['PINECONE_ENVIRONMENT'] = 'us-east-1'


PINECONE_INDEX_NAME = "trending-topics-phase1" 
embedding_model_name = 'all-MiniLM-L6-v2' 

lc_embeddings = None
langchain_llm = None
langchain_pinecone_store = None 
rag_chain = None
langsmith_client = None 

try:
    if os.environ.get("LANGCHAIN_TRACING_V2") == "true" and os.environ.get("LANGCHAIN_API_KEY"):
        langsmith_client = LangSmithClient(api_key=os.environ["LANGCHAIN_API_KEY"])
        print("LangSmith Client initialized for retrieval service.")
except Exception as e_ls:
    print(f"Error initializing LangSmith Client in retrieval service: {e_ls}")

# --- Helper Functions ---
@traceable(name="RetrievalService_GetCurrentTimestamp")
def get_current_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# Region mapping from your sample code, needed for constructing fallback URL
REGION_MAPPING_TWITTER = { 
    "wrl": "worldwide", "dz": "algeria", "ar": "argentina", "au": "australia",
    "at": "austria", "bh": "bahrain", "by": "belarus", "be": "belgium",
    "br": "brazil", "ca": "canada", "cl": "chile", "co": "colombia",
    "dk": "denmark", "do": "dominican-republic", "ec": "ecuador", "eg": "egypt",
    "fr": "france", "de": "germany", "gh": "ghana", "gr": "greece",
    "gt": "guatemala", "in": "india", "id": "indonesia", "ie": "ireland",
    "il": "israel", "it": "italy", "jp": "japan", "jo": "jordan",
    "ke": "kenya", "kr": "korea", "kw": "kuwait", "lv": "latvia",
    "lb": "lebanon", "my": "malaysia", "mx": "mexico", "nl": "netherlands",
    "nz": "new-zealand", "ng": "nigeria", "no": "norway", "om": "oman",
    "pk": "pakistan", "pa": "panama", "pe": "peru", "ph": "philippines",
    "pl": "poland", "pt": "portugal", "pr": "puerto-rico", "qa": "qatar",
    "ru": "russia", "sa": "saudi-arabia", "sg": "singapore", "za": "south-africa",
    "es": "spain", "se": "sweden", "ch": "switzerland", "th": "thailand",
    "tr": "turkey", "ua": "ukraine", "us": "united-states", "ae": "united-arab-emirates",
    "gb": "united-kingdom", "ve": "venezuela", "vn": "vietnam"
}

# --- Langchain Components Initialization ---
@traceable(name="RetrievalService_InitLangchainComponents")
def init_langchain_components_if_needed():
    global lc_embeddings, langchain_llm, langchain_pinecone_store, rag_chain
    
    components_ready = True
    if not lc_embeddings:
        try:
            lc_embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
            print(f"Langchain Embeddings loaded: {embedding_model_name}")
        except Exception as e:
            print(f"CRITICAL: Failed to load embeddings model: {e}."); lc_embeddings = None; components_ready = False

    if not langchain_llm:
        if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_MANUAL":
            try:
                langchain_llm = ChatOpenAI(temperature=0.25, model_name="gpt-3.5-turbo-0125", openai_api_key=OPENAI_API_KEY, request_timeout=120)
                print("Langchain LLM initialized.")
            except Exception as e: print(f"Error initializing LLM: {e}"); langchain_llm = None; components_ready = False
        else: print("OpenAI API Key not available for LLM."); langchain_llm = None; components_ready = False
    
    if not rag_chain and lc_embeddings and langchain_llm: 
        # Check if pinecone_index_client (from data_ingestion_service, representing pc.Index()) is available
        # For this service, it might not initialize Pinecone itself but rely on env vars for LangchainPinecone
        # However, direct passing of the index object to LangchainPineconeVectorStore is preferred if possible.
        # For now, from_existing_index relies on env vars PINECONE_API_KEY and PINECONE_ENVIRONMENT.
        try:
            if not langchain_pinecone_store:
                print(f"Attempting to initialize LangchainPineconeVectorStore for index: {PINECONE_INDEX_NAME}")
                if not os.environ.get('PINECONE_API_KEY') or not os.environ.get('PINECONE_ENVIRONMENT'):
                    print("Error: PINECONE_API_KEY or PINECONE_ENVIRONMENT not found in environment for LangchainPinecone.")
                    raise ValueError("Pinecone API key or Environment missing for Langchain integration")
                
                langchain_pinecone_store = LangchainPineconeVectorStore.from_existing_index(
                    index_name=PINECONE_INDEX_NAME, embedding=lc_embeddings, text_key="text"
                )
                print("Langchain Pinecone store initialized using from_existing_index.")

            retriever = langchain_pinecone_store.as_retriever(search_kwargs={"k": 5})
            prompt_template_str = """As an AI assistant, your task is to provide a detailed and comprehensive summary of the trending topic in India based on the provided context.
            Focus on: Key information, why it might be trending, actionable insights for content creators.
            Structure with bullet points. If context is insufficient, state that and avoid hallucination.
            Context: {context}
            Question: {question}
            Detailed India-focused Summary:"""
            QA_PROMPT = PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])
            rag_chain = RetrievalQA.from_chain_type(
                llm=langchain_llm, chain_type="stuff", retriever=retriever,
                chain_type_kwargs={"prompt": QA_PROMPT}, return_source_documents=True
            )
            print("Langchain RAG chain initialized.")
        except Exception as e:
            print(f"Error initializing Langchain Pinecone store or RAG chain: {e}") # This is where the previous error occurred
            rag_chain = None; components_ready = False
    elif rag_chain: pass 
    else: 
        components_ready = False
        missing = [p for p,v in [("Embeddings",lc_embeddings),("LLM",langchain_llm)] if not v]
        print(f"RAG chain prerequisites not met: {', '.join(missing)}")

    return components_ready

# --- Direct LLM Summarization (OpenAI) ---
@traceable(name="RetrievalService_SummarizeDirectly")
def summarize_directly_with_openai(text_to_summarize, api_key, topic_title="this topic"):
    print(f"Directly summarizing for '{topic_title}': '{text_to_summarize[:150]}...'") 
    if not text_to_summarize or len(text_to_summarize) < 50: 
        return f"Content for '{topic_title}' is too short for a detailed summary. Initial info: {text_to_summarize}"
    if not api_key or (USE_COLAB_USERDATA is False and api_key == "YOUR_OPENAI_API_KEY_MANUAL"):
        return "OpenAI API key not configured."
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt_content = f"""Provide a comprehensive and detailed summary for the topic: "{topic_title}". 
    This summary is for a content creator in India. Focus on:
    - Key information and main points of the provided text. If the text is just a topic title (e.g., a Twitter hashtag), explain what the topic likely refers to and why it might be trending in India.
    - Actionable insights or potential content angles for a creator.
    Structure the output clearly, using bullet points or numbered lists for main takeaways.
    Ensure the summary is informative. If the provided text is only a topic title, generate plausible insights based on general knowledge of current events or common themes associated with such topics in India.
    Provided Text: {text_to_summarize}"""
    payload = {
        "model": "gpt-3.5-turbo-0125", 
        "messages": [{"role": "system", "content": "You are an expert content analyst. Generate detailed, factual summaries for Indian content creators."},
                     {"role": "user", "content": prompt_content}],
        "max_tokens": 600, "temperature": 0.3 
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=120) 
        response.raise_for_status() 
        result = response.json()
        if result.get('choices') and result['choices'][0].get('message') and result['choices'][0]['message'].get('content'):
            print("OpenAI direct summary generated."); return result['choices'][0]['message']['content'].strip()
        else: print(f"OpenAI response error: {result}"); return "Could not extract direct summary."
    except Exception as e: print(f"OpenAI API call error: {e}"); return f"OpenAI API error: {e}"


# --- Main Trend Retrieval and Formatting Function ---
@traceable(name="GetFormattedTrendingTopics")
def get_formatted_trending_topics(manifest_filepath="topics_manifest.json", use_rag_summaries=True):
    print("\n--- Retrieving and Formatting Trending Topics ---")
    report_generation_time = get_current_timestamp()

    if not os.path.exists(manifest_filepath):
        print(f"Topics manifest file not found: {manifest_filepath}.")
        return {"report_generated_at": report_generation_time, "trending_topics_count":0, "trending_topics": []}

    try:
        with open(manifest_filepath, 'r', encoding='utf-8') as f:
            topics_from_manifest = json.load(f)
    except Exception as e:
        print(f"Error reading topics manifest: {e}"); return {"report_generated_at": report_generation_time, "trending_topics_count":0, "trending_topics": []}

    if not topics_from_manifest:
        print("No topics found in the manifest."); return {"report_generated_at": report_generation_time, "trending_topics_count":0, "trending_topics": []}

    langchain_ready = init_langchain_components_if_needed()
    if not langchain_ready and use_rag_summaries:
        print("Warning: Langchain components for RAG not fully initialized. RAG summaries might be disabled or fall back to direct summarization.")
        # If RAG chain isn't ready, use_rag_summaries might effectively be false for RAG attempts.

    final_structured_topics = []
    for topic_data in topics_from_manifest:
        topic_title = topic_data.get('title', 'Untitled Topic')
        print(f"\nFormatting topic: {topic_title}")

        structured_topic_output = {
            "Title": topic_title,
            "Info": topic_data.get('initial_info_desc', "No initial description available."),
            "Source": list(set([topic_data.get('source_name', 'Unknown Source')])),
            "Type": topic_data.get('content_type', 'Unknown Type'),
            "Region": topic_data.get('geo_region', 'India'),
            "Metadata": topic_data.get('meta_data', {}),
        }
        if 'collected_at' not in structured_topic_output["Metadata"]: 
             structured_topic_output["Metadata"]['collected_at'] = topic_data.get('meta_data', {}).get('collected_at', "N/A")
        
        structured_topic_output["Metadata"].setdefault("hashtags", [topic_title] if topic_title.startswith("#") else [])
        structured_topic_output["Metadata"].setdefault("tone_of_voice", "Neutral")
        structured_topic_output["Metadata"].setdefault("keywords", [])
        structured_topic_output["Metadata"].setdefault("likes", None); structured_topic_output["Metadata"].setdefault("dislikes", None)
        structured_topic_output["Metadata"].setdefault("comments_count", None); structured_topic_output["Metadata"].setdefault("reposts", None)
        
        # Populate Twitter-specific metadata
        if topic_data.get('source_name') == "Twitter (via www.twitter-trending.com)": 
            meta = topic_data.get('meta_data',{})
            structured_topic_output["Metadata"]["tweet_volume_display"] = meta.get('tweet_volume_display', 'N/A')
            structured_topic_output["Metadata"]["scraper_list_date"] = meta.get('scraper_list_date', 'N/A')
            structured_topic_output["Metadata"]["period"] = meta.get('period', 'N/A')
            structured_topic_output["Metadata"]["platform"] = "Twitter"
            # Corrected key for link_to_trend_list to use 'url' from meta_data if present, 
            # otherwise construct the fallback.
            # The 'url' in meta_data for Twitter trends from new scraper is the URL on twitter-trending.com
            link_url = meta.get('url', f"https://www.twitter-trending.com/{REGION_MAPPING_TWITTER.get(topic_data.get('geo_region','in').lower(), topic_data.get('geo_region','in').lower())}/en")
            structured_topic_output["Metadata"]["link_to_trend_list"] = link_url


        full_text_content = topic_data.get('full_text_content_for_llm', topic_data.get('initial_info_desc', ''))
        
        if topic_data.get('content_type') == "Social Trend" and full_text_content == topic_data.get('initial_info_desc'):
            meta = topic_data.get('meta_data',{})
            full_text_content = f"The trending topic on Twitter in India is '{topic_title}'. It was listed on {meta.get('scraper_list_date','an unspecified date')} for the {meta.get('period','N/A')} period with approximately {meta.get('tweet_volume_display','N/A')} tweets. Please provide insights on why this might be trending and potential content angles."
        elif topic_data.get('content_type') == "Search Trend" and full_text_content == topic_data.get('initial_info_desc'):
             full_text_content = f"The trending Google search in India is '{topic_title}'. Initial info: {topic_data.get('initial_info_desc', '')}. Please provide insights on why this might be trending and potential content angles."

        generated_info = structured_topic_output["Info"] 

        if use_rag_summaries and rag_chain: 
            rag_query = f"Provide a detailed summary about the trending topic in India: '{topic_title}'. Explain its key aspects, why it might be trending, and suggest potential content angles or discussion points for a content creator targeting an Indian audience. Use bullet points for key takeaways if appropriate. Base your answer *only* on the provided context retrieved for this topic."
            try:
                rag_response = rag_chain.invoke({"query": rag_query}) 
                if rag_response and rag_response.get('result') and len(rag_response.get('result').strip()) > 50 : 
                    generated_info = rag_response['result']
                    print(f"  RAG Chain Summary: {generated_info[:200]}...") 
                    if rag_response.get('source_documents'):
                        retrieved_urls = list(set([doc.metadata.get('source_url') for doc in rag_response['source_documents'] if doc.metadata.get('source_url')]))
                        for r_url in retrieved_urls: 
                            if r_url and r_url not in structured_topic_output["Metadata"].get('url', ''):
                                 structured_topic_output["Source"].append(f"Context from: {r_url}")
                else: 
                    print(f"  RAG chain result too short for '{topic_title}'. Trying direct summary.")
                    if full_text_content and langchain_llm: summary = summarize_directly_with_openai(full_text_content, OPENAI_API_KEY, topic_title)
                    if summary and len(summary) > 50: generated_info = summary
            except Exception as e:
                print(f"  Error running RAG chain for '{topic_title}': {e}. Trying direct summary.")
                if full_text_content and langchain_llm: summary = summarize_directly_with_openai(full_text_content, OPENAI_API_KEY, topic_title)
                if summary and len(summary) > 50: generated_info = summary
        elif full_text_content and langchain_llm: 
            print(f"  Using direct OpenAI summarization for '{topic_title}'")
            summary = summarize_directly_with_openai(full_text_content, OPENAI_API_KEY, topic_title)
            if summary and len(summary) > 50: generated_info = summary
        else:
            print(f"  Could not generate detailed summary for '{topic_title}'. Using initial description.")
        
        structured_topic_output["Info"] = generated_info
        structured_topic_output["Source"] = list(set(structured_topic_output["Source"])) 
        final_structured_topics.append(structured_topic_output)
        time.sleep(2) 

    output_data_for_json = {
        "report_generated_at": report_generation_time,
        "trending_topics_count": len(final_structured_topics),
        "trending_topics": final_structured_topics
    }
    print("\n\n--- FINAL FORMATTED TRENDING TOPICS (INDIA FOCUS) ---")
    return output_data_for_json


# --- LangSmith Evaluation Function (Sample) ---
@traceable(name="RetrievalService_RunLangsmithEvaluation")
def run_langsmith_evaluation():
    global langsmith_client, langchain_llm 
    if not langsmith_client: print("LangSmith client not initialized. Skipping evaluation."); return
    if not langchain_llm: print("Langchain LLM not initialized. Skipping evaluation."); return

    print("\n--- Running LangSmith Evaluation (Sample) ---")
    dataset_name = "Topic Summarization Quality - India v2" 
    dataset_description = "Evaluating summaries for trending Indian topics, focusing on detail and relevance."
    sample_inputs = [
        {"topic_title": "#आत्मनिर्भरभारत", "text_content": "Atmanirbhar Bharat Abhiyan translates to 'self-reliant India' or 'self-sufficient India'. It is a policy formulated by Prime Minister of India Narendra Modi for making India 'a bigger and more important part of the global economy', pursuing policies that are efficient, competitive and resilient, and being self-sustaining and self-generating."},
        {"topic_title": "Chandrayaan-3 Mission Update", "text_content": "ISRO provides new updates on the Chandrayaan-3 lunar mission. The Pragyan rover continues its exploration of the lunar south pole, sending back valuable data. Scientists are analyzing soil samples and thermal readings. The mission aims to demonstrate end-to-end landing and roving capabilities."},
    ]
    sample_outputs = [
        {"expected_summary": "Atmanirbhar Bharat, meaning self-reliant India, is a key policy by PM Modi aimed at making India a significant, efficient, competitive, and resilient part of the global economy through self-sustaining policies. Key focus areas include local manufacturing and reducing import dependence."},
        {"expected_summary": "ISRO's Chandrayaan-3 mission is actively exploring the lunar south pole with its Pragyan rover. The mission focuses on demonstrating landing/roving capabilities and is sending back crucial data from soil samples and thermal readings for analysis by scientists."}
    ]
    try:
        try: dataset = langsmith_client.read_dataset(dataset_name=dataset_name)
        except Exception: 
            print(f"Creating LangSmith dataset: '{dataset_name}'"); dataset = langsmith_client.create_dataset(dataset_name=dataset_name, description=dataset_description)
            for i, inp in enumerate(sample_inputs):
                langsmith_client.create_example(inputs={"topic_title":inp["topic_title"], "text_to_summarize":inp["text_content"]},
                                                outputs=sample_outputs[i] if i < len(sample_outputs) else None, dataset_id=dataset.id)
            print(f"Added {len(sample_inputs)} examples to dataset '{dataset_name}'.")
        
        @traceable(name="EvaluatedSummarizer_RetrievalService")
        def summarizer_for_evaluation(inputs: dict):
            return summarize_directly_with_openai(inputs["text_to_summarize"], OPENAI_API_KEY, inputs["topic_title"])

        print(f"Running summarizer over dataset '{dataset_name}' for review in LangSmith...")
        for example in langsmith_client.list_examples(dataset_name=dataset_name):
            print(f"Evaluating example for: {example.inputs['topic_title']}")
            try:
                prediction = summarizer_for_evaluation(example.inputs)
                print(f"  Prediction: {prediction[:100]}...")
            except Exception as e_run_eval: print(f"  Error running eval example: {e_run_eval}")
        print("Evaluation run (traced calls) complete. Check LangSmith project.")
    except Exception as e_eval_setup: print(f"LangSmith evaluation error: {e_eval_setup}")


if __name__ == '__main__':
    print("This script provides the 'get_formatted_trending_topics' function and 'run_langsmith_evaluation'.")
    print("It assumes 'topics_manifest.json' is created by 'data_ingestion_service.py'.")

    # Example: How to use the retrieval function (assuming manifest exists)
    if os.path.exists("topics_manifest.json"):
        # Ensure Langchain components are initialized once before calling
        # This can be called at the start of this script or before the first call to get_formatted_trending_topics
        init_langchain_components_if_needed() # Call it here to ensure components are ready
        
        formatted_data = get_formatted_trending_topics(use_rag_summaries=True)
        if formatted_data and formatted_data.get("trending_topics"):
            print(f"\nSuccessfully retrieved and formatted {len(formatted_data['trending_topics'])} topics.")
            # To print the full JSON output:
            print("\nFull JSON Output:\n", json.dumps(formatted_data, indent=2, ensure_ascii=False))
        else:
            print("No topics were formatted by the retrieval function.")
    else:
        print("Run data_ingestion_service.py first to create topics_manifest.json")

    # Example: How to run evaluation
    if langsmith_client and OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_MANUAL":
        if not langchain_llm: init_langchain_components_if_needed() # Ensure LLM is ready
        if langchain_llm: run_langsmith_evaluation()
    else:
        print("\nSkipping LangSmith evaluation due to missing LangSmith client or OpenAI API key.")