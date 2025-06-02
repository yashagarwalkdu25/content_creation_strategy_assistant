# --- Phase 2: Content Strategy & User Feedback (Single Chat API Version) ---
import json
import os
import re
from datetime import datetime, timezone
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Added by user
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv # Added by user

# Load the .env file
load_dotenv()

# Attempt to import Langchain components
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_pinecone import PineconeVectorStore
    LANGCHAIN_AVAILABLE = True
    print("Successfully imported Langchain components.")
except ImportError as e:
    print(f"WARNING: Langchain components could not be imported: {e}.")
    print("Please ensure langchain, langchain-openai, langchain-community, langchain-pinecone, pinecone-client, openai, and sentence-transformers are installed.")
    LANGCHAIN_AVAILABLE = False

# LangSmith Tracing
# Ensure environment variables are set for LangSmith to work:
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"  # Replace with your actual key
# os.environ["LANGCHAIN_PROJECT"] = "ContentStrategy-ChatAPI-v4" # Optional: project name

try:
    from langsmith import traceable
except ImportError:
    print("LangSmith SDK not found. Tracing will be disabled. pip install langsmith")
    def traceable(func_or_name=None, *, name=None, run_type=None, project_name=None, new_session=False, **kwargs): # type: ignore
        if callable(func_or_name):
            return func_or_name
        return lambda func: func # No-op decorator

# Attempt to import for Colab userdata (less relevant for API but kept for consistency)
try:
    from google.colab import userdata
    USE_COLAB_USERDATA = True
except ImportError:
    USE_COLAB_USERDATA = False

# --- Global State & Configuration ---
OPENAI_API_KEY = None
PINECONE_API_KEY = None
PINECONE_ENVIRONMENT = None
PINECONE_INDEX_NAME = "trending-topics-phase1"

langchain_llm = None
rag_chain = None
embeddings_model_global = None

user_profile: Dict[str, Any] = {
    "main_niche": None, "secondary_topics": [], "platform": None,
    "content_type": [], "follower_count": None, "post_frequency": None, 
    "tone_style": None, "current_goals": [], "last_updated": None
}
profiling_in_progress: bool = False
current_profiling_question_index: int = 0
profiling_failed_attempts: int = 0
profiling_questions_global: List[Dict[str, str]] = [
    {"key": "main_niche", "question": "Welcome! What's the main topic or niche you focus on in your content?"},
    {"key": "secondary_topics", "question": "Great! Do you also cover any secondary topics? (e.g., sports, technology, lifestyle, gaming)"}, 
    {"key": "platform", "question": "Which platform do you use most? (e.g., YouTube, Instagram, TikTok, X, Facebook)"}, 
    {"key": "content_type", "question": "What type of content do you primarily create? (e.g., Long-form Videos, Shorts, Text Posts, Image Carousels, Live Streams)"}, 
    {"key": "follower_count", "question": "Approximately how many followers/subscribers do you have? (e.g., 1k, 50k, 1M+)"}, 
    {"key": "post_frequency", "question": "How often do you typically post new content? (e.g., daily, 3 times a week, once a month)"}, 
    {"key": "tone_style", "question": "What's the tone or style of your content? (e.g., Educational & Serious, Funny & Casual, Inspirational & Uplifting, Tech-focused & Detailed)"}, 
    {"key": "current_goals", "question": "And finally, what are your current goals? (e.g., Audience growth, Better engagement, Monetization, Community building - list any that apply)"} 
]

current_selected_idea_object: Optional[Dict[str, Any]] = None
last_retrieved_ideas: List[Dict[str, Any]] = []
last_generated_content_type: Optional[str] = None
conversation_history: List[Dict[str, str]] = []
MAX_HISTORY_LENGTH = 10

# --- Utility Functions ---
@traceable(name="get_current_timestamp_util")
def get_current_timestamp_util():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

@traceable(name="add_to_conversation_history_util")
def add_to_conversation_history_util(role: str, content: str):
    global conversation_history
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > MAX_HISTORY_LENGTH * 2:
        conversation_history = conversation_history[-(MAX_HISTORY_LENGTH * 2):]

# --- Core Logic Functions (with @traceable) ---
@traceable(name="get_llm_response_service")
async def get_llm_response_service(prompt_text: str, system_message: str = "You are a helpful AI assistant.", is_json_output: bool = False, use_history: bool = True) -> str:
    global langchain_llm, conversation_history
    if not langchain_llm:
        raise HTTPException(status_code=503, detail="LLM service not available at the moment.")

    messages_payload = [{"role": "system", "content": system_message}]
    if use_history:
        history_to_include = conversation_history[-MAX_HISTORY_LENGTH:]
        messages_payload.extend(history_to_include)
    messages_payload.append({"role": "user", "content": prompt_text})
    
    try:
        if hasattr(langchain_llm, 'ainvoke'):
            response_obj = await langchain_llm.ainvoke(messages_payload)
        else:
            response_obj = langchain_llm.invoke(messages_payload)
        response_content = response_obj.content.strip()
        return response_content
    except Exception as e:
        print(f"Error interacting with LLM: {e}")
        error_detail = f"AI processing error: {str(e)}"
        if hasattr(e, 'response') and hasattr(e.response, 'text'): # type: ignore
            error_detail += f" LLM Response: {e.response.text}" # type: ignore
        raise HTTPException(status_code=500, detail=error_detail)


@traceable(name="update_profile_service")
async def update_profile_service(profile_key: str, user_answer: str) -> bool:
    global user_profile
    # --- v4: Even More Explicit Extraction Prompt ---
    extraction_system_prompt = f"""
You are an extremely precise data extraction assistant. Your ONLY task is to extract the specific information for the profile field '{profile_key}' from the user's answer.
User's answer is: "{user_answer}"

**VERY IMPORTANT INSTRUCTIONS for '{profile_key}':**
- If '{profile_key}' is 'main_niche': Extract the core topic.
  Examples: "cooking" -> "cooking"; "tech reviews" -> "tech reviews"; "youtube news" -> "youtube news"; "sports commentary" -> "sports commentary".
- If '{profile_key}' is 'secondary_topics': Extract a comma-separated list. If one item, extract just that.
  Examples: "sports" -> "sports"; "technology" -> "technology"; "sports and technology" -> "sports, technology"; "gaming, lifestyle" -> "gaming, lifestyle".
- If '{profile_key}' is 'platform': Extract the main platform name.
  Examples: "youtube" -> "YouTube"; "Instagram" -> "Instagram"; "tiktok" -> "TikTok".
- If '{profile_key}' is 'content_type': Extract a comma-separated list of content types. If one item, extract just that.
  Examples: "shorts" -> "shorts"; "Long-form Videos and Shorts" -> "Long-form Videos, Shorts"; "text posts, image carousels" -> "text posts, image carousels".
- If '{profile_key}' is 'follower_count': Extract the number, including 'k' or 'M' if present.
  Examples: "10k" -> "10k"; "50000" -> "50000"; "1M" -> "1M".
- If '{profile_key}' is 'post_frequency': Extract the frequency.
  Examples: "daily" -> "daily"; "twice a week" -> "twice a week".
- If '{profile_key}' is 'tone_style': Extract the descriptive phrase for tone.
  Examples: "educational" -> "educational"; "funny and informative" -> "funny and informative".
- If '{profile_key}' is 'current_goals': Extract a comma-separated list of goals. If one item, extract just that.
  Examples: "audience growth" -> "audience growth"; "engagement, monetization" -> "engagement, monetization".

If the user's answer is completely irrelevant to '{profile_key}', or if you are absolutely unsure how to map it to the examples, output the single word 'UNCLEAR'.
Otherwise, ALWAYS try to extract based on the examples. Output ONLY the extracted value(s) or 'UNCLEAR'. No other text.
"""
    # --- End of v4 Extraction Prompt ---

    extracted_value_str = await get_llm_response_service(user_answer, system_message=extraction_system_prompt, use_history=False)
    print(f"DEBUG: For profile_key '{profile_key}', user_answer '{user_answer}', LLM extracted: '{extracted_value_str}'")

    if extracted_value_str.upper() == 'UNCLEAR':
        # For 'tone_style', if LLM is UNCLEAR, directly use user's short answer.
        if profile_key == 'tone_style' and 0 < len(user_answer) < 100:
            print(f"INFO: Using direct user answer for 'tone_style' as LLM was UNCLEAR. Answer: '{user_answer}'")
            user_profile[profile_key] = user_answer.strip()
        else:
            print(f"AI could not extract info for {profile_key} (LLM returned UNCLEAR and no override).")
            return False # Explicitly fail if UNCLEAR and not overridden
    else: # LLM provided a value (not "UNCLEAR")
        if profile_key in ["secondary_topics", "current_goals", "content_type"]:
            # Ensure even single items become a list of one, and handle potential empty strings from split
            items = [item.strip() for item in extracted_value_str.split(',') if item.strip()]
            user_profile[profile_key] = items if items else [] # Store as list, even if empty after processing
        else:
            user_profile[profile_key] = extracted_value_str # Already stripped

    # Final check for success
    current_value = user_profile.get(profile_key)
    if current_value is not None: # Value exists
        if isinstance(current_value, list):
            # For list types, success if the list is populated OR if it's okay for it to be empty
            # (e.g., user might genuinely have no secondary_topics).
            # We consider it a successful *extraction turn* if a value (even empty list) is set.
            # The calling function will decide if an empty list is acceptable for the profile.
            user_profile["last_updated"] = get_current_timestamp_util()
            print(f"Profile updated: {profile_key} = {user_profile[profile_key]}")
            return True
        elif isinstance(current_value, str):
            if current_value: # String is not empty
                user_profile["last_updated"] = get_current_timestamp_util()
                print(f"Profile updated: {profile_key} = {user_profile[profile_key]}")
                return True
    
    # If value is None or an empty string for non-list types after processing, it's a failure for this turn.
    print(f"Profile update deemed unsuccessful for {profile_key} after processing. Final Value: '{current_value}'")
    # Ensure the field is reset to its initial state if the update was truly unsuccessful
    # to allow re-asking or reflect no valid value was captured.
    if profile_key in ["secondary_topics", "current_goals", "content_type"]:
        if not current_value: user_profile[profile_key] = [] # Reset to empty list
    elif current_value is None or str(current_value).strip() == "":
        user_profile[profile_key] = None # Reset to None
    return False


@traceable(name="formulate_rag_query_service")
async def formulate_rag_query_service() -> Optional[str]:
    global user_profile
    if not user_profile.get("main_niche"): return None
    prompt_to_llm = f"Profile: {json.dumps(user_profile)}. Formulate concise RAG query (max 10 words). Example: 'trending cooking YouTube India'. Query:"
    rag_query = await get_llm_response_service(prompt_to_llm, system_message="You are a RAG query optimization expert.", use_history=False)
    if "error" in rag_query.lower() or not rag_query.strip():
        query_parts = ["trending", user_profile.get("main_niche"), user_profile.get("platform"), "India"]
        rag_query = " ".join(filter(None, query_parts))
        if not user_profile.get("main_niche"): rag_query = "trending content ideas India"
    print(f"DEBUG: Formulated RAG Query: '{rag_query}'")
    return rag_query

@traceable(name="retrieve_content_ideas_service")
async def retrieve_content_ideas_service(num_ideas: int = 3) -> List[Dict[str, Any]]:
    global rag_chain, user_profile
    if not rag_chain: raise HTTPException(status_code=503, detail="RAG service not available.")
    query_for_retriever = await formulate_rag_query_service()
    if not query_for_retriever: return [] 

    question_for_llm_in_rag = f"RAG context for creator (interest: \"{query_for_retriever}\"). Identify {num_ideas} trending topics. For each: title, brief relevance (Niche={user_profile.get('main_niche')}, Platform={user_profile.get('platform')}), source URL. If none, state that. List format."
    try:
        if hasattr(rag_chain, 'ainvoke'):
            rag_response_obj = await rag_chain.ainvoke({"query": question_for_llm_in_rag})
        else:
            rag_response_obj = rag_chain.invoke({"query": question_for_llm_in_rag})
        if not (isinstance(rag_response_obj, dict) and 'result' in rag_response_obj):
            raise HTTPException(status_code=500, detail="Unexpected RAG response structure.")
        llm_generated_ideas_text = rag_response_obj['result']
        
        retrieved_ideas = []
        potential_ideas = re.split(r'\n\s*(?=\d+\.\s|\*\s|- \s|Idea \d+:|Title:)', llm_generated_ideas_text)
        for i, idea_text_chunk in enumerate(potential_ideas):
            idea_text_chunk = idea_text_chunk.strip()
            if len(idea_text_chunk) < 20: continue
            title_match = re.search(r"^(?:\d+\.\s*|[\*\-]\s*|(?:Idea \d+|Title):\s*)(.*?)(?:\n|$)", idea_text_chunk, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else f"Idea {i+1} from RAG"
            url_match = re.search(r"(?:URL|Source URL):\s*(https?://[^\s]+)", idea_text_chunk, re.IGNORECASE)
            
            explanation = idea_text_chunk 
            if title_match: explanation = idea_text_chunk.replace(title_match.group(0), "", 1).strip()
            if url_match: explanation = explanation.replace(url_match.group(0), "",1).strip()

            retrieved_ideas.append({
                "id": f"rag_idea_{uuid.uuid4()}", "title": title,
                "source_url": url_match.group(1) if url_match else "N/A",
                "relevance_explanation": explanation if explanation else "Details extracted from RAG.", 
                "original_source": "RAG System"
            })
            if len(retrieved_ideas) >= num_ideas: break
        return retrieved_ideas
    except Exception as e:
        print(f"Error in RAG service: {e}"); import traceback; traceback.print_exc()
        return [{"id": "rag_error", "title": "RAG Error", "relevance_explanation": str(e), "original_source": "System", "source_url": "N/A"}]


@traceable(name="generate_content_service")
async def generate_content_service(idea_object: Dict[str, Any], target_element: str, modification_instruction: Optional[str] = None, video_duration_preference: Optional[str] = None) -> str:
    global user_profile
    base_prompt_info = f"""
    You are an expert AI Content Strategist and Generator for a content creator.
    Their Profile: {json.dumps(user_profile)}
    Content Idea Details: {json.dumps(idea_object)}
    """
    if modification_instruction: base_prompt_info += f"\nIMPORTANT User Modification Request: '{modification_instruction}'\n"
    
    strategic_advice_prompt = f"""
    \n\n**Strategic Advice Section (IMPORTANT - Include this with all generated content):**
    Based on the user's profile (Niche: {user_profile.get('main_niche')}, Platform: {user_profile.get('platform')}, Tone: {user_profile.get('tone_style')}, Goals: {', '.join(user_profile.get('current_goals', ['Engagement']))}) and the generated content for '{idea_object.get('title', 'this topic')}', provide concise, actionable strategic advice:
    1.  **Optimal Posting Time:** Suggest an optimal day and time to post this specific content on '{user_profile.get('platform', 'their platform')}' for maximum reach/engagement with their target audience. Provide a brief reason (e.g., "Post this sports analysis on YouTube Sunday evening around 7 PM IST, as viewership often peaks before the work week, especially for sports recaps." or "For Instagram, consider posting this fashion tip around 11 AM - 1 PM or 7 PM - 9 PM when users are most active.").
    2.  **Tone Alignment & Enhancement:** Confirm how the generated content's tone aligns with the user's '{user_profile.get('tone_style', 'chosen')}' style. If applicable, suggest one specific way to further enhance the tone for this particular piece (e.g., "The informative tone is good. Consider adding a quick, energetic intro to match your 'energetic and informative' style for this match analysis." or "For this humorous post, ensure the emojis used amplify the fun, not distract.").
    3.  **Platform-Specific Visual & Engagement Strategy (Tailor to content type - {target_element}):**
        * If the content is for a **video platform (e.g., YouTube long-form, general video script)**:
            * **Thumbnail Idea:** Describe 1-2 compelling thumbnail concepts.
            * **Background/Setup:** Suggest a simple yet effective background or on-screen setup.
            * **Pacing & Visuals:** Briefly comment on pacing and suggest B-roll/graphics.
        * If the content is for **Shorts (e.g., YouTube Shorts, Instagram Reels, TikTok)**:
            * **Hook (First 3 Seconds):** CRITICAL. Suggest a specific, attention-grabbing hook.
            * **Visuals & Text Overlay:** Recommend dynamic visuals, quick cuts, and effective on-screen text.
            * **Trending Audio/Effects (if applicable):** Suggest checking for relevant trending sounds/effects.
            * **Thumbnail/Cover Image:** Suggest an engaging still frame or concept.
        * If the content is a **text post/caption (e.g., Instagram post, Facebook update, LinkedIn article, X/Twitter thread)**:
            * **Compelling First Line:** Ensure the first line is a strong hook.
            * **Accompanying Visual:** Suggest the best type of visual.
            * **Readability & Formatting:** Advise on using short paragraphs, bullets, emojis. For X/Twitter, suggest if it should be a thread.
            * **Relevant Hashtags:** (Mention importance, generated separately).
            * **Links & External Resources (if applicable):** Suggest if including links adds value.
    4.  **Call to Action (CTA) Effectiveness:** Briefly explain how the CTA helps achieve their goals.
    Present this strategic advice clearly, using bullet points under the 'Strategic Advice Section' heading.
    """

    generation_instructions = ""
    if target_element == "titles": generation_instructions = f"Generate 3-5 catchy titles for platform '{user_profile.get('platform')}' for the idea '{idea_object.get('title')}'."
    elif target_element == "script_outline":
        duration_text = f"Video duration: {video_duration_preference}." if video_duration_preference else "Standard length (e.g. 5-10 mins, or under 60s for Shorts)."
        video_type_context = "a Short (under 60 seconds)" if video_duration_preference and "under 60 seconds" in video_duration_preference.lower() else "a standard video"
        generation_instructions = f"Create a detailed script outline for {video_type_context} about '{idea_object.get('title')}' for {user_profile.get('platform')}. {duration_text} Include Hook, Main Segments, Visual Suggestions, optional Engagement Prompts, and CTA."
    elif target_element == "caption": generation_instructions = f"Write an engaging caption (100-250 words) for {user_profile.get('platform')} about '{idea_object.get('title')}'."
    elif target_element == "hashtags": generation_instructions = f"Suggest 7-10 relevant hashtags for a {user_profile.get('platform')} post about '{idea_object.get('title')}'."
    else: raise HTTPException(status_code=400, detail=f"Invalid target element: {target_element}")

    full_prompt = base_prompt_info + "\nTask Specifics:\n" + generation_instructions + strategic_advice_prompt
    system_msg = f"You are an expert AI Content Strategist. ALWAYS include 'Strategic Advice Section' for {target_element}."
    return await get_llm_response_service(full_prompt, system_message=system_msg, use_history=True)

@traceable(name="interpret_user_intent_service")
async def interpret_user_intent_service(user_input: str, current_stage: str) -> Dict[str, Any]:
    global last_generated_content_type, current_selected_idea_object, user_profile
    prompt = f"""
    User Profile: {json.dumps(user_profile)}
    Conversation Stage: {current_stage}
    User Input: "{user_input}"
    Last generated content type: {last_generated_content_type}
    Currently selected idea: {json.dumps(current_selected_idea_object) if current_selected_idea_object else 'None'}

    Determine user intent. Possible intents:
    - "answer_profile_question": (If stage is 'profiling')
    - "request_ideas": (e.g., "give me ideas")
    - "select_idea_by_number": (e.g., "1", "select 2") -> extract number
    - "request_specific_element": (e.g., "script for this", "titles for idea X") -> extract element, maybe topic
    - "provide_feedback_and_modify": (e.g., "make it funnier") -> extract modification
    - "provide_feedback_and_new_element": (e.g., "looks good, now give me titles") -> extract new element
    - "provide_feedback_affirmation": (e.g., "great!")
    - "update_profile_inline": (e.g., "my niche is now cooking") -> extract field and value
    - "request_help": ("help")
    - "request_profile_display": ("my profile")
    - "general_statement_or_question": (Fallback)

    Respond ONLY in JSON. Example for request_specific_element:
    {{"intent": "request_specific_element", "target_element": "script_outline", "topic_or_context": "current idea / user specified topic"}}
    Example for update_profile_inline:
    {{"intent": "update_profile_inline", "field": "niche", "new_value": "cooking"}}
    """
    analysis_str = await get_llm_response_service(prompt, system_message="You are an expert intent analysis AI. Respond only in JSON.", is_json_output=True, use_history=False)
    try:
        analysis = json.loads(analysis_str)
        if "intent" not in analysis:
            return {"intent": "general_statement_or_question", "original_text": user_input, "debug_raw_analysis": analysis_str}
        return analysis
    except json.JSONDecodeError:
        return {"intent": "general_statement_or_question", "original_text": user_input, "error": "JSONDecodeError", "raw_analysis": analysis_str}

# --- FastAPI App Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
    global langchain_llm, rag_chain, embeddings_model_global, profiling_in_progress, current_profiling_question_index, user_profile

    print("FastAPI application startup...")
    if USE_COLAB_USERDATA: 
        OPENAI_API_KEY = userdata.get("OPENAI_API_KEY") 
        PINECONE_API_KEY = userdata.get("PINECONE_API_KEY")
        PINECONE_ENVIRONMENT = userdata.get("PINECONE_ENVIRONMENT")
        PINECONE_INDEX_NAME = userdata.get("PINECONE_INDEX_NAME", "trending-topics-phase1")
    else:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
        PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "trending-topics-phase1")

    if not LANGCHAIN_AVAILABLE: print("CRITICAL: Langchain libraries not available.")
    else:
        if OPENAI_API_KEY:
            try:
                langchain_llm = ChatOpenAI(temperature=0.25, model_name="gpt-3.5-turbo-0125", openai_api_key=OPENAI_API_KEY, request_timeout=120)
                print("SUCCESS: langchain_llm (ChatOpenAI) initialized.")
            except Exception as e: print(f"ERROR initializing ChatOpenAI: {e}")
        else: print("ERROR: OPENAI_API_KEY missing for ChatOpenAI.")

        if langchain_llm and PINECONE_API_KEY and PINECONE_ENVIRONMENT and PINECONE_INDEX_NAME:
            try:
                embeddings_model_global = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
                vector_store = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings_model_global)
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                QA_PROMPT = PromptTemplate(template="Context: {context}\nQuestion: {question}\nHelpful Answer:", input_variables=["context", "question"])
                rag_chain = RetrievalQA.from_chain_type(llm=langchain_llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": QA_PROMPT}, return_source_documents=True)
                print("SUCCESS: RAG chain initialized.")
            except Exception as e: print(f"ERROR initializing RAG chain: {e}"); import traceback; traceback.print_exc()
        else: print("ERROR: Missing components for RAG chain initialization.")
    
    profile_is_incomplete = False
    first_unanswered_idx = 0
    for i, q_item in enumerate(profiling_questions_global):
        key = q_item["key"]
        value = user_profile.get(key)
        if isinstance(value, list): 
            if not value: 
                profile_is_incomplete = True
                first_unanswered_idx = i
                break
        elif value is None or str(value).strip() == "": 
            profile_is_incomplete = True
            first_unanswered_idx = i
            break
            
    if profile_is_incomplete:
        profiling_in_progress = True
        current_profiling_question_index = first_unanswered_idx
        print(f"INFO: Profile is incomplete. Starting profiling at question index {current_profiling_question_index} ({profiling_questions_global[current_profiling_question_index]['key']}).")
    else:
        profiling_in_progress = False
        print("INFO: Profile is already complete based on initial check.")

    yield
    print("FastAPI application shutdown.")

app = FastAPI(title="Content Strategy Chat API", version="2.1.0", lifespan=lifespan) 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

class ChatRequest(BaseModel):
    user_input: str

class ChatResponseData(BaseModel):
    ai_message: str
    current_stage: str 
    profile_complete: bool
    selected_idea: Optional[Dict[str, Any]] = None
    retrieved_ideas: Optional[List[Dict[str, Any]]] = None 
    generated_content_type: Optional[str] = None 

class ChatResponse(BaseModel):
    success: bool
    data: Optional[ChatResponseData] = None
    error_message: Optional[str] = None

@app.options("/chat")
async def chat_options():
    return {"message": "OK"}

@app.post("/chat", response_model=ChatResponse, summary="Main conversational endpoint")
@traceable(name="api_chat_orchestrator")
async def chat_orchestrator(request: ChatRequest):
    global user_profile, profiling_in_progress, current_profiling_question_index, profiling_questions_global
    global current_selected_idea_object, last_retrieved_ideas, last_generated_content_type, conversation_history
    global profiling_failed_attempts

    user_input = request.user_input
    add_to_conversation_history_util("user", user_input)
    
    ai_response_text = ""
    current_stage = "general" 
    response_retrieved_ideas = None
    response_generated_content_type = None

    try:
        if profiling_in_progress:
            current_stage = "profiling"
            if current_profiling_question_index < len(profiling_questions_global):
                question_item = profiling_questions_global[current_profiling_question_index]
                success = await update_profile_service(question_item["key"], user_input)
                if success:
                    profiling_failed_attempts = 0  
                    ai_response_text = f"Got it for {question_item['key']}. "
                    current_profiling_question_index += 1
                    if current_profiling_question_index < len(profiling_questions_global):
                        next_question_item = profiling_questions_global[current_profiling_question_index]
                        ai_response_text += next_question_item["question"]
                    else: 
                        profiling_in_progress = False
                        current_stage = "profile_complete"
                        ai_response_text += "Thanks! Your profile is now complete. What would you like to do next? (e.g., 'get content ideas')"
                        try:
                            ideas = await retrieve_content_ideas_service(num_ideas=1)
                            if ideas and ideas[0].get("id") != "rag_error":
                                current_selected_idea_object = ideas[0]
                                ai_response_text += f"\n\nBased on your profile, here's a trending idea: '{ideas[0]['title']}'. Would you like me to generate a script outline for this?"
                                current_stage = "proactive_suggestion"
                        except Exception as e_rag:
                            print(f"Error during proactive RAG suggestion: {e_rag}") 
                            ai_response_text += "\n(Could not fetch a proactive idea right now)."
                else: 
                    profiling_failed_attempts += 1
                    if profiling_failed_attempts >= 2: 
                        ai_response_text = f"Okay, I'm having a bit of trouble with '{question_item['key']}'. Let's skip this for now. "
                        current_profiling_question_index += 1
                        profiling_failed_attempts = 0 
                        if current_profiling_question_index < len(profiling_questions_global):
                            next_question_item = profiling_questions_global[current_profiling_question_index]
                            ai_response_text += next_question_item["question"]
                        else: 
                            profiling_in_progress = False
                            current_stage = "profile_complete_with_skips"
                            ai_response_text += "We've finished the initial questions. Some parts of your profile might be incomplete. What would you like to do next?"
                    else: 
                        ai_response_text = f"I had trouble understanding your answer for {question_item['key']}. Could you try rephrasing? {question_item['question']}"
            else: 
                profiling_in_progress = False
                current_stage = "profile_complete_unexpected"
                ai_response_text = "Profiling seems complete. What's next?"
        
        else: 
            intent_analysis = await interpret_user_intent_service(user_input, current_stage="general_interaction")
            intent = intent_analysis.get("intent", "general_statement_or_question")
            
            if intent == "general_statement_or_question": 
                if "help" in user_input.lower(): intent = "request_help"
                elif "my profile" in user_input.lower(): intent = "request_profile_display"
                elif "ideas" in user_input.lower() or "find ideas" in user_input.lower() or "suggest content" in user_input.lower(): intent = "request_ideas"
                elif user_input.isdigit() and last_retrieved_ideas: intent = "select_idea_by_number"
                elif (current_selected_idea_object or "script for" in user_input.lower() or "outline for" in user_input.lower()) and \
                     any(kw in user_input.lower() for kw in ["script", "titles", "caption", "hashtags", "outline"]):
                    intent = "request_specific_element"
                elif "update my" in user_input.lower() or "change my" in user_input.lower(): 
                    intent = "update_profile_inline" 

            if intent == "request_help":
                ai_response_text = "You can ask for 'content ideas', 'my profile', or if an idea is selected, ask for 'titles', 'script outline', 'caption', or 'hashtags'. You can also say 'update my niche to [new niche]'."
                current_stage = "general_help"
            
            elif intent == "request_profile_display":
                ai_response_text = f"Current Profile: {json.dumps(user_profile, indent=2)}"
                current_stage = "profile_display"

            elif intent == "update_profile_inline":
                field_to_update = intent_analysis.get("field")
                new_value_for_field = intent_analysis.get("new_value")
                if field_to_update and new_value_for_field and field_to_update in user_profile:
                    original_value = user_profile[field_to_update]
                    update_success = await update_profile_service(field_to_update, str(new_value_for_field))
                    if update_success:
                        ai_response_text = f"Okay, I've updated your {field_to_update} from '{original_value}' to '{user_profile[field_to_update]}'."
                    else:
                        user_profile[field_to_update] = original_value 
                        ai_response_text = f"I tried to update your {field_to_update} to '{new_value_for_field}', but had trouble processing that. Could you try rephrasing the update?"
                else: # LLM intent couldn't parse field/value, try a more direct extraction from user_input
                    # This is a simplified inline update parser, a more robust solution might be needed
                    match = re.search(r"(?:update|change)\s+my\s+(\w+)\s+to\s+(.+)", user_input, re.IGNORECASE)
                    if match:
                        field_to_update_re = match.group(1).lower().replace(" ", "_") # e.g. "main niche" -> "main_niche"
                        new_value_for_field_re = match.group(2).strip()
                        # Find the closest matching key in user_profile
                        actual_field_key = None
                        for key_option in user_profile.keys():
                            if field_to_update_re in key_option.lower().replace("_",""):
                                actual_field_key = key_option
                                break
                        
                        if actual_field_key and new_value_for_field_re:
                            original_value = user_profile[actual_field_key]
                            update_success = await update_profile_service(actual_field_key, new_value_for_field_re)
                            if update_success:
                                ai_response_text = f"Okay, I've updated your {actual_field_key} to '{user_profile[actual_field_key]}'."
                            else:
                                user_profile[actual_field_key] = original_value
                                ai_response_text = f"I tried to update your {actual_field_key} to '{new_value_for_field_re}', but had trouble. Please try again."
                        else:
                            ai_response_text = "I understood you want to update your profile, but couldn't figure out which part. Try 'update my niche to [your new niche]'."
                    else:
                        ai_response_text = "I understood you want to update your profile, but I couldn't figure out which part or what the new value is. Try 'update my niche to [your new niche]' or similar."
                current_stage = "profile_updated_inline"


            elif intent == "request_ideas":
                current_stage = "idea_retrieval"
                ideas = await retrieve_content_ideas_service(num_ideas=3)
                if ideas and ideas[0].get("id") != "rag_error":
                    last_retrieved_ideas = ideas
                    response_retrieved_ideas = ideas 
                    ai_response_text = "Here are some ideas I found:\n"
                    for i, idea_obj in enumerate(ideas):
                        ai_response_text += f"{i+1}. {idea_obj['title']} (ID: {idea_obj['id']})\n"
                    ai_response_text += "\nEnter the number or ID of the idea you're interested in."
                    current_stage = "idea_selection_pending"
                else:
                    ai_response_text = "I couldn't fetch specific ideas right now. Maybe try again or refine your profile?"
            
            elif intent == "select_idea_by_number":
                current_stage = "idea_selection"
                try:
                    idx = int(user_input) -1 
                    if 0 <= idx < len(last_retrieved_ideas):
                        current_selected_idea_object = last_retrieved_ideas[idx]
                        ai_response_text = f"Selected idea: '{current_selected_idea_object['title']}'. What would you like for this? (e.g., 'script outline', 'titles')"
                        current_stage = "element_generation_pending"
                    else:
                        ai_response_text = "Invalid selection. Please choose a number from the list."
                except ValueError: 
                    ai_response_text = "Please enter a valid number to select an idea from the list."
                    current_stage = "idea_selection_pending" 

            elif intent == "request_specific_element":
                current_stage = "element_generation"
                target_el = intent_analysis.get("target_element")
                topic_from_intent = intent_analysis.get("topic_or_context")

                # If no idea is selected, try to extract topic from user input for script generation
                if not current_selected_idea_object:
                    # Check if user specified a topic directly for a script
                    script_topic_match = re.search(r"(?:script|outline)\s*(?:for|on|about)\s+(.+)", user_input, re.IGNORECASE)
                    if script_topic_match:
                        topic = script_topic_match.group(1).strip()
                        current_selected_idea_object = {"title": topic, "relevance_explanation": f"User directly requested script for topic: {topic}"}
                        if not target_el: target_el = "script_outline" # Default to script_outline if topic is for script
                    elif topic_from_intent and topic_from_intent != "current idea": # LLM extracted a topic
                        current_selected_idea_object = {"title": topic_from_intent, "relevance_explanation": f"User requested element for topic: {topic_from_intent}"}
                        if not target_el: # If LLM didn't specify element, try to infer
                            if "script" in user_input.lower() or "outline" in user_input.lower(): target_el = "script_outline"
                            # Add more inferences if needed
                    else:
                        ai_response_text = "Please select an idea first or specify a topic (e.g., 'script for topic X')."
                        target_el = None 
                
                # If an idea is selected (either previously or just now)
                if current_selected_idea_object:
                    if not target_el: # Fallback keyword check if LLM intent parsing was not specific enough or idea was just set
                        if "script" in user_input.lower() or "outline" in user_input.lower() : target_el = "script_outline"
                        elif "titles" in user_input.lower(): target_el = "titles"
                        elif "caption" in user_input.lower(): target_el = "caption"
                        elif "hashtags" in user_input.lower(): target_el = "hashtags"
                
                if target_el and current_selected_idea_object:
                    duration_pref = None
                    duration_match = re.search(r"(\d+)\s*minutes", user_input, re.IGNORECASE)
                    if duration_match: duration_pref = f"{duration_match.group(1)} minutes"
                    elif "short" in user_input.lower() and "script" in user_input.lower(): duration_pref = "under 60 seconds"

                    ai_response_text = f"Generating {target_el} for '{current_selected_idea_object['title']}'...\n"
                    generated_text = await generate_content_service(current_selected_idea_object, target_el, video_duration_preference=duration_pref)
                    ai_response_text += generated_text 
                    last_generated_content_type = target_el
                    response_generated_content_type = target_el
                    current_stage = "awaiting_feedback"
                elif not target_el and current_selected_idea_object: 
                     ai_response_text = "What content element would you like for the selected idea? (titles, script, caption, hashtags)"
                     current_stage = "element_generation_pending"
                elif not target_el and not current_selected_idea_object and not ai_response_text: # If no idea, no target, and no other message set
                    ai_response_text = "I'm not sure what you'd like to do. You can ask for 'ideas' or 'help'."


            
            else: # Fallback for general_statement_or_question or unhandled intents
                current_stage = "general_response"
                fallback_prompt = f"User input: \"{user_input}\". Respond helpfully. Current context: Profile set = {not profiling_in_progress}. Selected idea = {current_selected_idea_object['title'] if current_selected_idea_object else 'None'}."
                ai_response_text = await get_llm_response_service(fallback_prompt, system_message="You are a conversational AI content assistant.", use_history=True)

        add_to_conversation_history_util("assistant", ai_response_text)
        
        return ChatResponse(
            success=True,
            data=ChatResponseData(
                ai_message=ai_response_text,
                current_stage=current_stage,
                profile_complete=not profiling_in_progress,
                selected_idea=current_selected_idea_object,
                retrieved_ideas=response_retrieved_ideas,
                generated_content_type=response_generated_content_type
            )
        )

    except HTTPException as http_exc: 
        add_to_conversation_history_util("system", str(http_exc.detail))
        return ChatResponse(success=False, error_message=str(http_exc.detail))
    except Exception as e:
        print(f"Unhandled error in chat orchestrator: {e}")
        import traceback; traceback.print_exc()
        add_to_conversation_history_util("system", f"An unexpected error occurred: {str(e)}")
        return ChatResponse(success=False, error_message=f"An unexpected server error occurred: {str(e)}")


@app.options("/reset-chat-state")
async def reset_chat_state_options():
    return {"message": "OK"}

@app.post("/reset-chat-state", summary="Resets profile and conversation state for the chat session")
@traceable(name="api_reset_chat_state")
async def reset_chat_state_endpoint(): 
    global user_profile, profiling_in_progress, current_profiling_question_index
    global current_selected_idea_object, last_retrieved_ideas, last_generated_content_type, conversation_history
    global profiling_failed_attempts
    
    user_profile = {q_item["key"]: ([] if q_item["key"] in ["secondary_topics", "current_goals", "content_type"] else None) for q_item in profiling_questions_global}
    user_profile["last_updated"] = None 

    profiling_in_progress = True 
    current_profiling_question_index = 0
    conversation_history = []
    current_selected_idea_object = None
    last_retrieved_ideas = []
    last_generated_content_type = None
    profiling_failed_attempts = 0
    
    first_question = ""
    if profiling_questions_global:
        first_question = profiling_questions_global[0]["question"]
        add_to_conversation_history_util("assistant", first_question)

    return ChatResponse(
        success=True,
        data=ChatResponseData(
            ai_message=f"Chat state reset. Let's start over. {first_question}",
            current_stage="profiling",
            profile_complete=False
        )
    )


if __name__ == "__main__":
    import uvicorn
    print("Starting Content Strategy Chat API server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=5002)
