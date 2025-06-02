# trending_topics_api.py
# Flask API to serve formatted trending topics.

import os
from flask import Flask, jsonify, request
from flask_cors import CORS # For handling Cross-Origin Resource Sharing
from dotenv import load_dotenv
# Load the .env file
load_dotenv()
#
# Import the retrieval function from trend_retrieval_service
# Assuming trend_retrieval_service.py is in the same directory or accessible in PYTHONPATH
try:
    import trend_retrieval_service 
except ImportError:
    print("Error: trend_retrieval_service.py not found. Make sure it's in the same directory or PYTHONPATH.")
    trend_retrieval_service = None 

# LangSmith specific imports (for tracing the API endpoint)
from langsmith import traceable

# Ensure LangSmith environment variables are set if tracing is desired for the API
if os.environ.get("LANGCHAIN_API_KEY") and os.environ.get("LANGCHAIN_API_KEY") != "YOUR_LANGSMITH_API_KEY_MANUAL":
    os.environ["LANGCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_ENDPOINT"] = os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "Trending_Topics_API_Project_V2") # Updated project name
    print(f"LangSmith tracing configured for API project: {os.environ.get('LANGCHAIN_PROJECT')}")
else:
    print("Warning: LangSmith API Key not found for API. API endpoint tracing might be disabled.")
    if "LANGCHAIN_TRACING_V2" not in os.environ : os.environ["LANGCHAIN_TRACING_V2"] = "false"


app = Flask(__name__)
CORS(app) 

MANIFEST_FILEPATH = "topics_manifest.json"

@app.route('/api/trending-topics/india', methods=['GET'])
@traceable(name="APIGetIndiaTrendingTopics") 
def get_india_trending_topics_api():
    """
    API endpoint to get formatted trending topics for India.
    Optionally uses RAG for summaries if 'use_rag' query parameter is true.
    """
    if not trend_retrieval_service or not hasattr(trend_retrieval_service, 'get_formatted_trending_topics'):
        return jsonify({"error": "Trend retrieval service or function not available."}), 500

    use_rag = request.args.get('use_rag', 'true').lower() == 'true'
    
    print(f"API call received for India trending topics. use_rag={use_rag}")

    try:
        # Ensure Langchain components within trend_retrieval_service are initialized
        # This function initializes embeddings, LLM, and the RAG chain using environment variables for Pinecone.
        components_initialized = trend_retrieval_service.init_langchain_components_if_needed()
        if not components_initialized and use_rag:
            print("Warning: Not all Langchain components for RAG could be initialized in retrieval service. RAG might not function as expected.")
            # Proceeding, but RAG might fall back to direct summarization if not fully ready.

        formatted_data = trend_retrieval_service.get_formatted_trending_topics(
            manifest_filepath=MANIFEST_FILEPATH,
            use_rag_summaries=use_rag
        )
        
        if formatted_data and formatted_data.get("trending_topics"):
            return jsonify(formatted_data), 200
        elif formatted_data and "trending_topics_count" in formatted_data and formatted_data["trending_topics_count"] == 0 :
             return jsonify({"message": "No trending topics found in the manifest or processed.", "data": formatted_data}), 200
        else:
            # This case might indicate an issue within get_formatted_trending_topics if it didn't return expected structure
            return jsonify({"error": "Failed to retrieve or format trending topics. Manifest might be empty or an internal error occurred in retrieval service."}), 500
            
    except FileNotFoundError:
        return jsonify({"error": f"Manifest file '{MANIFEST_FILEPATH}' not found. Run data ingestion service first."}), 404
    except Exception as e:
        print(f"API Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred while fetching trending topics.", "details": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask API for Trending Topics...")
    print(f"Data will be read from: {os.path.abspath(MANIFEST_FILEPATH)}")
    print("Make sure 'data_ingestion_service.py' has been run to create the manifest.")
    print("API Endpoint: http://127.0.0.1:5000/api/trending-topics/india")
    print("Optional query parameter: ?use_rag=true (default) or ?use_rag=false")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
