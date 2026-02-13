# ai_local.py
import ollama

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# RECOMMENDED MODEL: qwen2.5:7b (Best balance of speed, accuracy, and JSON quality)
MODEL_NAME = "qwen2.5:7b"

# ALTERNATIVE MODELS (uncomment to use):
# MODEL_NAME = "qwen2.5:14b"  # High-end systems (16GB+ RAM) - Best quality
# MODEL_NAME = "qwen2.5:3b"   # Low-end systems (4-6GB RAM) - Faster
# MODEL_NAME = "mistral:7b"   # Alternative 7B model - Good all-around
# MODEL_NAME = "qwen2.5:1.5b" # Testing only - Not recommended for production

# To install the recommended model, run:
# ollama pull qwen2.5:7b

# ============================================================================
# MODEL PERFORMANCE GUIDE
# ============================================================================
# qwen2.5:7b  → Best for NovaLens (Recommended)
#   - RAM: 8GB minimum
#   - JSON Quality: Excellent
#   - Speed: Fast
#   - Data Analysis: Very Good
#
# qwen2.5:14b → Best for accuracy (High-end systems)
#   - RAM: 16GB minimum
#   - JSON Quality: Excellent
#   - Speed: Moderate
#   - Data Analysis: Excellent
#
# qwen2.5:3b  → Best for speed (Budget systems)
#   - RAM: 4GB minimum
#   - JSON Quality: Good
#   - Speed: Very Fast
#   - Data Analysis: Good
# ============================================================================


def ask_ai(messages, model=None):
    """
    Send messages to local Ollama AI and get response.
    
    Args:
        messages: list of dicts with 'role' and 'content'
                  Example: [{"role": "system", "content": "..."}, 
                           {"role": "user", "content": "..."}]
        model: optional model name override
    
    Returns:
        str: AI response text
    """
    try:
        response = ollama.chat(
            model=model or MODEL_NAME,
            messages=messages,
            options={
                # Temperature: 0.0 = focused/deterministic, 1.0 = creative/random
                "temperature": 0.3,  # Lower for consistent JSON output
                
                # Max tokens to generate (adjust based on needs)
                "num_predict": 3000,  # Enough for detailed responses
                
                # Context window size
                "num_ctx": 4096,  # Larger context for complex tasks
            }
        )
        return response["message"]["content"]
    
    except ollama.ResponseError as e:
        return f"[AI Response Error] {e}"
    
    except ConnectionError:
        return "[AI Connection Error] Ollama is not running. Please start Ollama with 'ollama serve'."
    
    except Exception as e:
        return f"[AI Error] {e}"


def check_ollama_status():
    """
    Check if Ollama is running and which models are available.
    
    Returns:
        dict: Status information
    """
    try:
        models = ollama.list()
        available_models = [m['name'] for m in models.get('models', [])]
        
        return {
            "status": "running",
            "available_models": available_models,
            "model_in_use": MODEL_NAME,
            "model_available": MODEL_NAME in available_models,
            "recommendation": "qwen2.5:7b" if MODEL_NAME != "qwen2.5:7b" else None
        }
    
    except Exception as e:
        return {
            "status": "not_running",
            "error": str(e),
            "available_models": [],
            "model_in_use": MODEL_NAME,
            "model_available": False,
            "recommendation": "Start Ollama with 'ollama serve' then run 'ollama pull qwen2.5:7b'"
        }


def get_ai_summary(data_description, max_tokens=500):
    """
    Get a quick AI summary of something.
    
    Args:
        data_description: str describing what to summarize
        max_tokens: max response length (approximate)
    
    Returns:
        str: AI summary
    """
    messages = [
        {
            "role": "system", 
            "content": f"You are a helpful data analyst. Provide concise summaries in under {max_tokens} words."
        },
        {
            "role": "user", 
            "content": data_description
        }
    ]
    
    return ask_ai(messages)


def test_model_connection():
    """
    Test if the AI model is working properly.
    
    Returns:
        dict: Test results
    """
    try:
        # Simple test
        test_messages = [
            {"role": "system", "content": "Respond with only the word 'OK'"},
            {"role": "user", "content": "Test"}
        ]
        
        response = ask_ai(test_messages)
        
        # JSON test
        json_messages = [
            {"role": "system", "content": "Return valid JSON only"},
            {"role": "user", "content": 'Generate JSON: {"status": "working", "number": 42}'}
        ]
        
        json_response = ask_ai(json_messages)
        
        return {
            "basic_test": "OK" in response.upper(),
            "json_test": "{" in json_response and "}" in json_response,
            "model": MODEL_NAME,
            "status": "All tests passed" if ("OK" in response.upper() and "{" in json_response) else "Some tests failed"
        }
    
    except Exception as e:
        return {
            "basic_test": False,
            "json_test": False,
            "model": MODEL_NAME,
            "status": f"Error: {e}"
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NovaLens AI - Model Configuration Test")
    print("=" * 70)
    print()
    
    # Check status
    print("1. Checking Ollama status...")
    status = check_ollama_status()
    print(f"   Status: {status['status']}")
    print(f"   Model in use: {status['model_in_use']}")
    print(f"   Model available: {status['model_available']}")
    
    if status['recommendation']:
        print(f"   ⚠️  Recommendation: {status['recommendation']}")
    
    print()
    
    # Test connection
    if status['status'] == 'running':
        print("2. Testing model connection...")
        test_results = test_model_connection()
        print(f"   Basic test: {'✅ Pass' if test_results['basic_test'] else '❌ Fail'}")
        print(f"   JSON test: {'✅ Pass' if test_results['json_test'] else '❌ Fail'}")
        print(f"   Overall: {test_results['status']}")
    else:
        print("2. ❌ Ollama not running - skipping tests")
        print(f"   Error: {status.get('error', 'Unknown')}")
        print()
        print("To fix:")
        print("  1. Open a terminal")
        print("  2. Run: ollama serve")
        print("  3. Run: ollama pull qwen2.5:7b")
    
    print()
    print("=" * 70)

