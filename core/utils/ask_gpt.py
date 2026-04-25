import os
import json
from threading import Lock
import json_repair
from openai import OpenAI
from core.utils.config_utils import load_key
from rich import print as rprint
from core.utils.decorator import except_handler

# ------------
# cache gpt response
# ------------

LOCK = Lock()
GPT_LOG_FOLDER = 'output/gpt_log'

def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default"):
    with LOCK:
        logs = []
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append({"model": model, "prompt": prompt, "resp_content": resp_content, "resp_type": resp_type, "resp": resp, "message": message})
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)

def _load_cache(prompt, resp_type, log_title):
    with LOCK:
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type:
                        return item["resp"]
        return False

# ------------
# ask gpt once
# ------------

@except_handler("GPT request failed", retry=2, delay=5)
def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default"):
    if not load_key("api.key"):
        raise ValueError("API key is not set")
    # check cache
    cached = _load_cache(prompt, resp_type, log_title)
    if cached:
        rprint("use cache response")
        return cached

    # Build model list: primary model + fallback models
    primary_model = load_key("api.model")
    try:
        fallback_models = load_key("api.fallback_models") or []
    except KeyError:
        fallback_models = []
    models_to_try = [primary_model] + [m for m in fallback_models if m != primary_model]

    base_url = load_key("api.base_url")
    if 'ark' in base_url:
        base_url = "https://ark.cn-beijing.volces.com/api/v3" # huoshan base url
    elif 'v1' not in base_url:
        base_url = base_url.strip('/') + '/v1'
    client = OpenAI(api_key=load_key("api.key"), base_url=base_url)
    response_format = {"type": "json_object"} if resp_type == "json" and load_key("api.llm_support_json") else None

    messages = [{"role": "user", "content": prompt}]

    last_error = None
    for model in models_to_try:
        try:
            params = dict(
                model=model,
                messages=messages,
                response_format=response_format,
                timeout=300
            )
            resp_raw = client.chat.completions.create(**params)

            # Guard against None response content (message or content can be None)
            # This often happens when Gemini's safety filter blocks the content
            if (not resp_raw.choices 
                or resp_raw.choices[0].message is None 
                or resp_raw.choices[0].message.content is None):
                finish_reason = resp_raw.choices[0].finish_reason if resp_raw.choices else "no_choices"
                rprint(f"[red]⚠️ Empty response from {model}, finish_reason={finish_reason}[/red]")
                raise ValueError(f"Empty response from model {model} (finish_reason={finish_reason})")

            # process and return full result
            resp_content = resp_raw.choices[0].message.content
            if resp_type == "json":
                resp = json_repair.loads(resp_content)
            else:
                resp = resp_content
            
            # check if the response format is valid
            if valid_def:
                valid_resp = valid_def(resp)
                if valid_resp['status'] != 'success':
                    _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=valid_resp['message'])
                    raise ValueError(f"❎ API response error: {valid_resp['message']}")

            _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title)
            return resp
        except Exception as e:
            last_error = e
            error_str = str(e)
            # Try fallback model on 404, 429, or empty response
            is_fallback_worthy = ('404' in error_str or '429' in error_str or 'Empty response' in error_str)
            if is_fallback_worthy and model != models_to_try[-1]:
                rprint(f"[yellow]⚠️ Model '{model}' failed ({error_str[:80]}...), trying next fallback model...[/yellow]")
                continue
            else:
                raise  # Let except_handler handle retry with backoff

    raise last_error  # All models failed


if __name__ == '__main__':
    from rich import print as rprint
    
    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
