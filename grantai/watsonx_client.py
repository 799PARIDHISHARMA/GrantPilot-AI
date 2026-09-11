"""
watsonx_client.py — IBM watsonx.ai integration for GrantPilot AI.
Uses the watsonx.ai REST API directly with IBM IAM Bearer token auth.
Model: ibm/granite-4-h-small
Credentials are read from environment variables only — never hardcoded.
"""

import os
import json
import urllib.request
import urllib.parse
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL_ID = "ibm/granite-4-h-small"
_GENERATION_API_VERSION = "2023-05-29"

# IAM token endpoint map keyed by region substring in WATSONX_URL
_IAM_URL_MAP = {
    "us-south": "https://iam.cloud.ibm.com/identity/token",
    "eu-de":    "https://iam.eu-de.bluemix.net/identity/token",
    "eu-gb":    "https://iam.eu-gb.bluemix.net/identity/token",
    "au-syd":   "https://iam.au-syd.bluemix.net/identity/token",
    "jp-tok":   "https://iam.jp-tok.bluemix.net/identity/token",
    "ca-tor":   "https://iam.ca-tor.bluemix.net/identity/token",
}


def _get_credentials() -> tuple[str, str, str]:
    """Read and validate credentials from environment. Returns (api_key, project_id, url)."""
    api_key    = (os.environ.get("IBM_API_KEY")        or "").strip()
    project_id = (os.environ.get("WATSONX_PROJECT_ID") or "").strip()
    url        = (os.environ.get("WATSONX_URL")        or "https://us-south.ml.cloud.ibm.com").strip()

    if not api_key:
        raise EnvironmentError("IBM_API_KEY is not set in your .env file.")
    if not project_id:
        raise EnvironmentError("WATSONX_PROJECT_ID is not set in your .env file.")

    # Warn if user has a Service Credential key (ApiKey- prefix) instead of a Platform key
    if api_key.startswith("ApiKey-"):
        raise ValueError(
            "IBM_API_KEY looks like a Service Credential key (starts with 'ApiKey-'). "
            "Please generate an IBM Cloud Platform API key at "
            "https://cloud.ibm.com/iam/apikeys and use that instead."
        )

    return api_key, project_id, url


def _get_iam_token(api_key: str, watsonx_url: str) -> str:
    """Exchange an IBM Cloud Platform API key for a short-lived IAM Bearer token."""
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    for region, endpoint in _IAM_URL_MAP.items():
        if region in watsonx_url:
            iam_url = endpoint
            break

    data = urllib.parse.urlencode({
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key,
        "response_type": "cloud_iam",
    }).encode()

    req = urllib.request.Request(
        iam_url, data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read())
            return body["access_token"]
    except urllib.error.HTTPError as e:
        err = json.loads(e.read().decode())
        code = err.get("errorCode", str(e.code))
        msg  = err.get("errorMessage", "unknown")
        raise RuntimeError(f"IAM token exchange failed [{code}]: {msg}") from e


def _call_generation_api(
    token: str, project_id: str, watsonx_url: str, prompt: str
) -> str:
    """POST to the watsonx.ai /ml/v1/text/generation endpoint and return generated text."""
    endpoint = (
        f"{watsonx_url.rstrip('/')}/ml/v1/text/generation"
        f"?version={_GENERATION_API_VERSION}"
    )

    payload = json.dumps({
        "model_id": MODEL_ID,
        "project_id": project_id,
        "input": prompt,
        "parameters": {
            "max_new_tokens": 1500,
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint, data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
            return body["results"][0]["generated_text"]
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        raise RuntimeError(
            f"watsonx.ai generation failed [HTTP {e.code}]: {err_body[:400]}"
        ) from e


def _generate(prompt: str) -> str:
    """Full pipeline: load creds → get IAM token → call generation API."""
    api_key, project_id, url = _get_credentials()
    token = _get_iam_token(api_key, url)
    return _call_generation_api(token, project_id, url, prompt)


# ── Public functions called by app.py ─────────────────────────────────────────

def analyze_eligibility_and_rank(startup: dict, grants_text: str, language: str = "English") -> str:
    """Ask Granite to analyze startup eligibility and rank matched grants."""
    lang_instruction = (
        f"Respond entirely in {language}." if language != "English"
        else "Respond in English."
    )
    prompt = f"""You are GrantPilot AI, an expert grant advisor for startups. {lang_instruction}

## Startup Profile
- **Name:** {startup.get('name')}
- **Domain:** {startup.get('domain')}
- **Stage:** {startup.get('stage')}
- **Location:** {startup.get('location')}
- **Funding Required:** {startup.get('funding_required')}
- **Description:** {startup.get('description')}

## Retrieved Grants (from knowledge base)
{grants_text}

## Your Task
1. Analyze each grant above against the startup profile.
2. Rank them from most to least suitable (1 = best fit).
3. For each grant provide:
   - **Rank & Grant Name**
   - **Eligibility Status**: one of — Eligible / Partially Eligible / Not Eligible / Needs Verification
   - **Eligibility Match** (High / Medium / Low) with a one-line reason
   - **Requirements Met** — list what the startup already satisfies
   - **Requirements to Verify** — list what still needs confirmation
   - **Key Requirements** the startup must meet
   - **Apply Link**

Be concise, factual, and structured. Only include grants that have at least some relevance.
"""
    return _generate(prompt)


def generate_proposal(startup: dict, best_grant: dict, language: str = "English") -> str:
    """Generate a structured funding proposal draft for the best-matched grant."""
    lang_instruction = (
        f"Write the entire proposal in {language}." if language != "English"
        else "Write in English."
    )
    prompt = f"""You are GrantPilot AI, an expert grant writer for startups. {lang_instruction}

## Startup Profile
- **Name:** {startup.get('name')}
- **Domain:** {startup.get('domain')}
- **Stage:** {startup.get('stage')}
- **Location:** {startup.get('location')}
- **Funding Required:** {startup.get('funding_required')}
- **Description:** {startup.get('description')}

## Target Grant
- **Grant Name:** {best_grant.get('grant_name')}
- **Organization:** {best_grant.get('organization')}
- **Funding Range:** {best_grant.get('funding_min')} - {best_grant.get('funding_max')}
- **Eligibility Criteria:** {best_grant.get('eligibility')}
- **Required Documents:** {best_grant.get('required_documents')}
- **Grant Description:** {best_grant.get('description')}

## Your Task
Write a concise, professional funding proposal draft for the startup to apply to the above grant.
Structure it with these exact section headings:

**1. Executive Summary**
**2. Problem Statement**
**3. Our Solution**
**4. Impact & Scalability**
**5. Funding Ask & Usage**
**6. Why We Qualify**
**7. Call to Action**

Keep it professional, specific, and under 500 words. Do not add any submission instructions.
"""
    return _generate(prompt)
