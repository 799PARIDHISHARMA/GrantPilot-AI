# 🚀 GrantPilot AI — AI Grant & Funding Finder for Startups

An AI-powered MVP built with **Streamlit** and **IBM watsonx.ai Granite 4 H Small** that helps startups discover, evaluate, and apply for grants.

---

## Features

| Feature | Description |
|---|---|
| **RAG Grant Search** | TF-IDF cosine similarity retrieval over `grants.csv` knowledge base |
| **AI Eligibility Analysis** | Granite 4 ranks and scores each grant against your startup profile |
| **Proposal Generator** | Auto-generates a structured funding proposal draft for the best-matched grant |
| **Download Proposal** | Export the AI-generated proposal as a `.txt` file |

---

## Architecture

```
User Input (Streamlit form)
        │
        ▼
grants_rag.py  ──  TF-IDF retrieval from grants.csv  ──► Top-K grants
        │
        ▼
watsonx_client.py  ──  IBM watsonx.ai Granite 4 H Small
        ├──► Eligibility Analysis & Ranking
        └──► Funding Proposal Draft
        │
        ▼
Streamlit UI  ──  Results, ranked grants, downloadable proposal
```

---

## Quick Start

### 1. Clone & Install

```bash
pip install -r requirements.txt
```

### 2. Set Credentials

Copy `.env.example` to `.env` and fill in your IBM watsonx.ai credentials:

```bash
cp .env.example .env
```

```env
IBM_API_KEY=your_ibm_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

> **Never commit `.env` to version control.**

### 3. Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
grantai/
├── app.py               # Main Streamlit application
├── grants_rag.py        # RAG retrieval module (TF-IDF over grants.csv)
├── watsonx_client.py    # IBM watsonx.ai Granite 4 integration
├── grants.csv           # Grant knowledge base
├── requirements.txt     # Python dependencies
├── .env.example         # Credential template
└── README.md
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `IBM_API_KEY` | ✅ Yes | — | IBM Cloud API key |
| `WATSONX_PROJECT_ID` | ✅ Yes | — | watsonx.ai project ID |
| `WATSONX_URL` | ⬜ No | `https://us-south.ml.cloud.ibm.com` | watsonx.ai endpoint |

---

## Adding More Grants

Simply add rows to `grants.csv` following the existing schema:

```
grant_name, organization, domain, location, startup_stage,
funding_min, funding_max, deadline, eligibility,
required_documents, description, application_url
```

The RAG index is rebuilt on every search — no retraining needed.

---

## Model

- **Model ID:** `ibm/granite-4-h-small`
- **Provider:** IBM watsonx.ai
- **Tasks:** Eligibility analysis, grant ranking, proposal generation
