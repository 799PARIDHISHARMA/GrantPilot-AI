"""
app.py — GrantPilot AI: Streamlit MVP
AI-powered grant & funding finder for startups using IBM watsonx.ai Granite 4 H Small.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

import streamlit as st

from grants_rag import retrieve_grants, format_grants_for_prompt, get_kb_summary, score_grant_dimensions
from watsonx_client import analyze_eligibility_and_rank, generate_proposal

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GrantPilot AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #f5f6fa; }
[data-testid="stSidebar"] { background: #1a1f36; }
[data-testid="stSidebar"] * { color: #c9d1e8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }
[data-testid="stSidebar"] .stSlider label { color: #c9d1e8 !important; }
[data-testid="stSidebar"] hr { border-color: #2e3556 !important; }
[data-testid="stSidebar"] code { background: #2e3556 !important; color: #93c5fd !important; }

/* ── App header ── */
.gp-hero {
    background: linear-gradient(135deg, #1a1f36 0%, #2d3561 100%);
    border-radius: 12px;
    padding: 2rem 2.4rem 1.6rem;
    margin-bottom: 1.8rem;
    color: #fff;
}
.gp-hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 .3rem; color: #fff; }
.gp-hero p  { font-size: 1rem; color: #a5b4d4; margin: 0; }
.gp-badge {
    display: inline-block;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.18);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: .75rem;
    color: #c7d5f0;
    margin-top: .7rem;
    margin-right: .4rem;
}

/* ── Section headers ── */
.gp-section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1a1f36;
    border-left: 4px solid #3b5bdb;
    padding-left: .7rem;
    margin: 1.6rem 0 .8rem;
}

/* ── Cards ── */
.gp-card {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e4e8f0;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.gp-card-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1a1f36;
    margin-bottom: .2rem;
}
.gp-card-org {
    font-size: .82rem;
    color: #6b7280;
    margin-bottom: .8rem;
}

/* ── Score badge ── */
.gp-score {
    display: inline-block;
    background: #eef2ff;
    color: #3b5bdb;
    font-weight: 700;
    font-size: .9rem;
    border-radius: 20px;
    padding: 2px 14px;
    margin-bottom: .8rem;
}
.gp-score-high  { background: #dcfce7; color: #16a34a; }
.gp-score-med   { background: #fef9c3; color: #b45309; }
.gp-score-low   { background: #fee2e2; color: #dc2626; }

/* ── Fit pills ── */
.gp-pill {
    display: inline-block;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: .76rem;
    font-weight: 600;
    margin: 2px 3px 2px 0;
}
.pill-yes { background: #dcfce7; color: #15803d; }
.pill-no  { background: #fee2e2; color: #b91c1c; }
.pill-na  { background: #f1f5f9; color: #64748b; }

/* ── Eligibility status ── */
.elig-eligible    { background: #dcfce7; color: #15803d; border-radius:6px; padding:4px 12px; font-weight:700; font-size:.85rem; display:inline-block; }
.elig-partial     { background: #fef9c3; color: #92400e; border-radius:6px; padding:4px 12px; font-weight:700; font-size:.85rem; display:inline-block; }
.elig-not         { background: #fee2e2; color: #b91c1c; border-radius:6px; padding:4px 12px; font-weight:700; font-size:.85rem; display:inline-block; }
.elig-verify      { background: #e0f2fe; color: #0369a1; border-radius:6px; padding:4px 12px; font-weight:700; font-size:.85rem; display:inline-block; }

/* ── Notices / banners ── */
.gp-notice-warn {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: .8rem 1.1rem;
    font-size: .85rem;
    color: #78350f;
    margin: .8rem 0;
}
.gp-notice-info {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: .8rem 1.1rem;
    font-size: .85rem;
    color: #1e40af;
    margin: .8rem 0;
}
.gp-notice-success {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    border-radius: 8px;
    padding: 1rem 1.3rem;
    font-size: .88rem;
    color: #14532d;
    margin: .8rem 0;
}

/* ── KB summary bar ── */
.kb-bar {
    background: #eef2ff;
    border-radius: 8px;
    padding: .55rem 1rem;
    font-size: .8rem;
    color: #3730a3;
    margin-bottom: 1.2rem;
}

/* ── Form card ── */
.form-card {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e4e8f0;
    padding: 1.4rem 1.6rem 1rem;
    margin-bottom: 1.2rem;
}

/* ── Proposal box ── */
.proposal-box {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e4e8f0;
    padding: 1.4rem 1.8rem;
    line-height: 1.75;
    font-size: .92rem;
    color: #1f2937;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## GrantPilot AI")
    st.caption("Grant & Funding Finder for Startups")
    st.markdown("---")

    # Credential status
    api_key_set    = bool((os.environ.get("IBM_API_KEY")        or "").strip())
    project_id_set = bool((os.environ.get("WATSONX_PROJECT_ID") or "").strip())
    url_set        = bool((os.environ.get("WATSONX_URL")        or "").strip())

    st.markdown("**IBM watsonx.ai**")
    st.markdown(f"{'✅' if api_key_set    else '❌'} IBM_API_KEY {'set' if api_key_set    else '**not set**'}")
    st.markdown(f"{'✅' if project_id_set else '❌'} PROJECT_ID  {'set' if project_id_set else '**not set**'}")
    st.markdown(f"{'✅' if url_set        else '⬜'} WATSONX_URL {'set' if url_set        else 'using default'}")

    if not api_key_set or not project_id_set:
        st.warning("Set IBM_API_KEY and WATSONX_PROJECT_ID in your .env to enable AI features.")

    st.markdown("---")
    st.markdown("**Model**")
    st.code("ibm/granite-4-h-small", language=None)
    st.markdown("**Knowledge Base**")
    st.code("grants.csv  (demo dataset)", language=None)

    st.markdown("---")
    top_k = st.slider("Grants to retrieve", min_value=1, max_value=10, value=5)

    st.markdown("**Response Language**")
    language = st.selectbox(
        "Language",
        options=["English", "Hindi"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("GrantPilot AI does not submit applications automatically.")
    st.caption("AI outputs require human review before use.")

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="gp-hero">
  <h1>🚀 GrantPilot AI</h1>
  <p>AI-powered grant &amp; funding finder for startups — powered by IBM watsonx.ai Granite 4</p>
  <span class="gp-badge">IBM Granite 4 H Small</span>
  <span class="gp-badge">RAG Retrieval</span>
  <span class="gp-badge">Demo Dataset</span>
</div>
""", unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="gp-section-title">Tell Us About Your Startup</div>', unsafe_allow_html=True)

with st.form("startup_form"):
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        startup_name = st.text_input(
            "Startup Name *",
            value="GrantPilot Demo",
            placeholder="e.g., MediSync AI",
        )
        domain = st.selectbox(
            "Domain / Industry *",
            options=["AI", "HealthTech", "FinTech", "EdTech", "AgriTech",
                     "CleanTech", "Cybersecurity", "SaaS", "DeepTech", "Social Impact", "Other"],
            index=0,
        )
        stage = st.selectbox(
            "Startup Stage *",
            options=["Pre-Seed", "Seed", "Early Stage", "Growth", "Scale"],
            index=1,
        )

    with col2:
        location = st.text_input(
            "Location *",
            value="India",
            placeholder="e.g., Bangalore, India",
        )
        funding_required = st.number_input(
            "Funding Required (₹) *",
            min_value=100_000,
            max_value=50_000_000,
            value=3_000_000,
            step=100_000,
            format="%d",
        )

    description = st.text_area(
        "Startup Description *",
        value=(
            "AI-powered intelligent business automation platform helping startups and small businesses "
            "automate repetitive workflows and improve operational efficiency."
        ),
        height=110,
    )

    st.markdown('</div>', unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍  Find Matching Grants", use_container_width=True, type="primary")

# ── Results ───────────────────────────────────────────────────────────────────
if submitted:
    if not startup_name.strip() or not description.strip():
        st.error("Please fill in Startup Name and Description.")
        st.stop()

    startup_profile = {
        "name":             startup_name.strip(),
        "domain":           domain,
        "stage":            stage,
        "location":         location.strip(),
        "funding_required": f"{funding_required:,}",
        "description":      description.strip(),
    }

    st.markdown(f'<div class="gp-section-title">Results for {startup_name}</div>', unsafe_allow_html=True)

    # ── KB summary bar ────────────────────────────────────────────────────────
    kb = get_kb_summary()
    st.markdown(
        f'<div class="kb-bar">'
        f'📚 Knowledge base: <strong>{kb["total_grants"]} grants indexed</strong> &nbsp;|&nbsp; '
        f'<strong>{top_k} retrieved</strong> for this search &nbsp;|&nbsp; '
        f'Domains covered: {", ".join(kb["domains"])} &nbsp;|&nbsp; '
        f'Source: grants.csv (demo dataset — not real-time data)'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Step 1: RAG retrieval ─────────────────────────────────────────────────
    with st.spinner("Searching grant knowledge base…"):
        matched_grants = retrieve_grants(startup_profile, top_k=top_k)

    st.markdown('<div class="gp-section-title">Matched Grants</div>', unsafe_allow_html=True)
    st.caption("Retrieved using TF-IDF cosine similarity · Ranked by relevance score")

    for i, grant in enumerate(matched_grants, 1):
        raw_score = grant.get("score", 0)
        score_pct = int(raw_score * 100)
        score_cls = "gp-score-high" if score_pct >= 50 else ("gp-score-med" if score_pct >= 20 else "gp-score-low")
        dims = score_grant_dimensions(grant, startup_profile)

        def pill(label, fit):
            if fit is True:  return f'<span class="gp-pill pill-yes">✓ {label}</span>'
            if fit is False: return f'<span class="gp-pill pill-no">✗ {label}</span>'
            return f'<span class="gp-pill pill-na">? {label}</span>'

        try:
            fmin = f"₹{int(grant.get('funding_min', 0)):,}"
            fmax = f"₹{int(grant.get('funding_max', 0)):,}"
        except (ValueError, TypeError):
            fmin, fmax = "N/A", "N/A"

        with st.expander(
            f"#{i}  {grant.get('grant_name', '—')}  ·  {grant.get('organization', '—')}",
            expanded=(i == 1),
        ):
            # Score + fit pills row
            st.markdown(
                f'<span class="gp-score {score_cls}">Match Score: {score_pct}%</span>&nbsp;&nbsp;'
                + pill("Domain", dims["domain_fit"])
                + pill("Location", dims["location_fit"])
                + pill("Stage", dims["stage_fit"])
                + pill("Funding", dims["funding_fit"]),
                unsafe_allow_html=True,
            )

            # Key metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Funding Range", f"{fmin} – {fmax}")
            m2.metric("Deadline", grant.get("deadline", "—"))
            m3.metric("Stage", grant.get("startup_stage", "—"))
            m4.metric("Location", grant.get("location", "—"))

            st.markdown("---")

            # Details
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Organization**  \n{grant.get('organization', '—')}")
                st.markdown(f"**Domain**  \n{grant.get('domain', '—')}")
                st.markdown(f"**Eligibility**  \n{grant.get('eligibility', '—')}")
            with col_b:
                st.markdown(f"**Required Documents**  \n{grant.get('required_documents', '—')}")
                st.markdown(f"**Application Type**  \nNot specified")
                st.markdown(f"**Language**  \nNot specified")

            st.markdown(f"**About this grant**  \n{grant.get('description', '—')}")

            # Why this grant matches
            match_reasons = []
            if dims["domain_fit"]:    match_reasons.append(f"Domain aligned ({grant.get('domain')} matches your {domain})")
            if dims["location_fit"]:  match_reasons.append(f"Location eligible ({grant.get('location')} covers {location})")
            if dims["stage_fit"]:     match_reasons.append(f"Stage match ({stage} is within {grant.get('startup_stage')})")
            if dims["funding_fit"]:   match_reasons.append(f"Funding ask (₹{funding_required:,}) is within grant range")
            if not match_reasons:     match_reasons.append("Partial keyword overlap detected — review eligibility carefully")

            st.markdown(
                '<div class="gp-notice-info"><strong>Why this grant matches</strong><br>'
                + "<br>".join(f"• {r}" for r in match_reasons)
                + "</div>",
                unsafe_allow_html=True,
            )

            if grant.get("application_url"):
                st.markdown(f"🔗 **[Apply / Learn More]({grant['application_url']})**")

    # ── Step 2: AI analysis ───────────────────────────────────────────────────
    credentials_ready = api_key_set and project_id_set

    if not credentials_ready:
        st.markdown('<div class="gp-section-title">AI Eligibility Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="gp-notice-warn">'
            '⚠️ <strong>AI features unavailable.</strong> '
            'Set IBM_API_KEY and WATSONX_PROJECT_ID in your .env file to enable '
            'Granite-powered eligibility analysis and proposal generation.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        grants_text = format_grants_for_prompt(matched_grants)

        # ── Eligibility analysis ──────────────────────────────────────────────
        st.markdown('<div class="gp-section-title">AI Eligibility Analysis & Ranking</div>', unsafe_allow_html=True)
        st.caption(f"Powered by IBM watsonx.ai · ibm/granite-4-h-small · Language: {language}")

        st.markdown(
            '<div class="gp-notice-warn">'
            '⚠️ <strong>AI-generated eligibility assessment.</strong> '
            'Final eligibility must be verified with the funding organization before applying.'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Granite is analyzing eligibility across matched grants…"):
            try:
                analysis = analyze_eligibility_and_rank(startup_profile, grants_text, language=language)
                st.markdown(
                    f'<div class="gp-card">{analysis}</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"AI analysis failed: {e}")
                analysis = None

        # ── Proposal generation ───────────────────────────────────────────────
        if matched_grants:
            best_grant = matched_grants[0]

            st.markdown('<div class="gp-section-title">AI-Generated Funding Proposal Draft</div>', unsafe_allow_html=True)
            st.caption(
                f"Draft for **{best_grant.get('grant_name')}** · "
                f"ibm/granite-4-h-small · Language: {language}"
            )

            st.markdown(
                '<div class="gp-notice-warn">'
                '⚠️ <strong>Human Review Required</strong> — This AI-generated draft must be reviewed, '
                'verified, and edited by a qualified person before submission. '
                'GrantPilot AI does not automatically submit applications.'
                '</div>',
                unsafe_allow_html=True,
            )

            with st.spinner(f"Generating proposal for '{best_grant.get('grant_name')}'…"):
                try:
                    proposal = generate_proposal(startup_profile, best_grant, language=language)

                    st.markdown(
                        f'<div class="proposal-box">{proposal.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown("")
                    st.download_button(
                        label="⬇️  Download Proposal as .txt",
                        data=proposal,
                        file_name=f"{startup_name.replace(' ', '_')}_grant_proposal.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"Proposal generation failed: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:.78rem;padding:.4rem 0'>"
    "GrantPilot AI &nbsp;·&nbsp; Built with IBM watsonx.ai Granite 4 &amp; Streamlit &nbsp;·&nbsp; "
    "Demo dataset — not real or live grant data"
    "</div>",
    unsafe_allow_html=True,
)
