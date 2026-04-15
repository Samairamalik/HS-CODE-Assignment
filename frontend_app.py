from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.search import HierarchicalSemanticSearcher

st.set_page_config(page_title="HS-FLOW Dashboard", page_icon="📦", layout="wide")


@st.cache_resource
def get_searcher() -> HierarchicalSemanticSearcher:
    return HierarchicalSemanticSearcher.from_csv(
        hs6_csv=Path("data/hs6_global_chapters_07_11.csv"),
        country_csv=Path("data/hs_country_us_chapters_07_11.csv"),
        cache_dir=Path("cache"),
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #06080f;
            --panel: #0f1420;
            --panel-soft: #111827;
            --text: #f4f7ff;
            --muted: #91a0bf;
            --accent: #2f6bff;
            --teal: #18f2b2;
            --border: #1d2638;
        }

        .stApp {
            background:
                radial-gradient(1200px 500px at 75% -10%, #1a2450 0%, transparent 55%),
                linear-gradient(180deg, #05070d 0%, #070b14 100%);
            color: var(--text);
        }

        .main-title {
            font-size: 2.6rem;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.02em;
            text-align: center;
            margin-top: 0.5rem;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: var(--muted);
            text-align: center;
            margin-bottom: 1.2rem;
        }

        .pill {
            display: inline-block;
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            border: 1px solid #314057;
            color: #c7d5f3;
            font-size: 0.74rem;
            margin-right: 0.45rem;
            margin-top: 0.45rem;
            background: rgba(21, 31, 47, 0.7);
        }

        .card {
            background: linear-gradient(180deg, rgba(20,27,40,0.95), rgba(15,20,32,0.95));
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem 1rem;
            margin-bottom: 0.8rem;
        }

        .card-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            color: #8fa1c7;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }

        .metric-big {
            font-size: 1.7rem;
            font-weight: 700;
            color: #f7fbff;
            margin-bottom: 0.25rem;
        }

        .metric-sub {
            color: var(--muted);
            font-size: 0.86rem;
        }

        .confidence-high { color: #16f2ad; font-weight: 700; }
        .confidence-medium { color: #ffcc66; font-weight: 700; }
        .confidence-low { color: #ff7d8f; font-weight: 700; }

        .json-box {
            background: #050a14;
            border: 1px solid #1a2842;
            border-radius: 12px;
            padding: 0.9rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            color: #95f3c8;
            font-size: 0.8rem;
            overflow-x: auto;
            white-space: pre;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def confidence_class(label: str) -> str:
    key = label.lower()
    if key == "high":
        return "confidence-high"
    if key == "medium":
        return "confidence-medium"
    return "confidence-low"


def main() -> None:
    inject_styles()

    st.markdown('<div class="main-title">Trade Compliance, Distilled</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Semantic HS-code mapping with hierarchical resolution across chapters 07-11.</div>',
        unsafe_allow_html=True,
    )

    with st.form("analyze"):
        query = st.text_input(
            "",
            placeholder="Describe your product (e.g., Frozen round tomatoes from Italy, bulk packed)",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Analyze")

    if not submitted:
        st.info("Enter a product description and click Analyze.")
        return

    if not query.strip():
        st.warning("Please enter a non-empty product description.")
        return

    searcher = get_searcher()
    result = searcher.search(query)

    left, right = st.columns([2.0, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Classification Path</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-big">HS-6 {result.hs6.code}</div><div class="metric-sub">{result.hs6.description}</div>',
            unsafe_allow_html=True,
        )

        if result.country:
            st.markdown(
                f'<div style="margin-top:0.9rem" class="metric-big">HS-8/10 {result.country.code}</div>'
                f'<div class="metric-sub">{result.country.description}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="margin-top:0.9rem" class="metric-sub">No country-specific child found for this HS-6 branch.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="margin-top:0.9rem">', unsafe_allow_html=True)
        st.markdown('<span class="pill">Semantic Precision</span><span class="pill">Hierarchical Matching</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top HS-6 Candidates</div>', unsafe_allow_html=True)
        for cand in result.top_hs6[:5]:
            st.markdown(
                f"- {cand.code} | {cand.description} | score={cand.score:.3f}",
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">AI Matching Logic</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-big">{result.match_percent:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-sub">Primary Match Score</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="margin-top:0.6rem">Confidence: <span class="{confidence_class(result.confidence)}">{result.confidence}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if result.notes:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Scope Notes</div>', unsafe_allow_html=True)
            for note in result.notes:
                st.markdown(f"- {note}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Why It Was Picked</div>', unsafe_allow_html=True)
        st.write(result.explanation)
        st.markdown('</div>', unsafe_allow_html=True)

        if result.top_country:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Top Country Candidates</div>', unsafe_allow_html=True)
            for cand in result.top_country[:5]:
                st.markdown(f"- {cand.code} | {cand.description} | score={cand.score:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

    payload = {
        "status": "COMPLETED",
        "query": result.query,
        "confidence": result.confidence,
        "match_percent": round(result.match_percent, 2),
        "notes": result.notes,
        "hs6": {"code": result.hs6.code, "description": result.hs6.description, "score": round(result.hs6.score, 4)},
        "country": (
            {
                "code": result.country.code,
                "description": result.country.description,
                "score": round(result.country.score, 4),
            }
            if result.country
            else None
        ),
    }

    st.markdown('<div class="card-title" style="margin-top:1rem">Developer Preview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="json-box">' + str(payload).replace("'", '"') + "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
