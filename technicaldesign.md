
HS Code Classification
Using Semantic Search
Technical Design Document

Version
1.0 — Initial Design

Scope
HS Chapters 7–11 | Global HS-6 + Country-Specific HS-8/10

Stack
Python 3.10+ | sentence-transformers | scikit-learn | pandas

Goal
Semantic NLP-based product → HS code matching with hierarchical search

1. System Overview
This system accepts a plain-English product description and identifies the most relevant HS (Harmonized System) trade classification codes using semantic vector search. It implements a two-level hierarchical search: first identifying the best 6-digit global HS code, then drilling into that group to find the best country-specific 8/10-digit code.

The system covers HS Chapters 7–11 (vegetables, fruit, coffee/tea, cereals, milling products) and uses Sentence Transformer embeddings to capture semantic meaning — enabling matches that keyword-based approaches miss (e.g., 'garbanzo beans' → chickpea codes).

2. System Architecture
2.1 High-Level Flow
The system operates in two distinct phases: an offline preprocessing phase (run once) and an online query phase (run per query).

Offline Phase (Preprocessing):
	•	Load raw HS-6 CSV and HS-8/10 CSV data
	•	Parse and clean all descriptions; normalize code formats
	•	Build hierarchical dictionary: Chapter → Heading → HS-6 → HS-8/10
	•	Generate Sentence Transformer embeddings for all HS-6 descriptions
	•	Generate Sentence Transformer embeddings for all HS-8/10 descriptions
	•	Cache all embeddings to disk (.npy files) for fast reuse

Online Phase (Query):
	•	Accept a free-text product description from the user
	•	Encode the query using the same Sentence Transformer model
	•	Compute cosine similarity between query and all HS-6 embeddings
	•	Select the best-matching HS-6 code
	•	Filter HS-8/10 codes to only those under the matched HS-6
	•	Compute cosine similarity between query and the filtered HS-8/10 embeddings
	•	Return and display the best HS-6 and best HS-8/10 match with scores

2.2 Technology Stack

Component
Tool / Library
Rationale
Language
Python 3.10+
Widely supported, rich ML ecosystem
Embedding Model
sentence-transformers (all-MiniLM-L6-v2)
Fast, free, local — no API needed
Similarity Metric
Cosine Similarity (numpy / scikit-learn)
Standard for NLP embedding comparison
Keyword Baseline
scikit-learn TF-IDF + cosine similarity
Lightweight, interpretable comparison
Data Handling
pandas
CSV/Excel parsing, hierarchy construction
Embedding Cache
numpy .npy / pickle
Avoids recomputation on every run
Optional UI
argparse CLI or Jupyter Notebook
For demo / bonus requirements

3. Data Design
3.1 Input Datasets
Dataset 1 — Global HS-6 (Chapters 7–11):
	•	File format: CSV
	•	Fields: hs6_code (string), description (string)
	•	Example: 070200, Tomatoes, fresh or chilled
	•	Codes are 6-digit numeric strings

Dataset 2 — Country-Specific HS-8/10:
	•	File format: CSV (e.g., US HTS data)
	•	Fields: full_code (string), description (string), qualifiers (optional)
	•	Example: 0702.00.20.20, Tomatoes, round, fresh
	•	Codes may contain dots/dashes; must be normalized for prefix matching

3.2 Hierarchy Construction
Each HS code encodes its own hierarchy in its prefix. The system extracts this programmatically:

HS-6 code:    070200
  Chapter:    07        (first 2 digits)
  Heading:    0702      (first 4 digits)
  HS-6:       070200    (6 digits)

HS-10 code:   0702.00.20.20
  Normalized: 0702002020
  Parent HS-6: 070200   (first 6 digits after stripping dots)

The hierarchy is stored as a nested Python dictionary for O(1) lookup during filtering:

hierarchy = {
  '07': {
    '0702': {
      '070200': {
        'description': 'Tomatoes, fresh or chilled',
        'hs10_codes': [
          { 'code': '0702002020', 'description': 'Tomatoes, round, fresh' },
          ...
        ]
      }
    }
  }
}

4. Embedding & Similarity Design
4.1 Embedding Model
Model: all-MiniLM-L6-v2 (via sentence-transformers library)
	•	Output: 384-dimensional dense float vector per input string
	•	Completely local — no API key required, no network calls at inference time
	•	Inference time: ~1–5ms per description on CPU; suitable for full HS dataset in minutes
	•	Captures semantic meaning — synonyms, paraphrases, and implied context are handled

Alternative models (if higher accuracy needed):
	•	all-mpnet-base-v2 — 768-dim, slower but more accurate
	•	paraphrase-multilingual-MiniLM-L12-v2 — supports multi-language queries

4.2 Cosine Similarity
Similarity between query vector q and each HS description vector d is computed as:

cosine_similarity(q, d) = (q · d) / (||q|| × ||d||)

# Implemented via scikit-learn:
from sklearn.metrics.pairwise import cosine_similarity
scores = cosine_similarity([query_embedding], hs6_embeddings)[0]
best_idx = scores.argmax()

Scores range from 0.0 (no similarity) to 1.0 (identical). A score above 0.65 is generally a strong match for HS descriptions.

4.3 Embedding Cache Strategy
Embeddings are expensive to compute on first run but are deterministic. The system caches them:
	•	hs6_embeddings.npy — numpy array of shape (N_hs6, 384)
	•	hs10_embeddings.npy — numpy array of shape (N_hs10, 384)
	•	hs6_meta.pkl / hs10_meta.pkl — matching lists of codes and descriptions
	•	Cache is invalidated and regenerated if the CSV source files change

5. Hierarchical Search Algorithm
5.1 Step 1 — HS-6 Search
def search_hs6(query: str) -> dict:
    query_emb = model.encode([query])
    scores = cosine_similarity(query_emb, hs6_embeddings)[0]
    best_idx = scores.argmax()
    return {
        'code': hs6_codes[best_idx],
        'description': hs6_descriptions[best_idx],
        'score': float(scores[best_idx])
    }

5.2 Step 2 — HS-8/10 Search (Scoped)
def search_hs10(query: str, parent_hs6: str) -> dict:
    # Filter to only codes under the matched HS-6
    indices = [i for i, c in enumerate(hs10_codes)
               if c.startswith(parent_hs6[:6])]
    if not indices:
        return None
    sub_embeddings = hs10_embeddings[indices]
    query_emb = model.encode([query])
    scores = cosine_similarity(query_emb, sub_embeddings)[0]
    best_local = scores.argmax()
    best_global = indices[best_local]
    return {
        'code': hs10_codes[best_global],
        'description': hs10_descriptions[best_global],
        'score': float(scores[best_local])
    }

6. Keyword Baseline (TF-IDF)
A TF-IDF baseline is implemented for direct comparison against semantic search. It is fit on all HS descriptions and uses cosine similarity on sparse TF-IDF vectors.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_descriptions)

def tfidf_search(query: str):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    return all_descriptions[scores.argmax()]

6.1 Comparison Examples
Below are example queries showing where semantic search outperforms TF-IDF:

Query
TF-IDF Result
Semantic Result
Why Semantic Wins
"garbanzo beans"
0713.90 – Other leguminous vegetables
0713.20 – Chickpeas (Garbanzos)
Synonyms: garbanzo = chickpea
"dried chickpeas bulk"
0713.20 – Chickpeas
0713.20.40 – Chickpeas, dried, in bulk
Modifier 'bulk' understood semantically
"seed potatoes planting"
0701.10 – Potatoes for sowing
0701.10.00 – Seed potatoes
Intent 'for planting' captured
"frozen green peas"
0710.10 – Potatoes (frozen)
0710.22 – Peas, frozen
Adjective 'green' does not mislead
"fresh vine tomatoes"
0702.00 – Tomatoes, fresh
0702.00.20 – Tomatoes, round, fresh
Paraphrase 'vine' matched correctly

7. Implementation Phases

Phase
Name
Description
1
Data Acquisition
Obtain & inspect HS-6 and HS-8/10 datasets for Chapters 7–11
2
Parsing & Hierarchy
Clean data, extract chapter/heading prefixes, build parent-child tree
3
Embedding Generation
Encode all HS descriptions + user query using Sentence Transformers
4
Hierarchical Search
2-step cosine similarity: HS-6 first, then HS-8/10 within matched group
5
Keyword Baseline
TF-IDF vectorizer for 5+ query comparison vs semantic search
6
Output & Report
Formatted results, similarity scores, comparison table
7
Bonus
Synonym expansion, explanations, UI / notebook demo

8. Project File Structure
hs_classifier/
├── data/
│   ├── hs6_chapters7_11.csv        # Global HS-6 data
│   └── hs10_country.csv            # Country-specific HS-8/10 data
├── cache/
│   ├── hs6_embeddings.npy          # Pre-computed HS-6 vectors
│   ├── hs10_embeddings.npy         # Pre-computed HS-8/10 vectors
│   ├── hs6_meta.pkl                # Codes + descriptions list
│   └── hs10_meta.pkl
├── src/
│   ├── data_loader.py              # Parse CSV, build hierarchy
│   ├── embeddings.py               # Generate + cache embeddings
│   ├── search.py                   # Hierarchical semantic search
│   ├── baseline.py                 # TF-IDF keyword search
│   └── compare.py                  # Side-by-side comparison runner
├── main.py                         # CLI entrypoint
├── requirements.txt
└── README.md

9. Output Format
Query: "Fresh round tomatoes"

─────────────────────────────────────────────────────
HS-6 Match:
  Code:        070200
  Description: Tomatoes, fresh or chilled
  Score:       0.912

Country-Specific Match (HS-10):
  Code:        0702.00.20.20
  Description: Tomatoes, round, fresh
  Score:       0.887
─────────────────────────────────────────────────────

10. Bonus Features (Optional)
	•	Similarity scores — already produced by cosine similarity; trivial to include in output
	•	Explanation — prompt an LLM with matched code + description to generate a justification
	•	Synonym expansion — maintain a small dict: { 'garbanzo': 'chickpea', 'aubergine': 'eggplant' }
	•	Multi-country support — parameterize the HS-8/10 source file by country
	•	CLI — argparse with --query, --country, --top-k, --baseline flags
	•	Jupyter Notebook — interactive demo with ipywidgets text input

