
HS Code Classification
Using Semantic Search
Requirements Document

Version
1.0 — Initial Requirements

Scope
HS Chapters 7–11 | Global HS-6 + Country-Specific HS-8/10

Priority
MUST = core requirement | SHOULD = expected | MAY = bonus/optional

Reference
Assignment: HS Code Classification Using Semantic Search

1. Purpose & Scope
This document specifies the functional and non-functional requirements for an HS Code Classification system. The system accepts plain-English product descriptions and returns the most relevant trade classification codes using semantic NLP search, operating hierarchically over HS Chapters 7–11.

In Scope:
	•	HS-6 global code identification using semantic similarity
	•	Country-specific HS-8/10 code identification within matched HS-6 group
	•	TF-IDF keyword baseline comparison across 5+ queries
	•	Embedding generation, caching, and reuse

Out of Scope:
	•	Real-time trade duty or tariff rate lookup
	•	HS Chapters outside 7–11
	•	Legal trade compliance certification

2. Functional Requirements
Priority levels follow MoSCoW notation: MUST (mandatory), SHOULD (highly desirable), MAY (optional bonus).

ID
Priority
Requirement
FR-01
MUST
Accept a free-text natural language product description as input
FR-02
MUST
Return the most relevant HS-6 (6-digit) global code and its description
FR-03
MUST
Return the most relevant HS-8/10 country-specific code and its description
FR-04
MUST
Use semantic similarity (embeddings + cosine) — not keyword matching alone
FR-05
MUST
Perform hierarchical search: HS-6 first, then HS-8/10 scoped to the matched HS-6
FR-06
MUST
Cover HS Chapters 7–11 for both HS-6 and HS-8/10 datasets
FR-07
MUST
Implement a TF-IDF keyword baseline for comparison
FR-08
MUST
Demonstrate comparison on at least 5 queries with both methods
FR-09
SHOULD
Display similarity score alongside each match
FR-10
SHOULD
Provide a short natural-language explanation for each match
FR-11
MAY
Support synonym expansion (e.g., garbanzo ↔ chickpea)
FR-12
MAY
Support multiple countries' HS-8/10 datasets
FR-13
MAY
Provide a CLI or notebook UI for interactive use

3. Non-Functional Requirements
ID
Category
Requirement
NFR-01
Performance
Embedding generation (offline) must complete within 5 minutes for the full dataset on standard hardware
NFR-02
Performance
Query response time (online) must be under 3 seconds on CPU after cache is loaded
NFR-03
Accuracy
Semantic search must outperform TF-IDF on at least 3 of the 5 demonstration queries
NFR-04
Reliability
System must handle missing descriptions, malformed codes, and empty query strings without crashing
NFR-05
Portability
Must run on Python 3.10+ without GPU; all dependencies installable via pip
NFR-06
Maintainability
Code must be modular with clear separation between data loading, embedding, and search logic
NFR-07
Reproducibility
Embedding cache must produce identical results on re-runs with unchanged source data

4. Data Requirements
4.1 Input Data
	•	Global HS-6 CSV: must cover all codes in Chapters 07–11 with a code column and a description column
	•	Country-specific HS-8/10 CSV: must include full code strings and descriptions
	•	Both files must be available locally before the program is run
	•	Character encoding: UTF-8; separator: comma or tab

4.2 Data Quality Rules
	•	Codes with missing descriptions must be skipped with a logged warning
	•	Codes with non-numeric characters after stripping dots/dashes must be flagged
	•	Duplicate codes must be deduplicated (first occurrence retained)
	•	HS-10 codes whose 6-digit prefix does not exist in the HS-6 dataset must be logged as orphaned

5. Dependencies & Environment
5.1 Python Packages

Package
Version
Purpose
Required?
sentence-transformers
>=2.2.0
Embedding model (all-MiniLM-L6-v2)
REQUIRED
scikit-learn
>=1.3.0
TF-IDF baseline + cosine similarity
REQUIRED
pandas
>=2.0.0
CSV loading, data cleaning, hierarchy construction
REQUIRED
numpy
>=1.24.0
Embedding array storage + fast math operations
REQUIRED
torch
>=2.0.0
Backend for sentence-transformers (CPU mode)
REQUIRED
transformers
>=4.30.0
Hugging Face model utilities (pulled by sentence-transformers)
REQUIRED
rapidfuzz
>=3.0.0
Optional fuzzy matching baseline
OPTIONAL
ipywidgets
>=8.0.0
Interactive UI for Jupyter Notebook demo
OPTIONAL
argparse
stdlib
CLI argument parsing
STDLIB
pickle
stdlib
Metadata cache serialization
STDLIB

5.2 Environment
	•	Python 3.10 or higher
	•	No GPU required; all inference must run on CPU
	•	Minimum 4 GB RAM recommended for embedding matrix
	•	Disk space: ~500 MB for model weights (downloaded automatically on first run)
	•	Internet required only on first run to download the embedding model

6. Test Cases
The following test cases must be verified before submission. TC-01 and TC-06 through TC-08 are baseline sanity checks; TC-02 through TC-05 specifically demonstrate semantic advantage over TF-IDF.

TC
Test Name
Input
Expected Output
Notes
TC-01
Exact keyword match
"Fresh tomatoes"
070200 + HS-10 tomato code
Both methods should match
TC-02
Synonym mismatch
"Garbanzo beans"
0713.20 Chickpeas code
Only semantic should match
TC-03
Modifier + product
"Dried chickpeas bulk"
0713.20.xx dried bulk code
Semantic captures 'dried' + 'bulk'
TC-04
Intent-based query
"Seed potatoes for planting"
0701.10 for sowing
Semantic captures planting intent
TC-05
Foreign name
"Aubergine slices"
0709.30 Eggplants
Semantic handles UK vs US name
TC-06
Frozen modifier
"Frozen green peas"
0710.22 Peas, frozen
Correct modifier + product code
TC-07
Paraphrase
"Fresh vine tomatoes"
070200 + round tomato HS-10
Paraphrase matched via embedding
TC-08
Empty/gibberish
"xyzzy123"
Graceful low-confidence output
No crash; low similarity score shown

7. Acceptance Criteria
The assignment is considered complete when all of the following acceptance criteria are met:

ID
Acceptance Criterion
AC-01
Search returns a valid HS-6 code for every query in Chapters 7–11
AC-02
Search returns a valid HS-8/10 code that is a child of the returned HS-6
AC-03
Semantic search is demonstrably better than TF-IDF on synonym/paraphrase queries
AC-04
At least 5 comparison queries are run and results are displayed side-by-side
AC-05
Program runs end-to-end from fresh install using only pip dependencies
AC-06
Embedding cache is generated on first run and reused on subsequent runs
AC-07
Malformed input does not crash the program

8. Deliverables
	•	src/ — All Python source modules (data_loader, embeddings, search, baseline, compare)
	•	main.py — CLI entrypoint accepting --query argument
	•	requirements.txt — Full pip dependency list with pinned versions
	•	README.md — Setup instructions, usage examples, and comparison results
	•	cache/ — Pre-computed embedding .npy files (or instructions to regenerate)
	•	data/ — Source CSV files for HS-6 and HS-8/10
	•	comparison_report — Table or notebook showing 5+ query comparisons
	•	(Optional) notebook.ipynb — Interactive Jupyter demo

9. Constraints & Assumptions
	•	The HS data provided by the instructor covers exactly Chapters 7–11 and is structurally consistent
	•	The country-specific HS-8/10 codes all have a parent HS-6 code present in the global dataset
	•	The Sentence Transformer model (all-MiniLM-L6-v2) is sufficient for English-language queries
	•	No authentication or external API access is required beyond model weight download
	•	Hierarchical filtering uses exact 6-digit prefix matching — no fuzzy prefix matching

