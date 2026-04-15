1. Introduction
International trade requires classifying products using HS (Harmonized System) codes. These
codes determine duties, documentation, and compliance.
The HS system is hierarchical:
●
HS-6 (global) – used worldwide
●
HS-8/10 (country-specific) – detailed versions used by individual countries
You are provided HS data for Chapters 7–11, containing both:
1. Worldwide HS-6 codes, and
2. Country-specific HS-8/10 codes (e.g., US HTS)
The data is available here.
Your goal is to build a semantic search program that identifies the best HS code based on a
natural-language product description.
2. Problem Statement
Write a program that accepts a plain English description of a product (e.g.,
“Fresh round
tomatoes”
,
“Dried chickpeas”), and finds:
1. The most relevant 6-digit global HS code, and
2. The most relevant country-specific 8/10-digit HS code(s) under that category.
Your program should rely on semantic similarity, not just keyword matching.
3. Input & Output Requirements
Input
A free-text product description, such as:
●
“Dried chickpeas in bulk bags”
●
“Frozenx
`
x
`
peas”
●
“Fresh round tomatoes”
●
“Seed potatoes for planting”
Output
Your program must print:
●
Best HS-6 code + description
●
Best country-specific HS code (8/10-digit) + description
Example Output:
Query: "Fresh round tomatoes"
HS-6 Match:
070200 – Tomatoes, fresh or chilled
Country Code Match:
0702.00.20.20 – Tomatoes, round, fresh
4. Core Requirements
A. Hierarchical Search Structure
HS is hierarchical:
Chapter → Heading → HS-6 → HS-8/10
Your search must follow this:
1. Use semantic search to find the closest HS-6 category
2. Then search only within that HS-6 group to find the closest HS-8/10 code
This improves accuracy and avoids unrelated matches.
B. Semantic Search
You must implement semantic search using embeddings.
Tasks:
1. Convert HS descriptions (HS-6 and HS-8/10) into vector embeddings
2. Convert the query into an embedding
3. Compute similarity (e.g., cosine similarity)
4. Select the closest match following the hierarchy
You may use any embedding model (e.g., Sentence Transformers).
C. Baseline Comparison
Implement a simple keyword-based baseline, such as:
●
TF-IDF
●
Fuzzy keyword matching
Compare at least 5 queries and show where semantic search performs better.
5. Data Provided
You will receive:
1. Global 6-digit HS data
For Chapters 7–11:
●
HS-6 code
●
Description
2. Country-specific 8/10-digit HS data
Contains:
●
Full HS code
●
Description
●
Additional qualifiers (e.g.,
“frozen”
,
“for sowing”
,
“dried”)
You must load, clean, and structure these into a hierarchy.
6. Program Tasks
1. Parse and clean the HS data
Normalize fields and create a hierarchy:
Chapter → Heading → HS-6 → HS-8/10
2. Build embeddings
Generate embeddings for:
●
HS-6 descriptions
●
HS-8/10 descriptions
●
User query
3. Implement hierarchical semantic search
Process:
1. Find closest HS-6 using semantic similarity
2. Search within that HS-6 group to find closest HS-8/10
3. Print the results
4. Compare baseline vs semantic
Show examples where semantic search produces better matches than keyword-based search.
9. Bonus (Optional)
These are extra credit and not required:
●
Include a similarity score for each match
●
Provide a short explanation of why a code was selected
●
Implement synonym expansion (e.g., chickpea ↔ garbanzo)
●
Support multiple countries
●
Build a small UI or notebook demo