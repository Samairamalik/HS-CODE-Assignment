# HS Code Classification (Semantic Search)

This project classifies product descriptions into:

- Best global HS-6 code
- Best country-specific HS-8/10 code (scoped under matched HS-6)

It uses a hierarchical semantic search pipeline and a TF-IDF baseline for comparison.

## Project Structure

- `scripts/convert_pdfs_to_csv.py`: Converts provided PDF files to CSV datasets
- `data/hs6_global_chapters_07_11.csv`: Parsed global HS-6 records
- `data/hs_country_us_chapters_07_11.csv`: Parsed country-specific HS records
- `src/data_loader.py`: Cleaning, validation, hierarchy construction
- `src/embeddings.py`: SentenceTransformer embeddings + cache
- `src/search.py`: Hierarchical semantic search
- `src/baseline.py`: TF-IDF baseline search
- `src/compare.py`: Side-by-side comparison report generation
- `main.py`: CLI entrypoint

## Setup

```bash
/usr/local/bin/python3 -m pip install -r requirements.txt
```

## Convert PDFs to CSV

```bash
/usr/local/bin/python3 scripts/convert_pdfs_to_csv.py
```

## Run Single Query

```bash
/usr/local/bin/python3 main.py --query "Fresh round tomatoes"
```

## Run Semantic vs TF-IDF Comparison

```bash
/usr/local/bin/python3 main.py --compare
```

This prints a markdown table and writes `comparison_report.md`.

## Notes

- First semantic run downloads the embedding model and builds cache.
- Subsequent runs reuse cached embeddings from `cache/`.
- Search order is hierarchical: HS-6 first, then country-specific within that branch.
