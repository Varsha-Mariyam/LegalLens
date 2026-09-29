# Synthetic sample documents (Dataset A — part 1)

**Every document in this folder is synthetic.** They were written for the LegalLens academic project
to exercise the pipeline (extraction, segmentation, classification, risk rules and comparison).
All names, companies, addresses and amounts are fictitious. They are deliberately written with a mix
of balanced and one-sided clauses so that risk analysis and comparison have something to find.

`scripts/generate_sample_files.py` renders each `.txt` into `.pdf` and `.docx`, and renders one
page image (`*_scanned.png` / `*_scanned.pdf`) to test the OCR path.

Real documents: `data/raw_documents/cuad/` holds commercial contracts from the CUAD v1 dataset
(downloaded by `scripts/download_datasets.py`).
