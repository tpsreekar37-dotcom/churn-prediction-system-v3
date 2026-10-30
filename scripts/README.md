# scripts/legacy

One-off generators kept for provenance. **Nothing in the application imports
these files** — `dashboard/`, `src/`, `tests/` and `main.py` do not reference
them, and they are not needed to run or deploy the project.

They were used to produce the deliverables that ship with this repository:

| Script | Produced |
|---|---|
| `generate_report.py`, `generate_pdf_report.py`, `generate_complete_pdf.py` | The PDF reports |
| `generate_word_report.py`, `create_churn_pdf.py` | The DOCX reports |
| `build_review_ppt.py`, `generate_run_guide.py`, `generate_master_viva_manual_pdf.py`, `generate_project_guide_pdf.py` | The presentation decks and guide PDFs |
| `generate_diagrams.py` | The DFD / ER / UML / architecture figures in `reports/figures/` |
| `make_guide_pdf.py` | `mini_cmd.pdf` (command cheat-sheet) |
| `inspect_ppt.py`, `inspect_template.py`, `read_all_ppt.py`, `dump_ppt.py`, `verify_slides.py` | Helper/inspection utilities used while building the decks |
| `fix_unicode.py` | One-shot encoding repair, patched `generate_project_guide_pdf.py` in place |
| `show_results.py` | Prints model comparison stats to the console |

Notes if you ever re-run them:

- They assume the **project root as the working directory**, e.g.
  `python scripts/legacy/generate_report.py` from the repository root, because
  some of them read `models/` and write to `reports/`.
- A few contain absolute paths from the machine that generated the reports
  (`make_guide_pdf.py` writes to `~/Desktop`). Edit those before reuse.
- They are not covered by the test suite.
