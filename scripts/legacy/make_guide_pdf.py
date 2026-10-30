"""
make_guide_pdf.py
-----------------
Generates mini_cmd.pdf on the Desktop containing the full command guide
for the Customer Churn Prediction project.
Run once:  python make_guide_pdf.py
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ─── Output path ────────────────────────────────────────────────────────────
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
OUTPUT  = os.path.join(DESKTOP, "mini_cmd.pdf")

# ─── Colour palette (matches dashboard gold & white theme) ───────────────────
GOLD       = colors.HexColor("#C59B27")
GOLD_DARK  = colors.HexColor("#8F6700")
GOLD_LIGHT = colors.HexColor("#FBF4E0")
BLACK      = colors.HexColor("#000000")
WHITE      = colors.white
CREAM      = colors.HexColor("#FAFAF7")
BORDER     = colors.HexColor("#DDC8A4")
CODE_BG    = colors.HexColor("#1E1E1E")
CODE_FG    = colors.HexColor("#D4D4D4")
GREEN_ACC  = colors.HexColor("#047857")
RED_ACC    = colors.HexColor("#B91C1C")

# ─── Document ────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm,  bottomMargin=2*cm,
    title="mini_cmd — Quick Command Guide",
    author="Business Analytics System",
)

W, H = A4
CONTENT_W = W - 4*cm

styles = getSampleStyleSheet()

# ─── Custom styles ────────────────────────────────────────────────────────────
def S(name, **kw):
    return ParagraphStyle(name, **kw)

TITLE_STYLE = S("Title2",
    fontSize=26, leading=32, textColor=GOLD_DARK,
    fontName="Helvetica-Bold", spaceAfter=4, alignment=TA_CENTER)

SUBTITLE_STYLE = S("Subtitle2",
    fontSize=11, leading=14, textColor=colors.HexColor("#555555"),
    fontName="Helvetica", spaceAfter=2, alignment=TA_CENTER)

SECTION_STYLE = S("Section",
    fontSize=13, leading=17, textColor=WHITE,
    fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=2)

BODY_STYLE = S("Body2",
    fontSize=10, leading=14, textColor=BLACK,
    fontName="Helvetica", spaceBefore=2, spaceAfter=2)

LABEL_STYLE = S("Label",
    fontSize=9, leading=12, textColor=GOLD_DARK,
    fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=1)

NOTE_STYLE = S("Note",
    fontSize=8.5, leading=12, textColor=colors.HexColor("#555555"),
    fontName="Helvetica-Oblique", spaceBefore=1, spaceAfter=3)

CODE_STYLE = S("Code2",
    fontSize=9.5, leading=13, textColor=CODE_FG,
    fontName="Courier-Bold", spaceBefore=0, spaceAfter=0,
    leftIndent=8, rightIndent=8)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def section_header(text, color=GOLD):
    """Gold band with white text."""
    tbl = Table([[Paragraph(text, SECTION_STYLE)]], colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), color),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ]))
    return tbl


def code_block(lines):
    """Dark terminal-style block for commands."""
    content = "<br/>".join(lines)
    para = Paragraph(content, CODE_STYLE)
    tbl = Table([[para]], colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), CODE_BG),
        ("ROUNDEDCORNERS",[4]),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
    ]))
    return tbl


def step_row(num, title, desc):
    """Numbered step with gold circle."""
    badge = Table([[Paragraph(str(num), S("b",
        fontSize=11, fontName="Helvetica-Bold",
        textColor=WHITE, alignment=TA_CENTER))]],
        colWidths=[0.65*cm], rowHeights=[0.65*cm])
    badge.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), GOLD),
        ("ROUNDEDCORNERS",[20]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",  (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
    ]))
    text_cell = [
        Paragraph(f"<b>{title}</b>", S("st",
            fontSize=10, fontName="Helvetica-Bold", textColor=GOLD_DARK, leading=13)),
        Paragraph(desc, NOTE_STYLE),
    ]
    tbl = Table([[badge, text_cell]], colWidths=[0.9*cm, CONTENT_W - 0.9*cm])
    tbl.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",   (0,0), (0,-1), 0),
        ("RIGHTPADDING",  (0,0), (0,-1), 6),
        ("TOPPADDING",    (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))
    return tbl


def two_col_table(rows, col_labels=("Command / Flag", "Description")):
    """Two-column reference table."""
    header = [Paragraph(c, S("h", fontSize=9, fontName="Helvetica-Bold",
                              textColor=WHITE)) for c in col_labels]
    data   = [header]
    for cmd, desc in rows:
        data.append([
            Paragraph(f'<font name="Courier-Bold" size="8.5">{cmd}</font>',
                      S("tc", fontSize=8.5, leading=12, textColor=CODE_BG)),
            Paragraph(desc, S("td", fontSize=8.5, leading=12, textColor=BLACK)),
        ])
    col_w = [CONTENT_W * 0.40, CONTENT_W * 0.60]
    tbl = Table(data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  GOLD),
        ("BACKGROUND",    (0,1), (-1,-1), CREAM),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [CREAM, WHITE]),
        ("GRID",          (0,0), (-1,-1), 0.4, BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 7),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    return tbl


def tip_box(text, icon="💡"):
    tip_text = f'<b>{icon}  Tip:</b>  {text}'
    para = Paragraph(tip_text, S("tip",
        fontSize=8.5, leading=12, textColor=GOLD_DARK, fontName="Helvetica"))
    tbl = Table([[para]], colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), GOLD_LIGHT),
        ("ROUNDEDCORNERS",[4]),
        ("BOX",           (0,0), (-1,-1), 0.8, GOLD),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ]))
    return tbl


def spacer(h=0.25):
    return Spacer(1, h*cm)


def divider():
    return HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=4, spaceBefore=4)


# ─── Build content ────────────────────────────────────────────────────────────
story = []

# ── Cover / title ─────────────────────────────────────────────────────────────
story.append(spacer(0.4))

# Gold top banner
banner = Table([[Paragraph(
    "⚜  Business Analytics System",
    S("bn", fontSize=8.5, fontName="Helvetica-Bold", textColor=WHITE,
      alignment=TA_CENTER))]], colWidths=[CONTENT_W])
banner.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,-1), GOLD),
    ("TOPPADDING",    (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
]))
story.append(banner)
story.append(spacer(0.4))

story.append(Paragraph("mini_cmd", TITLE_STYLE))
story.append(Paragraph("Quick Command Reference — Customer Churn Prediction Project", SUBTITLE_STYLE))
story.append(spacer(0.15))
story.append(Paragraph(
    "All commands run from:  "
    "<font name='Courier-Bold' size='9' color='#8F6700'>"
    "C:\\Users\\tpvij\\Desktop\\churn_predicting\\churn_predicting"
    "</font>",
    S("path", fontSize=9, leading=13, textColor=colors.HexColor("#555555"),
      fontName="Helvetica", alignment=TA_CENTER)))
story.append(spacer(0.3))
story.append(divider())
story.append(spacer(0.2))

# ── Section 1: Quick Start ────────────────────────────────────────────────────
story.append(KeepTogether([
    section_header("  1.  Quick Start — Run the Dashboard"),
    spacer(0.2),
    Paragraph("Open PowerShell or Command Prompt, navigate to the project folder, then launch.", BODY_STYLE),
    spacer(0.15),
    step_row(1, "Open PowerShell", "Press  Win + X  →  Windows PowerShell  (or search 'PowerShell' in Start menu)"),
    spacer(0.1),
    step_row(2, "Navigate to the project", "Paste the cd command below and press Enter"),
    spacer(0.1),
    code_block([
        "cd C:\\Users\\tpvij\\Desktop\\churn_predicting\\churn_predicting",
    ]),
    spacer(0.1),
    step_row(3, "Launch the Streamlit dashboard", "The browser opens automatically at  http://localhost:8501"),
    spacer(0.1),
    code_block([
        "streamlit run dashboard/app.py",
    ]),
    spacer(0.15),
    tip_box("Keep this PowerShell window open while using the dashboard. Press  Ctrl + C  to stop the server."),
    spacer(0.25),
]))

# ── Section 2: Retrain models ──────────────────────────────────────────────────
story.append(KeepTogether([
    section_header("  2.  Retrain All Models  (Full Pipeline)"),
    spacer(0.2),
    Paragraph(
        "Run <b>main.py</b> to execute the complete end-to-end ML pipeline: data loading, "
        "preprocessing, feature engineering, EDA, model training, evaluation, and SQLite analytics. "
        "Takes ~90 seconds. Run this whenever you update the dataset or want fresh model artifacts.",
        BODY_STYLE),
    spacer(0.15),
    code_block([
        "cd C:\\Users\\tpvij\\Desktop\\churn_predicting\\churn_predicting",
        "python main.py",
    ]),
    spacer(0.15),
    Paragraph("What it produces:", LABEL_STYLE),
    two_col_table([
        ("models/best_model.pkl",        "Production model artifact (calibrated Voting Ensemble)"),
        ("models/random_forest.pkl",     "Individual Random Forest artifact"),
        ("models/xgboost.pkl",           "Individual XGBoost artifact"),
        ("models/logistic_regression.pkl","Individual Logistic Regression artifact"),
        ("models/lightgbm.pkl",          "Individual LightGBM artifact"),
        ("models/svm.pkl",               "Individual SVM artifact"),
        ("models/model_results.json",    "Saved metrics for all models (canonical reference)"),
        ("models/model_metadata.json",   "Best model name, features, sklearn version"),
        ("models/feature_importance.csv","Feature importance rankings"),
        ("models/cv_results.json",       "5-fold cross-validation AUC scores"),
        ("data/churn.db",                "SQLite database with scored predictions"),
        ("reports/figures/*.png",        "EDA charts, confusion matrix, ROC curves"),
    ]),
    spacer(0.25),
]))

# ── Section 3: Install dependencies ───────────────────────────────────────────
story.append(KeepTogether([
    section_header("  3.  Install / Verify Dependencies"),
    spacer(0.2),
    Paragraph(
        "All required libraries are listed in <b>requirements.txt</b>. "
        "Run once on a new machine or after updating the file.",
        BODY_STYLE),
    spacer(0.15),
    code_block([
        "cd C:\\Users\\tpvij\\Desktop\\churn_predicting\\churn_predicting",
        "pip install -r requirements.txt",
    ]),
    spacer(0.15),
    Paragraph("Check Python version (requires 3.10 – 3.12):", LABEL_STYLE),
    code_block(["python --version"]),
    spacer(0.25),
]))

# ── Section 4: Run tests ───────────────────────────────────────────────────────
story.append(KeepTogether([
    section_header("  4.  Run the Test Suite"),
    spacer(0.2),
    Paragraph(
        "22 automated tests cover risk banding, schema validation, predictions, "
        "SQL storage, and the end-to-end pipeline.",
        BODY_STYLE),
    spacer(0.15),
    Paragraph("Run all tests (verbose output):", LABEL_STYLE),
    code_block([
        "cd C:\\Users\\tpvij\\Desktop\\churn_predicting\\churn_predicting",
        "python -m pytest tests/ -v",
    ]),
    spacer(0.1),
    Paragraph("Run a single test file:", LABEL_STYLE),
    code_block([
        "python -m pytest tests/test_pipeline_e2e.py -v",
        "python -m pytest tests/test_contracts.py -v",
        "python -m pytest tests/test_sql_integration.py -v",
    ]),
    spacer(0.25),
]))

# ── Section 5: Useful individual scripts ──────────────────────────────────────
story.append(KeepTogether([
    section_header("  5.  Run Individual Pipeline Steps"),
    spacer(0.2),
    two_col_table([
        ("python src/train_model.py",       "Train & evaluate models only (no EDA, no SQL)"),
        ("python src/predict.py",           "Run a sample single-customer prediction"),
        ("python src/sql_integration.py",   "Re-run SQLite analytics on the last scored run"),
        ("python src/preprocessing.py",     "Test data loading and preprocessing only"),
        ("python src/feature_engineering.py","Test feature engineering + SMOTE only"),
        ("python src/eda.py",               "Regenerate all EDA figures in reports/figures/"),
        ("python src/generic_pipeline.py",  "Test the schema-agnostic generic pipeline"),
    ]),
    spacer(0.25),
]))

# ── Section 6: Dashboard pages ────────────────────────────────────────────────
story.append(KeepTogether([
    section_header("  6.  Dashboard Navigation (http://localhost:8501)"),
    spacer(0.2),
    two_col_table([
        ("Dataset Import & Model Selection", "Load data, inspect schema, choose or auto-select model"),
        ("Executive Overview",               "KPIs: churn rate, high-risk count, MRR/ARR at risk"),
        ("Business Revenue & Financial Risk","Revenue exposure by contract, payment method, tenure"),
        ("Model Comparison & Evaluation",    "Compare all 6 models, grouped bar chart, confusion matrix"),
        ("Model Testing & Evaluation",       "Live ground-truth evaluation, epoch training telemetry"),
        ("Customer Retention Strategy",      "ROI simulator, outreach queue, diagnosed problems"),
    ], col_labels=("Page", "What it shows")),
    spacer(0.25),
]))

# ── Section 7: Troubleshooting ────────────────────────────────────────────────
story.append(KeepTogether([
    section_header("  7.  Troubleshooting"),
    spacer(0.2),
    two_col_table([
        ("Port 8501 already in use",
         "Stop the old server (Ctrl+C) or run:  streamlit run dashboard/app.py --server.port 8502"),
        ("No model artifacts found",
         "Run  python main.py  first to train and save all model .pkl files"),
        ("ModuleNotFoundError",
         "Run  pip install -r requirements.txt  to install missing libraries"),
        ("Excel file shows high missing values",
         "Normal — the app auto-strips Excel ghost columns on upload"),
        ("Browser does not open automatically",
         "Manually open  http://localhost:8501  in your browser"),
        ("Metrics differ between pages",
         "All pages use saved training metrics from models/model_results.json — always consistent"),
        ("Changes not appearing in browser",
         "Hard-refresh with  Ctrl + Shift + R  or restart the Streamlit server"),
    ], col_labels=("Problem", "Solution")),
    spacer(0.25),
]))

# ── Section 8: One-liner cheat sheet ──────────────────────────────────────────
story.append(section_header("  8.  One-Liner Cheat Sheet"))
story.append(spacer(0.2))
story.append(code_block([
    "# --- Navigate to project ---",
    "cd C:\\Users\\tpvij\\Desktop\\churn_predicting\\churn_predicting",
    "",
    "# --- Launch dashboard (most common command) ---",
    "streamlit run dashboard/app.py",
    "",
    "# --- Retrain all models from scratch ---",
    "python main.py",
    "",
    "# --- Install all dependencies ---",
    "pip install -r requirements.txt",
    "",
    "# --- Run all tests ---",
    "python -m pytest tests/ -v",
    "",
    "# --- Stop dashboard server ---",
    "Ctrl + C",
]))
story.append(spacer(0.3))

# ── Footer ─────────────────────────────────────────────────────────────────────
story.append(divider())
footer_tbl = Table([[
    Paragraph("Business Analytics System — AI Customer Churn Prediction",
              S("fl", fontSize=7.5, fontName="Helvetica", textColor=colors.HexColor("#999999"))),
    Paragraph("mini_cmd  •  python 3.12  •  streamlit 1.29",
              S("fr", fontSize=7.5, fontName="Helvetica", textColor=colors.HexColor("#999999"),
                alignment=2)),  # TA_RIGHT = 2
]], colWidths=[CONTENT_W * 0.6, CONTENT_W * 0.4])
footer_tbl.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]))
story.append(footer_tbl)

# ─── Build PDF ────────────────────────────────────────────────────────────────
doc.build(story)
print(f"PDF created:  {OUTPUT}")
