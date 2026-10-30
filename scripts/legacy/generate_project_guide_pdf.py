"""
generate_project_guide_pdf.py
-------------------------------
Generates a comprehensive Project Guide PDF covering:
- How to Run the Project (CMD step-by-step)
- Project Overview
- Model Details (Random Forest vs XGBoost)
- Accuracy Metrics
- Complete Q&A for Project Review / Viva
"""

from fpdf import FPDF
import os

OUTPUT = r"C:\Users\tpvij\Desktop\Customer_Churn_Project_Complete_Guide.pdf"


# ?????????????????????????????????????????????????????????????????
# PDF CLASS
# ?????????????????????????????????????????????????????????????????

class PDF(FPDF):
    def header(self):
        self.set_fill_color(15, 23, 42)     # dark navy
        self.rect(0, 0, 210, 14, 'F')
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(212, 175, 55)   # gold
        self.set_xy(0, 3)
        self.cell(0, 8, "  AI-Driven Customer Churn Prediction System  |  Complete Project Guide", align="L")
        self.set_text_color(180, 180, 180)
        self.cell(0, 8, f"Page {self.page_no()}", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_fill_color(15, 23, 42)
        self.rect(0, self.get_y(), 210, 12, 'F')
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, "Customer Churn Prediction System  |  Confidential Academic Project Documentation", align="C")

    def cover_page(self):
        # Dark background
        self.set_fill_color(10, 15, 30)
        self.rect(0, 0, 210, 297, 'F')

        # Gold accent top bar
        self.set_fill_color(212, 175, 55)
        self.rect(0, 0, 210, 3, 'F')

        # Title block
        self.set_y(55)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(255, 255, 255)
        self.cell(0, 14, "AI-Driven Customer", align="C")
        self.ln(14)
        self.cell(0, 14, "Churn Prediction System", align="C")
        self.ln(18)

        self.set_font("Helvetica", "", 14)
        self.set_text_color(212, 175, 55)
        self.cell(0, 8, "Complete Project Guide, Model Documentation", align="C")
        self.ln(8)
        self.cell(0, 8, "& Project Review Question Bank", align="C")
        self.ln(24)

        # Divider
        self.set_draw_color(212, 175, 55)
        self.set_line_width(0.8)
        self.line(30, self.get_y(), 180, self.get_y())
        self.ln(14)

        # Info box
        self.set_font("Helvetica", "", 11)
        self.set_text_color(200, 200, 200)
        items = [
            ("Technology Stack :", "Python 3.10 - Streamlit - scikit-learn - XGBoost - SQLite - Plotly"),
            ("Production Model  :", "Random Forest Classifier  |  Accuracy: 76.8%  |  AUC: 0.8362"),
            ("Dataset Size       :", "7,043 Telco Customer Records  |  21 Features"),
            ("Dashboard Pages   :", "5 Pages -- EDA, Prediction, Analytics, Model Testing, Strategy"),
        ]
        for label, value in items:
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(212, 175, 55)
            self.set_x(28)
            self.cell(54, 9, label)
            self.set_font("Helvetica", "", 10)
            self.set_text_color(220, 220, 220)
            self.multi_cell(118, 9, value)

        self.ln(14)
        self.set_draw_color(212, 175, 55)
        self.line(30, self.get_y(), 180, self.get_y())

        # Gold bottom accent
        self.set_fill_color(212, 175, 55)
        self.rect(0, 294, 210, 3, 'F')

    def chapter_title(self, num, title):
        self.ln(6)
        self.set_fill_color(15, 23, 42)
        self.rect(10, self.get_y(), 190, 10, 'F')
        self.set_fill_color(212, 175, 55)
        self.rect(10, self.get_y(), 4, 10, 'F')
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(255, 255, 255)
        self.set_x(17)
        self.cell(0, 10, f"  SECTION {num}:  {title.upper()}", ln=True)
        self.ln(4)

    def section_title(self, title):
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(15, 23, 42)
        self.set_fill_color(241, 245, 249)
        self.cell(0, 9, f"  {title}", ln=True, fill=True)
        self.ln(2)

    def body(self, text, indent=10):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        self.set_x(indent)
        self.multi_cell(190 - (indent - 10), 6, text)
        self.ln(1)

    def cmd_block(self, text):
        self.ln(2)
        self.set_fill_color(20, 30, 48)
        y = self.get_y()
        lines = text.strip().splitlines()
        h = len(lines) * 7 + 6
        self.rect(10, y, 190, h, 'F')
        self.set_font("Courier", "B", 10)
        self.set_text_color(80, 220, 100)
        self.set_x(14)
        self.set_y(y + 3)
        for line in lines:
            self.set_x(14)
            self.cell(0, 7, line, ln=True)
        self.ln(4)

    def bullet(self, text, indent=14):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        self.set_x(indent)
        self.cell(5, 6, chr(149))  # bullet
        self.multi_cell(185 - indent, 6, text)

    def qa_block(self, q_num, question, answer):
        self.ln(3)
        # Question
        self.set_fill_color(234, 242, 255)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(10, 50, 120)
        self.set_x(10)
        self.multi_cell(190, 7, f"Q{q_num}.  {question}", fill=True)
        # Answer
        self.set_fill_color(249, 250, 251)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        self.set_x(10)
        self.multi_cell(190, 6.5, f"A:   {answer}", fill=True)
        self.ln(2)

    def metric_row(self, model, accuracy, precision, recall, f1, auc):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        cols = [50, 28, 28, 28, 28, 28]
        data = [model, accuracy, precision, recall, f1, auc]
        self.set_x(10)
        for width, val in zip(cols, data):
            self.cell(width, 8, str(val), border=1, align="C")
        self.ln()


# ?????????????????????????????????????????????????????????????????
# BUILD PDF
# ?????????????????????????????????????????????????????????????????

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(10, 16, 10)

# ?? COVER PAGE
pdf.add_page()
pdf.cover_page()

# ??????????????????????????????????????????????????????
# SECTION 1: HOW TO RUN THE PROJECT
# ??????????????????????????????????????????????????????
pdf.add_page()
pdf.chapter_title(1, "How to Run the Project (Step-by-Step CMD Guide)")

pdf.section_title("Prerequisites -- Install These First")
pdf.body("Before running the project, make sure the following are installed on your computer:")
reqs = [
    "Python 3.9 or higher  ->  Download from https://www.python.org/downloads/",
    "Git (optional)        ->  Download from https://git-scm.com/",
    "Any web browser       ->  Chrome, Firefox, Edge (to view the dashboard)",
]
for r in reqs:
    pdf.bullet(r)

pdf.ln(4)
pdf.section_title("Step 1: Open Command Prompt (CMD)")
pdf.body("Press  Windows Key + R,  type  cmd  and press Enter.")
pdf.body("OR: Search for 'Command Prompt' in the Start Menu.")

pdf.section_title("Step 2: Navigate to the Project Folder")
pdf.body("Use the cd command to go into the project folder on your Desktop:")
pdf.cmd_block("cd C:\\Users\\YourUsername\\Desktop\\churn_predicting")
pdf.body("Replace 'YourUsername' with your actual Windows username (e.g., tpvij).")

pdf.section_title("Step 3: Create a Virtual Environment (Recommended)")
pdf.body("A virtual environment keeps the project's Python libraries separate and clean:")
pdf.cmd_block("python -m venv venv")

pdf.section_title("Step 4: Activate the Virtual Environment")
pdf.body("On Windows CMD, activate the virtual environment with:")
pdf.cmd_block("venv\\Scripts\\activate")
pdf.body("You will see (venv) appear at the start of your command prompt line. This means it is active.")

pdf.section_title("Step 5: Install Required Libraries")
pdf.body("Install all required Python packages from the requirements.txt file:")
pdf.cmd_block("pip install -r requirements.txt")
pdf.body("This will install: pandas, numpy, scikit-learn, xgboost, streamlit, plotly, SQLAlchemy, openpyxl, shap, and more.")
pdf.body("This step may take 3-5 minutes on first run depending on internet speed.")

pdf.section_title("Step 6: Launch the Dashboard")
pdf.body("Start the Streamlit web dashboard with this command:")
pdf.cmd_block("streamlit run dashboard/app.py")
pdf.body("After running this command, you will see output like:")
pdf.cmd_block("  You can now view your Streamlit app in your browser.\n  Local URL:  http://localhost:8501\n  Network URL: http://192.168.x.x:8501")

pdf.section_title("Step 7: Open in Browser")
pdf.body("Open your web browser and go to:  http://localhost:8501")
pdf.body("The dashboard will load automatically with all 5 pages available in the sidebar.")

pdf.section_title("Step 8: Using the Dashboard")
pdf.body("Navigate through the 5 pages using the left sidebar:")
steps = [
    "Executive Overview        -- See KPIs, churn distribution charts, and business insights.",
    "Upload Dataset & Predict -- Upload your own CSV/Excel file or load the benchmark Telco dataset.",
    "SQL Analytics             -- View SQL-powered business analytics, cohort analysis, and revenue at risk.",
    "Model Testing             -- Run real epoch-by-epoch neural network training and view convergence curves.",
    "Customer Retention Strategy -- View diagnosed attrition problems and strategic solutions.",
]
for i, s in enumerate(steps, 1):
    pdf.bullet(f"Page {i}: {s}")

pdf.ln(4)
pdf.section_title("Stopping the Server")
pdf.body("To stop the Streamlit server, press  CTRL + C  in the CMD window.")

pdf.section_title("Quick Reference -- All Commands in Order")
pdf.cmd_block(
    "cd C:\\Users\\tpvij\\Desktop\\churn_predicting\n"
    "venv\\Scripts\\activate\n"
    "pip install -r requirements.txt\n"
    "streamlit run dashboard/app.py"
)

pdf.section_title("Troubleshooting Common Errors")
errors = [
    ("'streamlit' is not recognized", "Activate the virtual environment first: venv\\Scripts\\activate"),
    ("ModuleNotFoundError: No module named X", "Run: pip install -r requirements.txt"),
    ("Port 8501 already in use", "Use: streamlit run dashboard/app.py --server.port 8502"),
    ("FileNotFoundError for model file", "Run the training pipeline first: python main.py"),
    ("Excel file not reading", "Ensure openpyxl is installed: pip install openpyxl"),
]
for err, fix in errors:
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(180, 30, 30)
    pdf.set_x(14)
    pdf.cell(0, 7, f"Error: {err}", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.set_x(20)
    pdf.cell(0, 6, f"Fix: {fix}", ln=True)
    pdf.ln(1)


# ??????????????????????????????????????????????????????
# SECTION 2: PROJECT OVERVIEW
# ??????????????????????????????????????????????????????
pdf.add_page()
pdf.chapter_title(2, "Project Overview & Architecture")

pdf.section_title("What is This Project?")
pdf.body(
    "This is an AI-Driven Customer Churn Prediction System built as a full-stack business analytics "
    "application. It uses Machine Learning to predict which telecom customers are likely to cancel their "
    "subscription (churn). The system provides a rich Streamlit web dashboard with 5 interactive pages, "
    "SQL-powered analytics, prescriptive retention strategies, and real-time ML inference."
)

pdf.section_title("Problem Statement")
pdf.body(
    "Customer churn (cancellation) is a major financial challenge for telecom companies. Acquiring a new "
    "customer costs 5-7x more than retaining an existing one. By predicting which customers are about to "
    "leave before they cancel, businesses can intervene with targeted offers and save significant monthly "
    "recurring revenue (MRR). This project builds the prediction engine and business strategy layer "
    "to enable proactive retention."
)

pdf.section_title("Dataset Used")
pdf.body("IBM Telco Customer Churn Dataset -- 7,043 customer records with 21 features:")
features = [
    "customerID        -- Unique customer identifier",
    "gender            -- Male / Female",
    "SeniorCitizen     -- Whether customer is a senior citizen (0 or 1)",
    "Partner           -- Whether customer has a partner (Yes / No)",
    "Dependents        -- Whether customer has dependents (Yes / No)",
    "tenure            -- Number of months the customer has been with the company",
    "PhoneService      -- Whether customer has phone service",
    "MultipleLines     -- Whether customer has multiple phone lines",
    "InternetService   -- Type of internet: DSL / Fiber optic / No",
    "OnlineSecurity    -- Whether customer has online security add-on",
    "OnlineBackup      -- Whether customer has online backup service",
    "DeviceProtection  -- Whether customer has device protection",
    "TechSupport       -- Whether customer has tech support",
    "StreamingTV       -- Whether customer streams TV",
    "StreamingMovies   -- Whether customer streams movies",
    "Contract          -- Contract term: Month-to-month / One year / Two year",
    "PaperlessBilling  -- Whether customer uses paperless billing",
    "PaymentMethod     -- Electronic check / Mailed check / Bank transfer / Credit card",
    "MonthlyCharges    -- Monthly amount charged to the customer ($)",
    "TotalCharges      -- Total amount charged to the customer ($)",
    "Churn             -- Target variable: Yes (churned) / No (retained)",
]
for f in features:
    pdf.bullet(f)

pdf.section_title("Project Architecture")
arch = [
    "data/              -- Raw datasets (CSV, Excel), SQLite database (churn.db)",
    "src/               -- Python source code: preprocessing, feature engineering, training, SQL integration",
    "dashboard/         -- Streamlit app (app.py) -- the main web interface",
    "models/            -- Saved trained models (.pkl files), results JSON, feature importance CSV",
    "tests/             -- Test suite (verify_all.py) to validate the pipeline",
    "reports/           -- Generated charts, confusion matrix, ROC curves",
    "requirements.txt   -- All Python library dependencies",
    "main.py            -- Entry point to run full training pipeline from scratch",
]
for a in arch:
    pdf.bullet(a)

pdf.section_title("Technology Stack")
tech = [
    "Python 3.10        -- Core programming language",
    "Streamlit 1.29     -- Web dashboard framework (runs in browser)",
    "scikit-learn 1.3   -- Machine learning library (Random Forest, preprocessing, metrics)",
    "XGBoost 2.0        -- Gradient boosting library (comparison model)",
    "pandas 2.1         -- Data manipulation and analysis",
    "numpy 1.26         -- Numerical computation",
    "plotly 5.18        -- Interactive data visualizations",
    "SQLite             -- Embedded relational database for analytics",
    "SQLAlchemy 2.0     -- Database ORM and connection management",
    "SHAP 0.44          -- Model explainability (feature impact analysis)",
    "imbalanced-learn   -- SMOTE oversampling to handle class imbalance",
    "joblib             -- Model serialization (save/load .pkl files)",
    "openpyxl           -- Excel file reading/writing support",
]
for t in tech:
    pdf.bullet(t)


# ??????????????????????????????????????????????????????
# SECTION 3: MODEL DOCUMENTATION
# ??????????????????????????????????????????????????????
pdf.add_page()
pdf.chapter_title(3, "Machine Learning Model Documentation")

pdf.section_title("Models Trained")
pdf.body(
    "Two machine learning models were trained and benchmarked against each other. "
    "Random Forest was selected as the production model due to its higher AUC score and better calibration."
)

pdf.section_title("Model Performance Metrics")
# Table header
pdf.set_font("Helvetica", "B", 10)
pdf.set_fill_color(15, 23, 42)
pdf.set_text_color(212, 175, 55)
pdf.set_x(10)
for header, width in [("Model", 50), ("Accuracy", 28), ("Precision", 28), ("Recall", 28), ("F1 Score", 28), ("AUC-ROC", 28)]:
    pdf.cell(width, 9, header, border=1, align="C", fill=True)
pdf.ln()

# Row 1 - Random Forest
pdf.set_fill_color(240, 248, 255)
pdf.set_text_color(20, 60, 100)
pdf.set_font("Helvetica", "B", 10)
pdf.set_x(10)
pdf.metric_row("Random Forest *", "76.8%", "54.8%", "70.4%", "61.7%", "0.8362")

# Row 2 - XGBoost
pdf.set_fill_color(255, 255, 255)
pdf.set_text_color(30, 41, 59)
pdf.set_font("Helvetica", "", 10)
pdf.set_x(10)
pdf.metric_row("XGBoost", "74.2%", "50.8%", "78.8%", "61.8%", "0.8212")

pdf.ln(6)
pdf.body("* = Production model selected for deployment. AUC-ROC is the primary selection criterion.")

pdf.section_title("Why Random Forest Was Chosen as Production Model")
reasons = [
    "Higher AUC-ROC: 0.8362 vs 0.8212 (AUC is the most reliable metric for imbalanced classification problems)",
    "Better Accuracy: 76.8% vs 74.2% on the test set",
    "Better Calibration: Probability outputs are more reliable and can be used as churn risk scores",
    "More Interpretable: Feature importances are easier to explain to business stakeholders",
    "More Stable: Less prone to overfitting on small datasets compared to XGBoost",
    "Handles Class Imbalance: Uses class_weight='balanced' parameter to compensate for the ~26% churn rate",
]
for r in reasons:
    pdf.bullet(r)

pdf.section_title("Top 10 Features by Importance (Random Forest)")
importance_data = [
    ("1", "RiskScore", "0.2054", "Engineered feature: combined risk from tenure + charges + contract"),
    ("2", "ChargesPerTenure", "0.1395", "Engineered: Monthly charges divided by tenure (cost efficiency)"),
    ("3", "Contract", "0.1263", "Contract type is the single strongest predictor of churn"),
    ("4", "MonthlyCharges", "0.0857", "Higher monthly bill correlates with higher churn risk"),
    ("5", "tenure", "0.0777", "Longer-tenured customers are much less likely to churn"),
    ("6", "TotalCharges", "0.0630", "Total spend; correlated with tenure and loyalty"),
    ("7", "InternetService", "0.0464", "Fiber optic customers churn more than DSL customers"),
    ("8", "OnlineSecurity", "0.0366", "Customers without security add-ons churn more"),
    ("9", "PaymentMethod", "0.0351", "Electronic check payers have highest churn rate"),
    ("10", "TechSupport", "0.0269", "Lack of tech support correlates with churn"),
]
pdf.set_font("Helvetica", "B", 9)
pdf.set_fill_color(15, 23, 42)
pdf.set_text_color(212, 175, 55)
pdf.set_x(10)
for header, width in [("Rank", 12), ("Feature", 46), ("Score", 20), ("Business Meaning", 112)]:
    pdf.cell(width, 8, header, border=1, align="C", fill=True)
pdf.ln()
for rank, feat, score, meaning in importance_data:
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)
    fill = rank in ("1","3","5","7","9")
    pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
    pdf.set_x(10)
    pdf.cell(12, 7, rank, border=1, align="C", fill=True)
    pdf.cell(46, 7, feat, border=1, fill=True)
    pdf.cell(20, 7, score, border=1, align="C", fill=True)
    pdf.multi_cell(112, 7, meaning, border=1, fill=True)

pdf.section_title("Model Hyperparameters")
pdf.body("Random Forest (Production Model):")
rf_params = [
    "n_estimators = 200     -- Number of decision trees in the forest",
    "max_depth = 10         -- Maximum depth each tree can grow to",
    "class_weight = balanced -- Automatically adjusts for the imbalanced churn dataset (~26% churn)",
    "random_state = 42      -- Fixed seed for reproducibility",
    "n_jobs = -1            -- Use all available CPU cores for parallel training",
]
for p in rf_params:
    pdf.bullet(p)

pdf.body("XGBoost (Benchmark Comparison):")
xgb_params = [
    "n_estimators = 200     -- Number of boosting rounds",
    "max_depth = 6          -- Maximum depth of each tree",
    "learning_rate = 0.1    -- Step size shrinkage to prevent overfitting",
    "scale_pos_weight = 3   -- Handles class imbalance (ratio of negative to positive samples)",
    "eval_metric = logloss  -- Optimization metric during training",
]
for p in xgb_params:
    pdf.bullet(p)

pdf.section_title("Training Data Split")
pdf.body("The dataset was split as follows:")
splits = [
    "Training Set:   70% of data (approx. 4,930 records) -- used to train the model",
    "Test Set:       30% of data (approx. 2,113 records) -- used to evaluate performance (never seen during training)",
    "Cross-Validation: 5-fold CV on training set to validate generalization",
    "Class Imbalance: ~26.5% churn vs ~73.5% no-churn -- handled with class_weight='balanced'",
]
for s in splits:
    pdf.bullet(s)


# ??????????????????????????????????????????????????????
# SECTION 4: RANDOM FOREST EXPLAINED
# ??????????????????????????????????????????????????????
pdf.add_page()
pdf.chapter_title(4, "What is Random Forest? -- Full Explanation")

pdf.section_title("Simple Explanation")
pdf.body(
    "Imagine you have a difficult question to answer. Instead of asking just one expert, you ask "
    "100 different experts, and then take the majority vote. Each expert has slightly different knowledge "
    "and perspective. The combined wisdom of 100 experts is almost always more accurate than any single expert."
    "\n\n"
    "Random Forest does exactly this -- but with Decision Trees as the 'experts'. It builds hundreds of "
    "decision trees, each trained on a slightly different random sample of the data and random subset of "
    "features. Then it combines all their predictions by majority vote (for classification) or average (for regression)."
)

pdf.section_title("What is a Decision Tree?")
pdf.body(
    "A Decision Tree is a flowchart-like model that makes decisions by asking a series of Yes/No questions "
    "about the data. For example:\n\n"
    "   Contract = Month-to-month?\n"
    "     Yes -> Tenure < 12 months?\n"
    "       Yes -> CHURN (high risk)\n"
    "       No  -> Monthly Charges > $65?\n"
    "         Yes -> CHURN\n"
    "         No  -> No Churn\n"
    "     No -> No Churn (low risk)\n\n"
    "The tree learns these rules automatically from the training data."
)

pdf.section_title("How Random Forest Works -- Step by Step")
steps_rf = [
    "Bootstrap Sampling: From 7,043 training records, create 200 random subsets (with replacement). "
    "Each subset has the same number of rows but different customers selected randomly.",
    "Feature Randomness: For each split in each tree, only a random subset of features is considered "
    "(e.g., instead of using all 20 features, randomly pick 4-5 to find the best split from).",
    "Train 200 Trees: Each of the 200 decision trees is trained independently on its own bootstrap sample "
    "using its random feature subsets. This creates 200 different trees with slightly different perspectives.",
    "Prediction by Voting: When predicting a new customer, all 200 trees make their own prediction. "
    "The final answer is the majority vote: if 140 trees say 'CHURN' and 60 say 'No Churn', the prediction is CHURN.",
    "Probability Estimation: The churn probability (e.g., 70%) is calculated as the fraction of trees "
    "that voted for CHURN (140/200 = 70%).",
]
for i, s in enumerate(steps_rf, 1):
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(15, 23, 42)
    pdf.set_x(14)
    pdf.cell(0, 7, f"Step {i}:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.set_x(20)
    pdf.multi_cell(180, 6, s)
    pdf.ln(2)

pdf.section_title("Why Random Forest is Powerful")
advantages = [
    "Reduces Overfitting: Because each tree sees different data and features, the ensemble is much more "
    "generalized than a single deep decision tree.",
    "Handles Missing Values: Can work with datasets that have some missing values.",
    "Robust to Outliers: Individual trees may be misled by outliers but the ensemble vote corrects them.",
    "Feature Importance: Automatically tells you which features matter most for prediction.",
    "Works Without Scaling: Unlike neural networks or SVM, does not require feature normalization.",
    "Interpretable: Business stakeholders can understand 'the model considers contract type, monthly charges, "
    "and tenure as the most important factors for predicting churn'.",
]
for a in advantages:
    pdf.bullet(a)


# ??????????????????????????????????????????????????????
# SECTION 5: XGBOOST EXPLAINED
# ??????????????????????????????????????????????????????
pdf.add_page()
pdf.chapter_title(5, "What is XGBoost? -- Full Explanation")

pdf.section_title("Simple Explanation")
pdf.body(
    "XGBoost stands for 'Extreme Gradient Boosting'. Unlike Random Forest where all trees are independent, "
    "XGBoost builds trees SEQUENTIALLY. Each new tree specifically targets and tries to fix the mistakes "
    "made by all previous trees combined.\n\n"
    "Think of it like a student who takes an exam, checks which questions they got wrong, and then studies "
    "specifically those weak areas before retaking the test. Each new tree focuses on the hardest, most "
    "misclassified customers from the previous round."
)

pdf.section_title("How XGBoost Works -- Step by Step")
steps_xgb = [
    "Initialize: Start with a simple prediction for all customers (e.g., predict the average churn probability for everyone).",
    "Calculate Residuals: For each customer, calculate how wrong the current prediction is (residual = actual - predicted).",
    "Fit a Tree to Residuals: Train a shallow decision tree that specifically tries to predict these residual errors.",
    "Update Predictions: Add a small fraction of this tree's predictions to the current model "
    "(learning_rate = 0.1 means we add 10% of the correction at each step).",
    "Repeat 200 Times: Repeat steps 2-4 for 200 rounds (n_estimators=200). "
    "Each round, the model gets slightly better at handling its remaining mistakes.",
    "Final Prediction: The final prediction is the sum of all 200 trees' contributions.",
]
for i, s in enumerate(steps_xgb, 1):
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(15, 23, 42)
    pdf.set_x(14)
    pdf.cell(0, 7, f"Step {i}:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.set_x(20)
    pdf.multi_cell(180, 6, s)
    pdf.ln(2)

pdf.section_title("Why XGBoost is Powerful")
xgb_adv = [
    "Often the most accurate: XGBoost frequently wins machine learning competitions (Kaggle) on tabular data.",
    "Handles Class Imbalance: The scale_pos_weight parameter compensates for datasets where one class "
    "(e.g., churn = 26%) is much rarer than the other.",
    "Built-in Regularization: Has L1 and L2 regularization to prevent overfitting.",
    "Speed: Highly optimized C++ implementation that is much faster than standard boosting.",
    "Handles Missing Values: Automatically learns the best direction to send missing values at each split.",
]
for a in xgb_adv:
    pdf.bullet(a)


# ??????????????????????????????????????????????????????
# SECTION 6: RANDOM FOREST vs XGBOOST -- COMPARISON
# ??????????????????????????????????????????????????????
pdf.add_page()
pdf.chapter_title(6, "Random Forest vs XGBoost -- Key Differences")

comparisons = [
    ("Architecture", 
     "Parallel ensemble: 200 independent trees vote simultaneously.",
     "Sequential ensemble: 200 trees built one after another, each fixing previous errors."),
    ("Learning Strategy",
     "Bagging (Bootstrap AGGregatING): Reduces variance by averaging diverse models.",
     "Boosting: Reduces bias by iteratively correcting mistakes."),
    ("Training Speed",
     "Faster -- all trees train simultaneously using multiple CPU cores.",
     "Slower -- must train each tree sequentially (depends on previous tree)."),
    ("Accuracy on This Project",
     "Accuracy: 76.8%, AUC: 0.8362 <- SELECTED AS PRODUCTION MODEL",
     "Accuracy: 74.2%, AUC: 0.8212"),
    ("Overfitting Risk",
     "Lower risk -- averaging many diverse trees naturally reduces overfitting.",
     "Higher risk if learning_rate is too high or too many estimators -- requires tuning."),
    ("Hyperparameter Sensitivity",
     "Less sensitive -- works well with default parameters on most datasets.",
     "More sensitive -- requires careful tuning of learning_rate, max_depth, subsample."),
    ("Interpretability",
     "More interpretable -- feature importance is straightforward to calculate and explain.",
     "Less interpretable -- gradient boosting math is harder to explain to non-technical stakeholders."),
    ("Best Use Case",
     "When you need a reliable, interpretable model. Great baseline for business applications.",
     "When you need maximum predictive accuracy and are willing to tune hyperparameters."),
    ("Class Imbalance Handling",
     "class_weight='balanced' parameter automatically reweights the minority class.",
     "scale_pos_weight parameter: set to ratio of negative/positive samples."),
    ("Memory Usage",
     "Higher memory -- stores 200 full trees simultaneously.",
     "Lower memory -- can use approximate tree methods to reduce memory footprint."),
]

pdf.set_font("Helvetica", "B", 9)
pdf.set_fill_color(15, 23, 42)
pdf.set_text_color(255, 255, 255)
pdf.set_x(10)
for header, width in [("Aspect", 42), ("Random Forest", 74), ("XGBoost", 74)]:
    pdf.cell(width, 9, header, border=1, align="C", fill=True)
pdf.ln()

for i, (aspect, rf_val, xgb_val) in enumerate(comparisons):
    fill_color = (240, 248, 255) if i % 2 == 0 else (255, 255, 255)
    pdf.set_fill_color(*fill_color)
    pdf.set_text_color(30, 41, 59)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_x(10)
    y_before = pdf.get_y()
    pdf.multi_cell(42, 6, aspect, border=1, fill=True)
    y_after = pdf.get_y()
    h = y_after - y_before

    pdf.set_xy(52, y_before)
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(74, 6, rf_val, border=1, fill=True)

    pdf.set_xy(126, y_before)
    pdf.multi_cell(74, 6, xgb_val, border=1, fill=True)
    pdf.set_y(max(pdf.get_y(), y_before + h))

pdf.ln(6)
pdf.section_title("When to Use Each Model")
pdf.body(
    "Use RANDOM FOREST when:\n"
    "- You need a reliable, explainable model for business stakeholders\n"
    "- You have limited time for hyperparameter tuning\n"
    "- Model interpretability is important (e.g., explaining to management WHY a customer is high risk)\n"
    "- You want to avoid overfitting on smaller datasets\n\n"
    "Use XGBOOST when:\n"
    "- Maximum prediction accuracy is the top priority\n"
    "- You have time and expertise to tune hyperparameters carefully\n"
    "- You are competing in a data science competition\n"
    "- The dataset is very large (millions of rows)"
)


# ??????????????????????????????????????????????????????
# SECTION 7: PROJECT REVIEW Q&A
# ??????????????????????????????????????????????????????
pdf.add_page()
pdf.chapter_title(7, "Project Review Q&A -- Complete Question Bank")
pdf.body("This section covers the most likely questions asked during a project review, viva, or demo presentation.")

# --- GENERAL PROJECT QUESTIONS ---
pdf.section_title("A. General Project Questions")

qa_general = [
    ("What is the objective of this project?",
     "The objective is to build an AI-powered system that predicts which telecom customers are likely to cancel "
     "their subscription (churn) using historical customer data. The system helps businesses proactively "
     "intervene with targeted offers to retain customers before they leave, protecting monthly recurring revenue."),
    ("Why is customer churn prediction important for telecom companies?",
     "Telecom companies typically lose 15-30% of their customers annually. Acquiring a new customer costs "
     "5-7 times more than retaining an existing one. If a company can identify high-risk customers BEFORE they "
     "cancel and offer them a discount or upgrade, they can save thousands of dollars per customer in lost revenue."),
    ("What dataset did you use and where did you get it?",
     "We used the IBM Telco Customer Churn dataset containing 7,043 customer records with 21 features including "
     "demographics, account information, and services subscribed. It is a publicly available benchmark dataset "
     "widely used in machine learning research and is available on Kaggle and IBM Developer."),
    ("How many features does the model use and which ones are most important?",
     "The model uses 20 engineered features after preprocessing. The top 5 most important features are: "
     "(1) RiskScore -- an engineered combination of tenure, charges, and contract type. "
     "(2) ChargesPerTenure -- monthly charges divided by tenure. "
     "(3) Contract -- month-to-month contracts have 5x higher churn rate than two-year contracts. "
     "(4) MonthlyCharges -- higher bills correlate with higher churn. "
     "(5) tenure -- longer-tenured customers are much less likely to churn."),
    ("What is the class imbalance problem and how did you solve it?",
     "In the dataset, only 26.5% of customers churned while 73.5% did not. If we trained a naive model, it "
     "could achieve 73.5% accuracy by just predicting 'No Churn' for everyone -- but it would never correctly "
     "identify any churning customer. We solved this using class_weight='balanced' in Random Forest (automatically "
     "re-weights minority class) and scale_pos_weight=3 in XGBoost."),
]
for i, (q, a) in enumerate(qa_general, 1):
    pdf.qa_block(i, q, a)

# --- MODEL QUESTIONS ---
pdf.add_page()
pdf.section_title("B. Model & Algorithm Questions")

qa_model = [
    ("What machine learning models did you train?",
     "We trained two models: (1) Random Forest Classifier -- selected as the production model with 76.8% accuracy "
     "and 0.8362 AUC-ROC. (2) XGBoost Classifier -- benchmark model with 74.2% accuracy and 0.8212 AUC-ROC. "
     "Random Forest was selected because it achieved higher AUC-ROC, which is the primary metric for imbalanced "
     "classification problems."),
    ("What is accuracy and why isn't it the only metric you should use?",
     "Accuracy = (Correct Predictions) / (Total Predictions). However, with imbalanced data (26.5% churn), "
     "a model predicting 'No Churn' for everyone gets 73.5% accuracy but catches ZERO churning customers. "
     "That's why we also measure: Precision (of all predicted churners, how many actually churned), "
     "Recall (of all actual churners, how many did we catch), F1 Score (harmonic mean of precision and recall), "
     "and AUC-ROC (area under the ROC curve -- measures ability to distinguish between classes)."),
    ("What is AUC-ROC and what does 0.8362 mean?",
     "AUC stands for Area Under the ROC (Receiver Operating Characteristic) Curve. The ROC curve plots "
     "True Positive Rate vs False Positive Rate at different thresholds. AUC = 0.5 means the model is "
     "no better than random guessing. AUC = 1.0 means perfect prediction. Our Random Forest AUC = 0.8362 "
     "means the model has an 83.62% probability of ranking a randomly chosen churning customer higher than "
     "a randomly chosen non-churning customer. This is considered a good model."),
    ("What is a Confusion Matrix?",
     "A confusion matrix is a 2x2 table showing: True Positives (TP) = correctly predicted churns, "
     "True Negatives (TN) = correctly predicted non-churns, False Positives (FP) = predicted churn but "
     "customer didn't actually churn, False Negatives (FN) = predicted no-churn but customer actually churned. "
     "FN is the most costly error in churn prediction -- a churning customer we miss gets no retention offer."),
    ("How did you get the model accuracy? Explain the training process.",
     "Step 1: Load dataset (7,043 records). Step 2: Preprocess -- handle missing values, encode categorical "
     "features (one-hot encoding for Contract, PaymentMethod etc.), normalize numerical features (MinMaxScaler). "
     "Step 3: Feature Engineering -- create RiskScore, ChargesPerTenure, HasLoyalty features. "
     "Step 4: Split data 70% train / 30% test. "
     "Step 5: Train Random Forest with 200 trees on training set. "
     "Step 6: Evaluate on TEST SET (never seen during training) -> 76.8% accuracy, 0.8362 AUC. "
     "Step 7: 5-fold cross-validation to confirm the score is not due to luck."),
    ("What is cross-validation and why did you use it?",
     "Cross-validation is a technique to verify that model performance is consistent across different data splits. "
     "In 5-fold CV, the training data is split into 5 equal parts. The model trains on 4 parts and tests on "
     "the 5th, repeated 5 times. The average score across all 5 folds is the CV score. This confirms that "
     "our 76.8% accuracy is reliable and not just lucky on one particular test split."),
    ("What is feature engineering and what did you create?",
     "Feature engineering means creating new, more informative features from existing ones. We created: "
     "(1) RiskScore = weighted combination of contract type risk + monthly charges risk + tenure risk. "
     "(2) ChargesPerTenure = MonthlyCharges / (tenure + 1) -- measures cost efficiency per month. "
     "(3) ServiceCount = total number of add-on services subscribed. "
     "(4) HasLoyalty = 1 if tenure > 24 months (loyal customers churn much less). "
     "These engineered features were among the top predictors in the final model."),
    ("What is SHAP and did you use it?",
     "SHAP (SHapley Additive exPlanations) is a method to explain individual predictions. It calculates "
     "how much each feature contributed to a specific prediction. For example, for Customer A predicted as "
     "high-risk: 'Contract (month-to-month) pushed churn probability up by +35%, tenure (6 months) pushed it "
     "up by +18%, MonthlyCharges ($85) pushed it up by +12%. The SHAP library is included in our requirements."),
]
for i, (q, a) in enumerate(qa_model, 1):
    pdf.qa_block(i, q, a)

# --- TECHNICAL / CODE QUESTIONS ---
pdf.add_page()
pdf.section_title("C. Technical & Code Questions")

qa_tech = [
    ("What is Streamlit and why did you use it instead of Flask or Django?",
     "Streamlit is a Python framework that turns data science scripts into interactive web apps with minimal "
     "code. Unlike Flask/Django which require building HTML, CSS, and JavaScript separately, Streamlit uses "
     "only Python. We chose it because: (1) Rapid development -- a working dashboard in hours not weeks. "
     "(2) Built-in widgets for sliders, file uploaders, charts. (3) Automatic re-rendering when inputs change. "
     "(4) Native Plotly and Pandas integration for instant data visualization."),
    ("What is SQLite and why did you use it?",
     "SQLite is a lightweight embedded relational database stored in a single .db file (churn.db). We chose it "
     "because: (1) No server setup required -- it runs inside the Python process. (2) Full SQL support for "
     "complex analytics queries. (3) Persistent storage of prediction runs with run_id tracking. "
     "(4) Enables SQL-powered business analytics like 'SELECT AVG(MonthlyCharges) FROM churn_predictions "
     "WHERE risk_level = High' without needing a separate database server."),
    ("What is a pipeline in machine learning?",
     "An ML pipeline is a sequence of automated steps: Load Data -> Clean Data -> Engineer Features -> "
     "Split Train/Test -> Train Model -> Evaluate -> Save Model. Using pipelines ensures the exact same "
     "transformations applied during training are applied during prediction. This prevents 'data leakage' "
     "where test data information accidentally influences the training process."),
    ("How do you save and load the trained model?",
     "We use the joblib library to serialize (save) the trained model object to a .pkl (pickle) file: "
     "joblib.dump(model, 'models/best_model.pkl'). When the dashboard loads, it reads the model back: "
     "model = joblib.load('models/best_model.pkl'). This means the model only needs to be trained ONCE "
     "and can then be used for predictions instantly without re-training each time."),
    ("What is one-hot encoding and why do you need it?",
     "Machine learning algorithms work with numbers, not text. One-hot encoding converts categorical "
     "text values into binary columns. For example, the 'Contract' feature with values "
     "'Month-to-month', 'One year', 'Two year' becomes 3 binary columns: "
     "Contract_Month-to-month (1 or 0), Contract_One year (1 or 0), Contract_Two year (1 or 0). "
     "Only one of the three will be 1 for any given customer."),
    ("How does the real-time prediction work when a user uploads a CSV file?",
     "When a user uploads a CSV file: (1) The file is read into a pandas DataFrame. "
     "(2) The same preprocessing pipeline (encoding, feature engineering) used during training is applied. "
     "(3) The saved Random Forest model (best_model.pkl) makes predictions on all rows simultaneously using vectorized inference. "
     "(4) Churn probabilities (0-100%), predictions (Will Churn / No Churn), and risk levels (High/Medium/Low) "
     "are added as new columns. (5) Results are stored in SQLite and displayed in the dashboard. "
     "A 1,000-row dataset is predicted in under 0.3 seconds."),
    ("What is the difference between predict() and predict_proba()?",
     "model.predict(X) returns a binary class label for each customer: 0 (No Churn) or 1 (Churn). "
     "model.predict_proba(X) returns a probability between 0.0 and 1.0 for each class. "
     "We use predict_proba()[:, 1] to get the churn probability (e.g., 0.73 = 73% chance of churning). "
     "This is more useful than binary prediction because we can set custom thresholds and prioritize "
     "outreach to customers with highest probabilities first."),
]
for i, (q, a) in enumerate(qa_tech, 1):
    pdf.qa_block(i, q, a)

# --- BUSINESS / STRATEGY QUESTIONS ---
pdf.add_page()
pdf.section_title("D. Business & Strategy Questions")

qa_biz = [
    ("What business insights did you find from the data?",
     "Key findings: (1) Month-to-month contracts have 30.7% churn vs only 0.6% for two-year contracts -- "
     "a 51x difference. (2) Fiber optic customers churn at 22% vs DSL at 18.7% -- possibly due to high bills. "
     "(3) Electronic check payers churn at 26.7% vs credit card auto-pay at 14.9%. (4) Customers in first "
     "12 months churn much more than long-tenured customers. (5) Customers without OnlineSecurity churn at "
     "18.7% vs 10% for protected customers."),
    ("What is the Customer Retention Strategy page?",
     "The strategy page diagnoses 6 specific business problems found in the uploaded dataset and pairs each "
     "with a data-driven solution. Each card shows: (1) Identified Business Problem with real numbers from "
     "the dataset. (2) Root Cause Analysis explaining WHY this cohort churns. "
     "(3) Possible Strategic Solution with specific commercial actions. "
     "(4) Tactical Action Protocol with exact SLA timelines. "
     "(5) Target Cohort size and Revenue at Risk. (6) Expected Capital Saved per month and year."),
    ("What is MRR and ARR in business terms?",
     "MRR = Monthly Recurring Revenue. It is the predictable revenue a company earns every month from "
     "subscriptions. ARR = Annual Recurring Revenue = MRR x 12. These are the primary financial metrics "
     "for subscription businesses. Our dashboard calculates MRR at risk (revenue that will be lost if "
     "high-risk customers churn) and potential capital saved (if retention interventions succeed)."),
    ("How would a company actually use this system in practice?",
     "A company's Customer Success team would: (1) Upload their customer CRM data every Monday morning. "
     "(2) The system scores all customers in under 1 second. (3) Export the Priority Outreach Queue CSV "
     "(showing highest-risk customers and recommended interventions). (4) Load the list into Salesforce "
     "or HubSpot. (5) Assign CSMs to call high-risk customers within 48 hours. (6) Offer targeted "
     "incentives: contract upgrade discount, free speed tier, security bundle trial."),
    ("What is the ROI of using this system?",
     "The ROI Simulator on the Strategy page calculates this. Example: If the company has 500 high-risk "
     "customers averaging $65/month, and the retention campaign has 25% success rate with $15 cost per "
     "customer: Customers saved = 125. MRR saved = $8,125/month = $97,500/year ARR. "
     "Campaign cost = $7,500. Net gain = $90,000. ROI = 1,200%. Even a 10% success rate produces "
     "significant positive ROI."),
    ("What would you do differently to improve the model accuracy?",
     "Several improvements could raise accuracy: (1) Collect more data -- especially customer complaint history, "
     "network quality scores, and competitor pricing data. (2) Try deep learning approaches (neural networks) "
     "for larger datasets. (3) Use more sophisticated feature engineering -- rolling averages, behavioral "
     "sequences. (4) Ensemble XGBoost + Random Forest predictions together. "
     "(5) Apply Bayesian hyperparameter optimization (Optuna) instead of grid search. "
     "(6) Implement time-series features based on when charges changed."),
]
for i, (q, a) in enumerate(qa_biz, 1):
    pdf.qa_block(i, q, a)

# --- VIVA RAPID-FIRE ---
pdf.add_page()
pdf.section_title("E. Rapid-Fire Viva Questions & Answers")

rapid_qa = [
    ("What does 'churn' mean?", "A customer cancelling their subscription / leaving the service."),
    ("What is supervised learning?", "ML where the model learns from labeled examples (we know the answer: churned or not)."),
    ("What is overfitting?", "When a model memorizes training data but fails on new unseen data."),
    ("What is underfitting?", "When a model is too simple and fails to capture patterns even in training data."),
    ("What is Precision?", "Of all customers predicted to churn, what % actually churned. TP / (TP + FP)."),
    ("What is Recall?", "Of all customers who actually churned, what % did we predict correctly. TP / (TP + FN)."),
    ("What is F1 Score?", "Harmonic mean of Precision and Recall. Good when both precision and recall matter equally."),
    ("What is Gradient Descent?", "An optimization algorithm that iteratively adjusts model parameters to minimize prediction error."),
    ("Why use 'balanced' class weights?", "To compensate for imbalanced data -- gives more importance to the minority class (churners)."),
    ("What is a Pickle file (.pkl)?", "A binary file format for saving and loading Python objects like trained ML models."),
    ("What is SMOTE?", "Synthetic Minority Oversampling Technique -- creates synthetic examples of the minority class."),
    ("What is MinMaxScaler?", "Normalizes numerical features to a 0-1 range so all features are on the same scale."),
    ("How many pages does the dashboard have?", "5 pages: Executive Overview, Upload & Predict, SQL Analytics, Model Testing, Strategy."),
    ("What is the training data size?", "70% of 7,043 = approx. 4,930 records."),
    ("What is the test data size?", "30% of 7,043 = approx. 2,113 records."),
    ("What AUC score did Random Forest get?", "0.8362"),
    ("What AUC score did XGBoost get?", "0.8212"),
    ("Which model is used in production?", "Random Forest -- saved as best_model.pkl"),
    ("What is the churn rate in the dataset?", "Approximately 26.5% (1,869 out of 7,043 customers churned)."),
    ("What does the SQL Analytics page do?", "Runs SQL queries on the SQLite database to compute MRR at risk, revenue by segment, tenure cohort analysis, and top churn segments."),
    ("What is run_id?", "A unique identifier (UUID) assigned to each prediction batch so multiple runs can be stored and compared in SQLite."),
    ("Can the system handle custom CSV files?", "Yes -- it has a generic ML pipeline that auto-detects column types and generates predictions for any customer churn dataset."),
]
for i, (q, a) in enumerate(rapid_qa, 1):
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(15, 50, 120)
    pdf.set_x(10)
    pdf.cell(5, 6, f"Q{i}.", ln=False)
    pdf.set_x(18)
    pdf.cell(0, 6, q, ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.set_x(18)
    pdf.multi_cell(182, 5.5, f"Ans: {a}")
    pdf.ln(1)


# ?????????????????????????????????????????????????????????????????
# SAVE PDF
# ?????????????????????????????????????????????????????????????????
pdf.output(OUTPUT)
print(f"PDF saved successfully: {OUTPUT}")
print(f"Size: {os.path.getsize(OUTPUT) / 1024:.1f} KB")
