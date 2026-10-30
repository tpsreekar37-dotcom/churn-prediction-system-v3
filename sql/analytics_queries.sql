-- ============================================================
-- Customer Churn Prediction System – SQL Analytics Queries
-- Database: SQLite (churn.db)
-- ============================================================

-- ────────────────────────────────────────────────────────────
-- QUERY 1: Overall Churn Summary
-- ────────────────────────────────────────────────────────────
SELECT
    SUM(Churn)                                         AS total_churned,
    COUNT(*) - SUM(Churn)                              AS total_retained,
    COUNT(*)                                           AS total_customers,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)            AS churn_rate_pct
FROM customers;

-- ────────────────────────────────────────────────────────────
-- QUERY 2: High-Risk Customers (churn=1, high monthly charges)
-- ────────────────────────────────────────────────────────────
SELECT
    rowid              AS customer_id,
    tenure,
    MonthlyCharges,
    TotalCharges,
    Contract,
    PaymentMethod,
    InternetService,
    Churn
FROM customers
WHERE Churn = 1
  AND MonthlyCharges > (SELECT AVG(MonthlyCharges) FROM customers)
ORDER BY MonthlyCharges DESC
LIMIT 50;

-- ────────────────────────────────────────────────────────────
-- QUERY 3: Churn Rate by Payment Method
-- ────────────────────────────────────────────────────────────
SELECT
    PaymentMethod,
    COUNT(*)                                           AS total_customers,
    SUM(Churn)                                         AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)            AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                      AS avg_monthly_charges
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- ────────────────────────────────────────────────────────────
-- QUERY 4: Churn Rate by Contract Type
-- ────────────────────────────────────────────────────────────
SELECT
    Contract,
    COUNT(*)                                           AS total_customers,
    SUM(Churn)                                         AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)            AS churn_rate_pct,
    ROUND(AVG(tenure), 1)                              AS avg_tenure_months,
    ROUND(AVG(MonthlyCharges), 2)                      AS avg_monthly_charges
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- ────────────────────────────────────────────────────────────
-- QUERY 5: Average Monthly Revenue by Segment
-- ────────────────────────────────────────────────────────────
SELECT
    CASE WHEN Churn = 1 THEN 'Churned' ELSE 'Retained' END AS segment,
    COUNT(*)                                                AS customers,
    ROUND(AVG(MonthlyCharges), 2)                           AS avg_monthly_revenue,
    ROUND(SUM(MonthlyCharges), 2)                           AS total_monthly_revenue,
    ROUND(AVG(TotalCharges), 2)                             AS avg_lifetime_value
FROM customers
GROUP BY Churn;

-- ────────────────────────────────────────────────────────────
-- QUERY 6: Top Churn Segments (Contract + Internet Service)
-- ────────────────────────────────────────────────────────────
SELECT
    Contract,
    InternetService,
    COUNT(*)                                           AS total,
    SUM(Churn)                                         AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)            AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                      AS avg_charges
FROM customers
GROUP BY Contract, InternetService
ORDER BY churn_rate_pct DESC
LIMIT 10;

-- ────────────────────────────────────────────────────────────
-- QUERY 7: Revenue at Risk (Churned Customers Monthly Revenue)
-- ────────────────────────────────────────────────────────────
SELECT
    ROUND(SUM(MonthlyCharges), 2)                      AS monthly_revenue_at_risk,
    ROUND(AVG(MonthlyCharges), 2)                      AS avg_monthly_charges_churned,
    COUNT(*)                                           AS churned_customers
FROM customers
WHERE Churn = 1;

-- ────────────────────────────────────────────────────────────
-- QUERY 8: Tenure Cohort Churn Analysis
-- ────────────────────────────────────────────────────────────
SELECT
    CASE
        WHEN tenure <= 12 THEN '0-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_cohort,
    COUNT(*)                                           AS total_customers,
    SUM(Churn)                                         AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)            AS churn_rate_pct
FROM customers
GROUP BY tenure_cohort
ORDER BY MIN(tenure);

-- ────────────────────────────────────────────────────────────
-- QUERY 9: Senior Citizen Churn Analysis
-- ────────────────────────────────────────────────────────────
SELECT
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS customer_type,
    COUNT(*)                                           AS total,
    SUM(Churn)                                         AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)            AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                      AS avg_charges
FROM customers
GROUP BY SeniorCitizen;

-- ────────────────────────────────────────────────────────────
-- QUERY 10: Month-to-Month Electronic Check High Risk
-- ────────────────────────────────────────────────────────────
SELECT
    COUNT(*)                                           AS high_risk_count,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)            AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                      AS avg_charges,
    ROUND(SUM(MonthlyCharges), 2)                      AS total_monthly_revenue
FROM customers
WHERE Contract = 0  -- Month-to-month encoded
   OR PaymentMethod = 0  -- Electronic check encoded
ORDER BY churn_rate_pct DESC;
