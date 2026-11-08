# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds an internal Customer Intelligence and Support Copilot using synthetic Instamart data.

The project is divided into four parts:

- **Part A:** Data cleaning, SQLite database creation, SQL queries and exploratory data analysis.
- **Part B:** Customer segmentation and lapse prediction.
- **Part C:** Support ticket classification, PII redaction and ticket routing.
- **Part D:** Policy-based Retrieval-Augmented Generation (RAG), refund tools and a ReAct refund agent.

All data is synthetic and fictional. All language-model calls use the supplied offline `mock_llm.py`. No API key, paid service or external model is required.

All monetary values are in Indian Rupees (INR).

## Reproduce

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd instamart-copilot
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Generate the synthetic data

```powershell
python generate_data.py
```

### 5. Run Part A

```powershell
python audit_raw_data.py
python clean_customers.py
python inspect_products.py
python clean_products.py
python inspect_orders.py
python check_duplicate_orders.py
python check_order_status.py
python check_order_amounts.py
python clean_orders.py
python inspect_tickets.py
python clean_tickets.py
python create_database.py
python verify_clean_data.py
python run_sql_queries.py
python part_a_eda.py
```

### 6. Run Part B

```powershell
python part_b_rfm.py
python part_b_clustering.py
python part_b_final_segments.py
python part_b_lapse_dataset.py
python part_b_model_training.py
python part_b_train_models.py
python part_b_oversampling.py
python part_b_feature_importance.py
python part_b_save_model.py
```

### 7. Run Part C

```powershell
python part_c_guardrails.py
python part_c_prompts.py
python part_c_retry_test.py
python part_c_test_response.py
python part_c_classification.py
python part_c_budget.py
python part_c_final_routing.py
python part_c_verify_final.py
Copy-Item outputs/triage_final.csv outputs/triage_results.csv
```

### 8. Run Part D

```powershell
python part_d_chunking.py
python part_d_retrieval.py
python part_d_rag.py
python part_d_evaluate.py
python part_d_save_evaluation.py
python part_d_tools.py
python part_d_test_scenarios.py
python part_d_resume_test.py
```

The project uses the supplied `generate_data.py` and `mock_llm.py` files. They should remain unmodified.

## Part A — Data Cleaning, SQLite Database and Analysis

### 1. Raw Data Audit

Inspected the four raw datasets: customers, products, orders and tickets.

The initial audit identified the following data-quality issues:

- **Customers:** 61 missing ages, 6 ages outside the valid range, 24 different city spellings, 10 gender spellings and 30 blank gender values.
- **Products:** 40 product records.
- **Orders:** 13,084 raw rows, 119 exact duplicate rows, 120 missing or non-numeric order amounts, 106 negative delivery times and 2,541 blank delivery times.
- **Tickets:** 164 records, including blank order IDs and blank ticket text.

### 2. Data Cleaning

**Customers**

- Standardized city names by removing extra spaces and applying consistent capitalization.
- Standardized gender values into `male`, `female` and `unknown`.
- Converted age values to numeric.
- Replaced ages outside the valid range of 10–100 with missing values.
- Filled missing ages using the median age of the corresponding city.
- Converted signup dates into datetime format.

Final customer records: **1,200**.

**Products**

- Checked for missing values, duplicate product IDs and invalid prices.
- No cleaning changes were required.

Final product records: **40**.

**Orders**

- Removed 119 exact duplicate rows.
- Converted timestamps and numeric columns into their correct data types.
- Converted negative delivery times into missing values.
- Kept the latest record for each order ID.
- Filled missing order amounts using product price, quantity and discount.
- Filled missing delivery times for completed orders using the median delivery time for the corresponding product category.
- Kept delivery times for cancelled orders as missing because those orders were not delivered.

Final order records: **12,309**.

- Completed: 11,110
- Cancelled: 1,199
- Missing order amounts: 0
- Missing delivery times for completed orders: 0

**Tickets**

- Preserved blank text values and blank order IDs during loading.
- Standardized and cleaned the ticket dataset.
- Saved the cleaned ticket data separately.

Final ticket records: **164**.

### 3. SQLite Database

Created `data/instamart.db` using the cleaned datasets.

The database contains four tables:

- `customers`
- `products`
- `orders`
- `tickets`

Validated the tables and checked the cleaned order records using SQL queries.

### 4. SQL Analysis

Executed seven SQL queries from `sql/queries.sql`.

The analysis included:

- Customer and order analysis.
- Revenue by city and product category.
- Customer spending analysis.
- Ranking customers within each city.
- Running cumulative revenue over time.

Important results:

- Total completed revenue: **₹29,98,668.80**
- Average customer spending: **₹2,552.06**
- Customers spending above the average: **550**
- Q6 ranking reference check: passed.

### 5. Pandas Analysis and Insights

Used Pandas to perform additional analysis, including pivot tables, grouped aggregations, descriptive statistics, outlier detection and correlation analysis.

Key insights:

1. Hyderabad generated the highest completed revenue at ₹5,28,589.90.
2. Bengaluru generated ₹5,23,643.40 in completed revenue.
3. Mumbai had the highest average order value at ₹297.00.
4. Order amounts were positively skewed, with a skewness of approximately 4.61.
5. Discount percentage had a weak negative correlation with order amount (-0.095).

### 6. Validation

Ran the validation script to verify the cleaned datasets and SQLite database.

All nine validation checkpoints passed successfully.

```text
ALL CHECKPOINTS PASSED
```

## Part B — Customer Segmentation and Lapse Prediction

### 1. RFM Analysis and Customer Segmentation

Calculated Recency, Frequency and Monetary (RFM) features for 1,175 customers.

Evaluated K-means clustering for K values from 2 to 8. Selected K=4 based on the highest silhouette score of 0.526.

The four customer segments are:

- Regular Customers: 436
- Premium Customers: 138
- Loyal Customers: 419
- At-Risk Customers: 182

PCA explained 85.2% of the variance using the first two components.

### 2. Customer Lapse Prediction Dataset

Created a lapse prediction dataset containing 1,156 customers, with 443 lapsed customers (38.32%).

Used a stratified 80/20 train-test split:

- Training: 924 customers
- Testing: 232 customers

The baseline model achieved 61.6% accuracy but an F1-score of 0 for the lapsed class.

### 3. Model Comparison

Trained Logistic Regression and Random Forest models using pipelines with StandardScaler and OneHotEncoder preprocessing.

| Model | CV F1 | Test F1 |
|---|---:|---:|
| Logistic Regression | 0.634 | 0.708 |
| Random Forest | 0.669 | 0.717 |

Random Forest achieved the better overall test F1-score among the original models.

### 4. Oversampling

Oversampled the minority class in the training dataset only, balancing the classes at 570 customers each.

The oversampled Logistic Regression model achieved:

- Precision: 0.599
- Recall: 0.921
- F1-score: 0.726

The improved recall helps identify more potentially lapsed customers.

### 5. Feature Importance

The top Random Forest features were:

1. Frequency
2. Monetary
3. Cancellation rate
4. Recency
5. Electronics category

Frequency had the highest feature importance (0.3429).

### 6. Model Persistence and Retention List

Saved the final oversampled Logistic Regression pipeline using joblib.

Reloaded the model and verified that its predictions matched the original model predictions.

Generated a retention list containing the top 10 Premium customers from the test set, ranked by predicted lapse probability.

**Limitations:**

- The Premium segment is derived from the overall RFM analysis, so the retention list is illustrative and not a fully leakage-free prospective evaluation.
- Model preprocessing was fitted on training data only.
- September 2025 data was excluded from the lapse prediction features to avoid future-data leakage.
- Segment labels were not used as model features.

## Part C — Support Ticket Classification and Routing

### 1. Objective

Built a rule-guarded ticket classification pipeline using the provided mock LLM.

The pipeline classifies customer support tickets into four categories:

- Billing
- Delivery
- Technical
- Fraud

### 2. Input Data

- Raw input: `data/raw/tickets.csv`
- Total tickets: 164
- Allowed categories: billing, delivery, technical, fraud

### 3. Guardrails and PII Redaction

Before sending tickets to the model, the pipeline applies the following checks:

- Empty or whitespace-only tickets are marked `skipped_empty`.
- Tickets exceeding 400 estimated tokens are marked `flagged_long`.
- Empty and long tickets are never sent to the model.
- Email addresses are replaced with `[EMAIL]`.
- Phone numbers are replaced with `[PHONE]`.
- Order IDs are replaced with `[ORDER_ID]`.
- Only redacted ticket text is sent as the user message.

Guardrail results:

| Check | Count |
|---|---:|
| Total tickets | 164 |
| Ready for classification | 156 |
| Empty tickets | 4 |
| Long tickets | 4 |
| Emails redacted | 55 |
| Phone numbers redacted | 91 |
| Order IDs redacted | 132 |
| Total PII replacements | 278 |

No matching PII patterns remained after redaction.

### 4. Prompt Design

Two prompts were created and validated:

- Zero-shot prompt: classification instructions without examples.
- Few-shot prompt: classification instructions with four labelled examples, including fraud.

Both prompts contain the four allowed categories and require JSON-only responses.

### 5. Retry Handling

Implemented retry handling for transient model errors.

- Maximum retries: 3 per ticket.
- Base delay: 0.05 seconds.
- Exponential backoff with random jitter.
- One retry log is printed for each retry attempt.

All 19 transient errors were recovered successfully in each experiment.

### 6. Classification and Response Validation

Model responses are processed using the following rules:

1. Valid JSON with an allowed category is marked `classified_json`.
2. Invalid JSON is checked for exactly one allowed category keyword and marked `classified_keyword` when successful.
3. Invalid JSON without exactly one allowed category is marked `flagged_invalid_output`.
4. Valid JSON containing an unsupported category is flagged without keyword fallback.

Results:

| Status | Tickets |
|---|---:|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |
| Total | 164 |

### 7. Zero-shot vs Few-shot Comparison

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy | 85.71% | 100% |
| Total tokens | 16,420 | 26,560 |
| Tokens per classified ticket | 111.70 | 180.68 |

Few-shot classification improved accuracy by 14.29 percentage points but used 10,140 additional tokens, approximately 61.8% more than zero-shot.

The zero-shot confusion table showed 11 fraud tickets and 10 technical tickets incorrectly classified as billing. Few-shot classification correctly classified all 147 classified tickets.

### 8. Token Budget Experiment

A zero-shot classification experiment was run with a 6,000-token budget.

| Metric | Result |
|---|---:|
| Configured budget | 6,000 |
| Actual tokens used | 6,095 |
| Classified JSON | 51 |
| Classified keyword | 4 |
| Invalid output | 3 |
| Budget exhausted | 98 |
| Empty tickets skipped | 4 |
| Long tickets flagged | 4 |

The budget is checked before every model call. The final permitted call can exceed the configured budget because the script checks the running total before making the call.

### 9. Final Ticket Routing

The final few-shot classification run used a 100,000-token budget.

Routing rules:

- Fraud tickets: `security_team`, HIGH.
- Premium customers with valid non-fraud classifications: `premium_desk`, HIGH.
- Other valid classifications: `support_queue`, NORMAL.
- Empty, long, invalid-output, or budget-exhausted tickets: `human_review`, NORMAL.

Final routing results:

| Route | Tickets |
|---|---:|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |
| Total | 164 |

Priority distribution:

| Priority | Tickets |
|---|---:|
| HIGH | 45 |
| NORMAL | 119 |

Total tokens used: 26,560 out of the 100,000-token budget.

### 10. Output Files and Verification

Important output files:

- `outputs/tickets_guardrails.csv`
- `outputs/triage_zero_shot.csv`
- `outputs/triage_few_shot.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_final.csv`
- `outputs/triage_results.csv`

Created `part_c_verify_final.py` to verify:

- All 164 ticket records are present and unique.
- All eight required output columns exist.
- Fraud tickets are routed to the security team.
- Premium customers are routed correctly.
- Human-review statuses are routed correctly.
- Support-queue tickets have NORMAL priority.

All final verification checks passed.

## Part D — Policy RAG Assistant and Refund Agent

### 1. Policy Document Chunking

Loaded the five policy documents from `data/policies/`.

Split the documents into chunks using:

- Chunk size: 40 words
- Overlap: 10 words
- Step size: 30 words

Generated 15 policy chunks in total.

| Policy document | Chunks |
|---|---:|
| refund_policy.txt | 4 |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| **Total** | **15** |

Saved the chunked documents in `outputs/policy_chunks.csv`.

### 2. TF-IDF Retrieval

Implemented document retrieval using `TfidfVectorizer()` and cosine similarity.

The `retrieve(query, k=3)` function returns the top three matching chunks with their source document, similarity score and text.

Tested the query:

`refund policy late delivery`

All three top results came from `refund_policy.txt`.

TF-IDF assigns importance to words based on their frequency in a document and how uncommon they are across the document collection. Cosine similarity compares the query vector with the document chunk vectors.

**Limitation:** TF-IDF is based on word matching and does not understand meaning or context like a semantic embedding model.

### 3. RAG Answer Generation

Implemented the `answer(question)` function.

The function:

1. Retrieves the top three policy chunks.
2. Combines the retrieved context while handling overlapping text.
3. Sends the context and question to the supplied offline mock LLM.
4. Returns the generated answer and source documents.

The system prompt instructs the assistant to answer only from the supplied policy context and to say when the information cannot be found.

### 4. RAG Evaluation

Evaluated the RAG system using 14 questions from `data/raw/policy_eval.csv`.

The evaluation included 12 answerable questions and 2 questions that were not covered by the policy documents.

| Configuration | Document retrieval | Expected facts | Unanswerable fallback |
|---|---:|---:|---:|
| 10-word overlap | 12/12 | 12/12 | 2/2 |
| Zero overlap | 12/12 | 11/12 | 2/2 |

With 10-word overlap, all 12 answerable questions retrieved the expected document and returned the expected fact.

With zero overlap, one expected fact was missed:

`How soon is money returned when a payment fails but is deducted?`

The expected fact `48 hours` was missed because the relevant information was split across a chunk boundary.

Both unanswerable questions correctly triggered the fallback response.

Saved the evaluation summary in `outputs/policy_evaluation_summary.csv`.

### 5. Tool Definitions and Database Integration

Implemented five tools:

| Tool | Purpose |
|---|---|
| `get_order` | Retrieves order details from the SQLite database. |
| `search_policy` | Searches the policy documents. |
| `calculate_refund` | Calculates refund eligibility and amount. |
| `issue_refund` | Records an approved refund. |
| `escalate_to_human` | Records a refund request requiring human approval. |

The `get_order` tool retrieves order details from `data/instamart.db`.

The refund calculation follows the policy:

- Only completed orders are eligible.
- Orders delivered in more than 45 minutes qualify for a full refund.
- Orders delivered in 45 minutes or less do not qualify.
- Refunds up to INR 1000 can be approved automatically.
- Refunds above INR 1000 require human approval.

Refund and escalation actions are recorded in `outputs/refund_ledger.jsonl`.

### 6. ReAct Refund Agent

Implemented a ReAct agent using the supplied offline mock LLM.

The agent:

1. Receives a customer refund request.
2. Retrieves the order details.
3. Searches the relevant policy.
4. Calculates the refund amount.
5. Issues a refund or escalates the request.
6. Returns a final status and answer.

The agent supports a maximum iteration limit, token limit and time limit.

### 7. Scenario Evaluation

Tested the agent using all seven scenarios from `data/raw/agent_scenarios.csv`.

| Scenario | Status | Refund amount (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 958 |
| S2 | REFUND | 135.00 | 5 | 958 |
| S3 | ESCALATE | 1999.00 | 5 | 945 |
| S4 | DENIED | 0 | 4 | 642 |
| S5 | DENIED | 0 | 3 | 359 |
| S6 | NOT_FOUND | 0 | 2 | 85 |
| S7 | FALLBACK_MAX_ITERATIONS | 0 | 6 | 3084 |

All seven scenarios produced their expected outcomes, including the maximum-iteration fallback for S7.

Saved the scenario results and tool execution details in `outputs/agent_runs.jsonl`.

### 8. Safety and Fallback Tests

Tested the agent's safety limits.

| Test | Configuration | Result |
|---|---|---|
| Token limit | 200 tokens | FALLBACK_TOKEN_LIMIT |
| Time limit | 0 seconds | FALLBACK_TIMEOUT |
| Maximum iterations | 6 iterations | FALLBACK_MAX_ITERATIONS |

The token-limit test stopped the agent before executing the tool call returned by the over-budget model response.

The timeout test stopped before making the first model call.

The maximum-iteration test stopped S7 after six model calls and returned a fallback handoff summary.

### 9. Checkpoint and Resume Test

Implemented checkpoint saving after every tool observation.

Checkpoint files are stored in the `state/` directory.

Tested crash recovery using scenario S1:

1. Started the agent with S1.
2. Simulated a crash immediately after the second tool execution.
3. Confirmed that the checkpoint was saved.
4. Resumed the agent using `resume_agent(run_id)`.
5. Completed the refund successfully.

The final result was **REFUND INR 128.25**.

The complete tool execution sequence across the crash and resume was:

`get_order → search_policy → calculate_refund → issue_refund`

No tool was executed twice.

### 10. ReAct vs Plan-and-Execute

The ReAct approach was selected because it allows the agent to make decisions based on the observations returned by each tool.

The measured model calls varied by scenario: S1 required five calls, S5 required three calls and S6 required two calls. This avoids forcing every request through a fixed number of steps.

ReAct also handles unexpected results, such as missing orders and cancelled orders, by using the observations before deciding what to do next.

A Plan-and-Execute approach could be useful for larger workflows with predictable steps. However, for this refund workflow, ReAct is more suitable because the next action depends on order status, policy eligibility and refund amount.

Iteration, token and time limits provide safeguards against excessive model calls and repeated tool execution.

### 11. Part D Output Files

- `outputs/policy_chunks.csv`
- `outputs/policy_evaluation_summary.csv`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `state/S1.json` and other scenario checkpoints

## Limitations and Next Steps

- All data is synthetic and does not represent actual Instamart customers, transactions or policies.
- The supplied mock LLM is an offline simulation and does not represent the capabilities of a production language model.
- TF-IDF retrieval relies on word overlap and may miss semantically related information.
- The lapse model is trained and evaluated on a synthetic dataset; its performance may differ on real customer data.
- The Premium retention list is illustrative because the segment information is derived from the overall RFM analysis.
- The refund agent demonstrates policy-based decision-making and tool execution but is not connected to a real payment system.
- A production implementation would require stronger authentication, authorization, audit logging, human approval workflows and monitoring.

Future improvements could include semantic vector retrieval, real-time order integration, more extensive model evaluation and deployment behind a secure support dashboard.