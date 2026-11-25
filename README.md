# Online Recruitment Fraud Detection System Using Pretrained Transformer Models

An end-to-end Machine Learning, Natural Language Processing (NLP), and Deep Learning academic project developed for BS Software Engineering students. This project builds a binary text classification system that predicts whether an online job posting is **Genuine (0)** or **Fraudulent (1)** by fine-tuning and comparing two state-of-the-art Transformer architectures: **ModernBERT-base** and **RoBERTa-base**.

---

## 1. Project Objective

With the rapid expansion of remote employment, online recruitment scams have proliferated. Malicious actors create deceitful job postings to harvest personal data, compromise banking information, or solicit fraudulent application/licensing fees.

The primary objective of this project is to:
1. Utilize the **Kaggle Employment Scam Aegean Dataset (EMSCAD)** (17,880 job records).
2. Clean, process, and structure multi-attribute job advertisements without destroying subtle linguistic scam markers.
3. Address severe class imbalance (~95% Genuine vs. ~5% Fraudulent) using stratified data splitting and class-weighted cross-entropy loss.
4. Fine-tune two base Transformer models sequentially:
   - **ModernBERT-base** (`answerdotai/ModernBERT-base`)
   - **RoBERTa-base** (`FacebookAI/roberta-base`)
5. Evaluate both models on an untouched held-out testing set using Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
6. Programmatically select the superior model based on validation PR-AUC and deploy it via a standalone **Streamlit** web application.

---

## 2. Pretrained Transformer Architectures

| Architecture | Hugging Face Model ID | Key Attributes |
|---|---|---|
| **ModernBERT-base** | `answerdotai/ModernBERT-base` | Modernized BERT encoder featuring rotary positional embeddings (RoPE), flash attention concepts, GeGLU activations, and extended context window efficiency. |
| **RoBERTa-base** | `FacebookAI/roberta-base` | Robustly optimized BERT approach trained without next-sentence prediction, utilizing dynamic masking and Byte-Pair Encoding (BPE). |

> **Academic Rigor Notice:** Both models are initialized from their original pretrained base checkpoints with fresh binary classification heads and fine-tuned strictly on our dataset. No externally fine-tuned recruitment fraud weights or fabricated evaluation metrics are used.

---

## 3. Project Directory Structure

```text
Recruitment_Fraud_Detection/
│
├── Recruitment_Fraud_Detection.ipynb   # Complete 14-Step Educational Lab Notebook
│
├── README.md                           # Comprehensive Academic Project Documentation
│
└── deployment/
    ├── app.py                          # Standalone Streamlit Web Inference Application
    ├── requirements.txt                # Production Dependencies for Web Deployment
    └── README.md                       # Deployment & Streamlit Cloud Guide
```

---

## 4. Notebook Architecture (14 Educational Steps)

The core laboratory notebook `Recruitment_Fraud_Detection.ipynb` is structured as an interactive tutorial. Every single substep contains:
- **What are we doing?**
- **Why are we doing it?**
- **How does it work?**
- **Expected output.**
Followed by dedicated, modular code cells with error-free syntax.

### Step-by-Step Breakdown:
1. **STEP 1: IMPORT LIBRARIES**
   - *1.1:* Install required packages (Transformers, PyTorch, Accelerate, Scikit-learn).
   - *1.2:* Import modules and inspect versions.
   - *1.3:* Set random seed (42) for deterministic reproducibility.
   - *1.4:* Detect and configure CUDA GPU accelerator (or CPU fallback).
2. **STEP 2: LOAD DATASET**
   - *2.1:* Load EMSCAD dataset with multiple fallback options (local CSV, Colab upload, synthetic generator).
   - *2.2:* View dataset dimensions (17,880 rows, 18 columns) and sample rows.
   - *2.3:* Analyze column semantics (title, company_profile, description, requirements, etc.).
   - *2.4:* Identify missing values and duplicate records.
   - *2.5:* Calculate target class frequencies and render distribution bar chart.
   - *2.6:* Formally demonstrate the **Accuracy Paradox** and why PR-AUC / F1 matter.
3. **STEP 3: DATA CLEANING AND PREPROCESSING**
   - *3.1:* Select 10 core text and metadata features.
   - *3.2:* Audit missing values in text features.
   - *3.3:* Impute missing text with empty strings (`""`).
   - *3.4:* Remove duplicate postings to eliminate data leakage.
   - *3.5:* Concatenate features into structured text with explicit section headers.
   - *3.6:* Inspect genuine vs. fraudulent combined text samples.
   - *3.7:* Perform conservative text cleaning (preserving punctuation, numbers, currency `$`, and URLs).
   - *3.8:* Format target labels strictly as integers (`0` = Genuine, `1` = Fraudulent).
   - *3.9:* Compute word count percentiles.
   - *3.10:* Export preprocessed data as `clean_recruitment_fraud_dataset.csv`.
4. **STEP 4: SPLIT DATA INTO TRAINING, VALIDATION, AND TESTING SETS**
   - *4.1:* Establish the roles of Train (70%), Validation (15%), and Test (15%).
   - *4.2:* Explain the mathematical rationale behind stratified sampling.
   - *4.3:* Execute two-stage stratified split with seed 42.
   - *4.4:* Verify identical class ratios across all three partitions.
5. **STEP 5: TOKENIZATION AND FEATURE PREPARATION**
   - *5.1:* Explain subword tokenization, input IDs, attention masks, and padding.
   - *5.2:* Load ModernBERT tokenizer (`answerdotai/ModernBERT-base`).
   - *5.3:* Test ModernBERT tokenization on sample text.
   - *5.4:* Load RoBERTa tokenizer (`FacebookAI/roberta-base`).
   - *5.5:* Test RoBERTa tokenization on sample text.
   - *5.6:* Construct custom PyTorch `Dataset` and `DataLoader` objects (batch size = 4, max length = 256).
6. **STEP 6: EXECUTE THE TRAINING PHASE**
   - *Sequential & Safe Execution:* Trains one model at a time with automatic GPU memory release.
   - *No Retraining if Saved:* Detects existing checkpoints in `./saved_models/` to bypass redundant training.
   - *ModernBERT (6.1 - 6.7):* Load base model, configure AdamW and linear warm-up, calculate training-only class weights, execute training loop with gradient accumulation (effective batch = 16), validate per epoch, save best PR-AUC checkpoint, plot loss curve, and release GPU memory.
   - *RoBERTa (6.8 - 6.14):* Load base model, configure matched hyperparameters, apply identical weighted loss, execute training loop, validate per epoch, save best checkpoint, plot loss curve, and release GPU memory.
7. **STEP 7: EXECUTE THE TESTING PHASE**
   - *7.1:* Reload best saved ModernBERT and RoBERTa checkpoints.
   - *7.2:* Generate prediction probabilities on the untouched held-out test set.
   - *7.3:* Tune decision threshold using **validation set only** (zero test leakage).
   - *7.4 & 7.5:* Calculate test metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC).
   - *7.6 & 7.7:* Generate and analyze confusion matrices (TP, TN, FP, FN).
   - *7.8:* Print full Scikit-Learn classification reports.
   - *7.9 & 7.10:* Plot comparative ROC and Precision-Recall curves.
   - *7.11:* Compile final testing benchmark table.
8. **STEP 8: COMPARE THE ALGORITHMS AND SELECT THE BEST MODEL**
   - *8.1:* Assemble validation comparison table.
   - *8.2:* Assemble testing comparison table.
   - *8.3:* Analyze the impact of class-weighted loss on minority recall.
   - *8.4:* Generate grouped bar chart comparing fraud precision, recall, F1, and PR-AUC.
   - *8.5:* Programmatically select the winning model based on validation PR-AUC and save to `./saved_models/best_model`.
   - *8.6:* Review detailed classification report of the winning architecture.
9. **STEP 9: EXECUTE THE APPLICATION PHASE**
   - *9.1 & 9.2:* Load winning model and tokenizer from `./saved_models/best_model`.
   - *9.3:* Implement end-to-end `predict_job(job_details)` function.
   - *9.4 & 9.5:* Evaluate 5 realistic dummy job postings (genuine and fraudulent scenarios) with dynamic outputs.
   - *9.6 - 9.8:* Interactive prediction on custom user input.
10. **STEP 10: EXECUTE THE FEEDBACK PHASE**
    - *10.1 & 10.2:* Explain continuous monitoring and build `record_feedback()`.
    - *10.3:* Store feedback entries in `prediction_feedback.csv`.
    - *10.4:* View persistent feedback log.
    - *10.5:* Explain safe model updating (human-curated retraining gate, never automatic).
11. **STEP 11: SAVE AND EXPORT THE BEST MODEL**
    - *11.1 - 11.6:* Export model weights, tokenizer, config.json, label mapping, and preprocessing settings.
    - *11.7:* Provide optional Hugging Face Hub publishing code (guarded by manual execution flag).
12. **STEP 12: CREATE THE STREAMLIT APPLICATION**
    - *12.1 - 12.7:* Review deployment architecture, app.py, requirements.txt, and cloud deployment steps.
13. **STEP 13: LESSONS LEARNED ABOUT IMBALANCED DATA**
    - Detailed discussion on the Accuracy Paradox, stratified splitting, loss weighting, PR-AUC, and the real-world cost of False Negatives.
14. **STEP 14: CONCLUSION**
    - Final academic conclusion summarizing pipeline stages, experimental outcomes, limitations, and future directions.

---

## 5. Execution Guide

### Option A: Running in Google Colab (Recommended for T4 GPU)
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Runtime > Change runtime type** and set **Hardware accelerator** to **T4 GPU**.
3. Upload `Recruitment_Fraud_Detection.ipynb`.
4. Upload `fake_job_postings.csv` (downloaded from [Kaggle EMSCAD](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)).
5. Run the cells step-by-step from Step 1 through Step 14.

### Option B: Running Locally in Jupyter Notebook
1. Navigate to the project directory:
   ```bash
   cd "d:\STUDY\ML\LAB\ass 4\Recruitment_Fraud_Detection"
   ```
2. Install dependencies:
   ```bash
   pip install -r deployment/requirements.txt
   ```
3. Place `fake_job_postings.csv` in the directory.
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Recruitment_Fraud_Detection.ipynb
   ```

### Option C: Running the Streamlit Web Application
1. Navigate to the `deployment/` directory:
   ```bash
   cd deployment
   ```
2. Launch the Streamlit server:
   ```bash
   streamlit run app.py
   ```
3. Open `http://localhost:8501` in your browser.

---

## 6. Smart Checkpoint Handling (No Redundant Retraining)

Training Transformer models requires significant computational resources. To protect against accidental re-execution:
- Step 6 checks whether `./saved_models/modernbert_best` and `./saved_models/roberta_best` exist on disk.
- If checkpoints are detected, the notebook prints a notification and **loads the saved model instantly without retraining**.
- If you explicitly wish to retrain from scratch, simply set `FORCE_RETRAIN_MODERNBERT = True` or `FORCE_RETRAIN_ROBERTA = True`.

---

## 7. Key Lessons on Imbalanced Data

- **Accuracy is Deceptive:** In a 95:5 class split, a model predicting "Genuine" for every single sample achieves 95% accuracy while protecting 0 job seekers from fraud.
- **Stratified Partitioning:** Preserves exact class ratios across Train, Validation, and Test splits, preventing unrepresentative evaluation sets.
- **Class-Weighted Cross-Entropy:** Applying a ~20x higher penalty to fraudulent misclassifications forces the Transformer's self-attention layers to focus on subtle scam markers.
- **Precision-Recall AUC (PR-AUC):** Provides an uninflated metric that evaluates the true trade-off between finding all scams (Recall) and minimizing false alarms (Precision).
- **Validation-Driven Threshold Tuning:** Adjusting decision thresholds using validation data optimizes sensitivity without causing test set data snooping.

---

## 8. Academic Project Disclaimer

This system is an academic research and engineering prototype developed using fine-tuned Transformer models. Predictions are statistical estimates based on historical patterns in the EMSCAD dataset and do not constitute absolute legal or financial judgments. Always exercise due diligence before sharing sensitive personal information or transferring funds.
