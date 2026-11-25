# Deployment Guide: Online Recruitment Fraud Detection System

This directory contains the production-ready **Streamlit web application** for real-time recruitment fraud detection using the fine-tuned Transformer model developed in `Recruitment_Fraud_Detection.ipynb`.

---

## 1. Directory Structure

```text
deployment/
├── app.py                # Streamlit web application interface
├── requirements.txt      # Production runtime dependencies
└── README.md             # Deployment and usage instructions
```

---

## 2. Architecture Overview

- **Input:** User submits job details (Job Title, Company Profile, Description, Requirements, Benefits, etc.).
- **Preprocessing:** Identical text concatenation and formatting as Step 3 in the notebook.
- **Model Inference:** Loaded via Hugging Face Transformers (`AutoModelForSequenceClassification` and `AutoTokenizer`) cached using Streamlit's `@st.cache_resource`.
- **Output:** Genuine (0) vs. Fraudulent (1) prediction, exact class probabilities, and risk tier (Low, Moderate, High).
- **Zero Training Code:** The deployment application strictly performs inference; it does not contain training loops, dataset loading for training, or evaluation code.

---

## 3. Local Execution

To run the Streamlit app locally on your machine:

1. **Open a terminal** and navigate to the `deployment/` directory:
   ```bash
   cd "d:\STUDY\ML\LAB\ass 4\Recruitment_Fraud_Detection\deployment"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model checkpoint exists**:
   - By default, `app.py` looks for the saved model at `../saved_models/best_model` (exported during Step 11 of the notebook).
   - Alternatively, you can enter any local path or your Hugging Face Hub repository in the sidebar.

4. **Launch Streamlit**:
   ```bash
   streamlit run app.py
   ```

5. Open your web browser at `http://localhost:8501`.

---

## 4. Deploying to Streamlit Community Cloud

Follow these steps to deploy your trained project to the web for free:

### Step A: Push Model to Hugging Face Hub
In Step 11 of the notebook (`Recruitment_Fraud_Detection.ipynb`), run the export cell to push your fine-tuned model and tokenizer to your Hugging Face account:
```python
model.push_to_hub("your-hf-username/recruitment-fraud-detector")
tokenizer.push_to_hub("your-hf-username/recruitment-fraud-detector")
```

### Step B: Push Repository to GitHub
1. Initialize git in the root `Recruitment_Fraud_Detection/` directory:
   ```bash
   git init
   git add deployment/ README.md
   git commit -m "Initial commit of Recruitment Fraud Detection deployment app"
   ```
2. Create a new GitHub repository (e.g. `recruitment-fraud-detection`) and push your code:
   ```bash
   git remote add origin https://github.com/<your-username>/recruitment-fraud-detection.git
   git branch -M main
   git push -u origin main
   ```

### Step C: Deploy on Streamlit Cloud
1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub account.
2. Click **New app**.
3. Select your repository: `<your-username>/recruitment-fraud-detection`.
4. Set **Main file path** to `deployment/app.py`.
5. Under **Advanced settings**, set any necessary secrets (such as `HF_TOKEN` if using a private Hugging Face repo).
6. Click **Deploy!**

Your web app will be live and accessible to anyone via a public URL to demonstrate to university evaluators and peers.
