import os
import json
import torch
import streamlit as st
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Recruitment Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM STYLING
# ==============================================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        padding: 1.2rem;
        border-radius: 0.5rem;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        padding: 0.9rem;
        border-radius: 0.4rem;
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        color: #92400E;
        font-size: 0.88rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODEL LOADING FUNCTION (CACHED)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_path_or_repo: str):
    """
    Loads pretrained/fine-tuned Transformer model and tokenizer using st.cache_resource.
    Supports either local directory or Hugging Face Hub model identifier.
    """
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_repo)
        
        # Load model for sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(model_path_or_repo)
        model.eval()
        
        # Check for optional saved threshold or config
        threshold = 0.5
        config_path = os.path.join(model_path_or_repo, "model_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    threshold = float(cfg.get("best_threshold", 0.5))
            except Exception:
                pass
                
        return tokenizer, model, threshold, None
    except Exception as e:
        return None, None, 0.5, str(e)


# ==============================================================================
# TEXT PREPROCESSING PIPELINE
# ==============================================================================
def combine_job_text(
    title: str,
    company_profile: str,
    description: str,
    requirements: str,
    benefits: str,
    employment_type: str = "",
    required_experience: str = "",
    required_education: str = "",
    industry: str = "",
    function: str = ""
) -> str:
    """
    Combines job fields into a single structured text string identical to Step 3
    of the Recruitment_Fraud_Detection notebook.
    Preserves numbers, punctuation, currency symbols, and URLs.
    """
    parts = [
        f"Job Title: {title.strip() if title else ''}",
        f"Company Profile: {company_profile.strip() if company_profile else ''}",
        f"Job Description: {description.strip() if description else ''}",
        f"Requirements: {requirements.strip() if requirements else ''}",
        f"Benefits: {benefits.strip() if benefits else ''}",
        f"Employment Type: {employment_type.strip() if employment_type else ''}",
        f"Required Experience: {required_experience.strip() if required_experience else ''}",
        f"Required Education: {required_education.strip() if required_education else ''}",
        f"Industry: {industry.strip() if industry else ''}",
        f"Function: {function.strip() if function else ''}"
    ]
    return "\n".join(parts)


# ==============================================================================
# SIDEBAR: MODEL & SETTINGS CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Model Configuration")
    st.markdown("Select where the fine-tuned Transformer model should be loaded from.")
    
    # Candidate local paths
    default_candidates = [
        os.path.join("..", "saved_models", "best_model"),
        os.path.join(".", "saved_models", "best_model"),
        os.path.join("..", "saved_models", "modernbert_best"),
        os.path.join("..", "saved_models", "roberta_best"),
        "answerdotai/ModernBERT-base",
        "FacebookAI/roberta-base"
    ]
    
    # Pick first valid local directory if exists, else first entry
    initial_default = default_candidates[0]
    for c in default_candidates[:4]:
        if os.path.exists(c):
            initial_default = c
            break

    model_source = st.text_input(
        "Model Path or Hugging Face Repo ID:",
        value=initial_default,
        help="Provide local folder path (e.g., ../saved_models/best_model) or your Hugging Face model repository (e.g., your-username/recruitment-fraud-detector)."
    )

    threshold_override = st.slider(
        "Decision Threshold for Fraud Detection:",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Job is classified as Fraudulent if Fraud Probability >= Threshold. Lower threshold increases sensitivity (Recall)."
    )
    
    st.markdown("---")
    st.markdown("### 🎓 Academic Project Info")
    st.markdown("""
    - **Course:** Machine Learning / NLP Lab
    - **Models:** ModernBERT-base & RoBERTa-base
    - **Dataset:** EMSCAD (17,880 postings)
    - **Target:** 0 = Genuine, 1 = Fraudulent
    """)

# ==============================================================================
# MAIN INTERFACE
# ==============================================================================
st.markdown('<div class="main-title">🛡️ Online Recruitment Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Binary Classification of Online Job Postings using Fine-Tuned Transformer Models</div>', unsafe_allow_html=True)

# Load model status
with st.spinner("Loading Transformer model and tokenizer..."):
    tokenizer, model, loaded_threshold, load_error = load_model_and_tokenizer(model_source)

if load_error:
    st.error(f"❌ Failed to load model from `{model_source}`.\n\nError details: {load_error}\n\nPlease verify that the model path exists or that the Hugging Face repository identifier is correct.")
    st.info("💡 **Tip:** If running locally after training in the notebook, make sure you ran Step 11 which saves the best model to `saved_models/best_model`.")
else:
    st.success(f"✅ Model successfully loaded from `{model_source}`!")

# ==============================================================================
# PRESET SAMPLE TEMPLATES FOR TESTING
# ==============================================================================
st.markdown("### 📝 Input Job Details")
col_preset, col_clear = st.columns([4, 1])

with col_preset:
    preset_choice = st.selectbox(
        "Load an Example Job Posting Template:",
        [
            "Select an option or write custom...",
            "Example 1: Genuine Senior Python/Django Software Engineer",
            "Example 2: Suspicious Work-From-Home Data Entry ($500/day)",
            "Example 3: Fraudulent Fee-Charging Executive Assistant",
            "Example 4: Genuine University Graduate Internship"
        ]
    )

# Template values
preset_values = {
    "title": "",
    "company_profile": "",
    "description": "",
    "requirements": "",
    "benefits": "",
    "employment_type": "Full-time",
    "required_experience": "Mid-Senior level",
    "required_education": "Bachelor's Degree",
    "industry": "Information Technology",
    "function": "Engineering"
}

if preset_choice == "Example 1: Genuine Senior Python/Django Software Engineer":
    preset_values = {
        "title": "Senior Python Backend Engineer",
        "company_profile": "Innovatech Solutions is an established cloud infrastructure provider founded in 2014 with over 150 employees worldwide.",
        "description": "We are seeking a seasoned Senior Python Backend Engineer to build high-scale microservices, optimize PostgreSQL queries, and design robust REST/GraphQL APIs.",
        "requirements": "5+ years of production Python experience with Django/FastAPI. Strong grasp of Docker, Kubernetes, CI/CD pipelines, and AWS cloud architecture.",
        "benefits": "Competitive annual salary ($120k-$145k), comprehensive health/dental insurance, 401(k) matching, and 25 days paid annual vacation.",
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "Bachelor's Degree",
        "industry": "Information Technology and Services",
        "function": "Engineering"
    }
elif preset_choice == "Example 2: Suspicious Work-From-Home Data Entry ($500/day)":
    preset_values = {
        "title": "URGENT: Online Data Entry Clerk - Work From Home Immediate Start!",
        "company_profile": "Leading global investment agency expanding fast worldwide.",
        "description": "Earn $500 to $800 daily working just 2 hours from your smartphone or laptop. No interview required! Immediate start for motivated individuals. Send your personal WhatsApp contact.",
        "requirements": "Must have mobile phone, WhatsApp account, and personal bank account to receive immediate wire transfers. No prior experience or degree needed.",
        "benefits": "Daily payouts via Western Union / Wire, guaranteed bonuses, zero supervision.",
        "employment_type": "Part-time",
        "required_experience": "Not Applicable",
        "required_education": "Unspecified",
        "industry": "Financial Services",
        "function": "Administrative"
    }
elif preset_choice == "Example 3: Fraudulent Fee-Charging Executive Assistant":
    preset_values = {
        "title": "Executive Virtual Assistant - Guaranteed Placement",
        "company_profile": "Confidential Global VIP Management Group.",
        "description": "Provide administrative support to international executives. Prior to commencing duties, selected applicants must purchase an onboarding software security license ($150 refundable fee).",
        "requirements": "Basic English proficiency, ability to process gift card transfers, active PayPal or CashApp account.",
        "benefits": "$85,000 yearly salary, fully paid remote workspace equipment provided upon registration fee clearance.",
        "employment_type": "Contract",
        "required_experience": "Entry level",
        "required_education": "High School or equivalent",
        "industry": "Management Consulting",
        "function": "Administrative"
    }
elif preset_choice == "Example 4: Genuine University Graduate Internship":
    preset_values = {
        "title": "Junior Machine Learning Research Intern (Summer 2026)",
        "company_profile": "BioAnalytics Labs conducts academic and industrial biomedical data research in partnership with regional universities.",
        "description": "We offer a 3-month paid summer internship for computer science students interested in computer vision and natural language processing.",
        "requirements": "Currently enrolled in BS/MS in Computer Science, Data Science, or related field. Familiarity with PyTorch, Scikit-learn, and Git.",
        "benefits": "Monthly stipend of $2,500, mentorship from senior research scientists, publication co-authorship opportunities.",
        "employment_type": "Internship",
        "required_experience": "Internship",
        "required_education": "Bachelor's Degree",
        "industry": "Research",
        "function": "Research"
    }

# ==============================================================================
# INPUT FORM
# ==============================================================================
with st.form("job_form"):
    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input("Job Title *", value=preset_values["title"], placeholder="e.g. Senior Software Engineer")
        employment_type = st.selectbox(
            "Employment Type",
            ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"],
            index=0 if preset_values["employment_type"] not in ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"] else ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"].index(preset_values["employment_type"])
        )
        required_experience = st.text_input("Required Experience", value=preset_values["required_experience"], placeholder="e.g. Mid-Senior level, 3+ years")
        industry = st.text_input("Industry", value=preset_values["industry"], placeholder="e.g. Information Technology")

    with col2:
        required_education = st.text_input("Required Education", value=preset_values["required_education"], placeholder="e.g. Bachelor's Degree")
        function = st.text_input("Function / Department", value=preset_values["function"], placeholder="e.g. Engineering, Sales")
        company_profile = st.text_area("Company Profile", value=preset_values["company_profile"], height=108, placeholder="Brief overview of the hiring company...")

    job_description = st.text_area("Job Description *", value=preset_values["description"], height=140, placeholder="Detailed job responsibilities and scope...")
    
    col3, col4 = st.columns(2)
    with col3:
        requirements = st.text_area("Requirements / Qualifications", value=preset_values["requirements"], height=100, placeholder="Skills, qualifications, and past experience required...")
    with col4:
        benefits = st.text_area("Benefits / Perks Offered", value=preset_values["benefits"], height=100, placeholder="Health insurance, 401(k), compensation details...")

    submitted = st.form_submit_button("🔍 Analyze Job Posting", use_container_width=True)

# ==============================================================================
# INFERENCE EXECUTION
# ==============================================================================
if submitted:
    if not job_title and not job_description:
        st.warning("⚠️ Please provide at least a **Job Title** or a **Job Description** to analyze.")
    elif tokenizer is None or model is None:
        st.error("❌ Model is not loaded. Cannot perform inference.")
    else:
        # Preprocess text
        combined_text = combine_job_text(
            title=job_title,
            company_profile=company_profile,
            description=job_description,
            requirements=requirements,
            benefits=benefits,
            employment_type=employment_type,
            required_experience=required_experience,
            required_education=required_education,
            industry=industry,
            function=function
        )

        with st.spinner("Analyzing job posting with Transformer model..."):
            inputs = tokenizer(
                combined_text,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )

            # Move inputs to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            genuine_prob = float(probs[0])
            fraud_prob = float(probs[1])

            # Apply user-selected threshold
            pred_label = 1 if fraud_prob >= threshold_override else 0

        # ==============================================================================
        # DISPLAY RESULTS
        # ==============================================================================
        st.markdown("---")
        st.subheader("📊 Analysis Results")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if pred_label == 1:
                st.error("### 🚨 Prediction: FRAUDULENT JOB POSTING")
                st.markdown(f"The model detected high indicators of **recruitment scam or fraudulent activity** (Fraud Probability: **{fraud_prob * 100:.2f}%** exceeds the threshold of {threshold_override * 100:.1f}%).")
            else:
                st.success("### ✅ Prediction: GENUINE JOB POSTING")
                st.markdown(f"The job posting matches the textual characteristics of **legitimate employment offers** (Genuine Probability: **{genuine_prob * 100:.2f}%**).")

            # Risk indicator
            if fraud_prob >= 0.70:
                risk_badge = "🔴 **HIGH RISK**"
                risk_desc = "Significant scam flags (e.g. unrealistic remuneration, suspicious payment/registration wording, vague corporate details)."
            elif fraud_prob >= threshold_override:
                risk_badge = "🟠 **MODERATE RISK**"
                risk_desc = "Several abnormal patterns detected. Manual verification of employer credentials is recommended."
            else:
                risk_badge = "🟢 **LOW RISK**"
                risk_desc = "Posting exhibits normal professional job specifications and realistic requirements."

            st.markdown(f"**Risk Level:** {risk_badge}")
            st.caption(risk_desc)

        with res_col2:
            st.markdown("#### Probability Distribution")
            st.write(f"**Genuine Probability:** `{genuine_prob * 100:.2f}%`")
            st.progress(genuine_prob)
            st.write(f"**Fraudulent Probability:** `{fraud_prob * 100:.2f}%`")
            st.progress(fraud_prob)

        # Inspection of combined text input
        with st.expander("🔍 View Preprocessed Model Input"):
            st.text(combined_text)

# ==============================================================================
# DISCLAIMER (MANDATORY REQUIREMENT)
# ==============================================================================
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Academic Project Disclaimer:</strong><br>
    This Online Recruitment Fraud Detection System is an academic research and engineering prototype developed using fine-tuned Transformer models (ModernBERT-base and RoBERTa-base). Model predictions are statistical estimates based on historical patterns in the EMSCAD dataset and do not constitute absolute legal or financial judgments. Always exercise due diligence before sharing sensitive personal information or transferring funds.
</div>
""", unsafe_allow_html=True)
