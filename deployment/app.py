import os
import re
import json
import torch
import streamlit as st
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================================================================
# ENTERPRISE PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="VeriJob AI | Recruitment Fraud & Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# PROFESSIONAL SAAS UI STYLING
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 2rem 2.5rem;
        border-radius: 0.75rem;
        color: #FFFFFF;
        margin-bottom: 1.75rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .main-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 0.35rem;
        font-weight: 400;
    }
    .status-badge-online {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background-color: #064E3B;
        color: #34D399;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        border: 1px solid #059669;
        margin-top: 0.75rem;
    }
    .kpi-card {
        background: #FFFFFF;
        padding: 1.25rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
    }
    .result-alert-fraud {
        background-color: #FEF2F2;
        border: 1px solid #F87171;
        border-left: 6px solid #DC2626;
        padding: 1.25rem;
        border-radius: 0.5rem;
        color: #991B1B;
        margin-bottom: 1.25rem;
    }
    .result-alert-genuine {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        border-left: 6px solid #16A34A;
        padding: 1.25rem;
        border-radius: 0.5rem;
        color: #166534;
        margin-bottom: 1.25rem;
    }
    .section-label {
        font-size: 0.88rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.35rem;
    }
    .footer-note {
        text-align: center;
        padding: 1.5rem;
        color: #64748B;
        font-size: 0.82rem;
        border-top: 1px solid #E2E8F0;
        margin-top: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODEL LOADING ENGINE (CACHED)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_risk_engine(model_source_path: str):
    """
    Loads Transformer model and tokenizer using st.cache_resource.
    Supports local directories or Hugging Face repository identifiers.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_source_path)
        model.eval()

        optimal_threshold = 0.50
        cfg_file = os.path.join(model_source_path, "model_config.json")
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    optimal_threshold = float(meta.get("best_threshold", 0.50))
            except Exception:
                pass

        return tokenizer, model, optimal_threshold, None
    except Exception as e:
        return None, None, 0.50, str(e)


# ==============================================================================
# ROBUST PREPROCESSING PIPELINE
# ==============================================================================
def preprocess_job_payload(
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
    Formats multi-attribute job specifications into a structured sequence.
    Prioritizes Job Description, Requirements, and Benefits before Company Profile,
    and caps Company Profile to max 60 words to eliminate feature dominance and truncation.
    """
    # Budget company profile to avoid attention dilution
    cp_words = company_profile.strip().split() if company_profile else []
    cp_budgeted = " ".join(cp_words[:60]) if cp_words else ""

    parts = [
        f"Job Title: {title.strip() if title else ''}",
        f"Job Description: {description.strip() if description else ''}",
        f"Requirements: {requirements.strip() if requirements else ''}",
        f"Benefits: {benefits.strip() if benefits else ''}",
        f"Company Profile: {cp_budgeted}",
        f"Employment Type: {employment_type.strip() if employment_type else ''}",
        f"Required Experience: {required_experience.strip() if required_experience else ''}",
        f"Required Education: {required_education.strip() if required_education else ''}",
        f"Industry: {industry.strip() if industry else ''}",
        f"Function: {function.strip() if function else ''}"
    ]
    raw_str = "\n".join(parts)
    # Collapse redundant newlines and whitespace
    clean_str = re.sub(r'\r\n|\r|\n', ' ', raw_str)
    clean_str = re.sub(r'\s+', ' ', clean_str)
    return clean_str.strip()


# ==============================================================================
# SIDEBAR: SYSTEM CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### 🛡️ System Settings")
    
    # Detect environment variable, Streamlit secrets, or local checkpoints
    env_model = ""
    try:
        if hasattr(st, "secrets") and st.secrets is not None:
            if "HF_MODEL_REPO" in st.secrets:
                env_model = st.secrets["HF_MODEL_REPO"]
            elif "HF_MODEL_ID" in st.secrets:
                env_model = st.secrets["HF_MODEL_ID"]
    except Exception:
        pass

    if not env_model:
        env_model = os.getenv("HF_MODEL_REPO", os.getenv("HF_MODEL_ID", "")).strip()

    default_candidates = [
        os.path.join("..", "saved_models", "best_model"),
        os.path.join(".", "saved_models", "best_model"),
        os.path.join("saved_models", "best_model"),
    ]
    selected_default = env_model if env_model else "sami2715/recruitment-fraud-detector"
    for c in default_candidates:
        if os.path.exists(c):
            selected_default = c
            break

    model_target = st.text_input(
        "Inference Model / Repository:",
        value=selected_default,
        help="Local checkpoint path or Hugging Face Model ID (e.g., your-username/recruitment-fraud-detector)"
    )

    risk_threshold = st.slider(
        "Sensitivity Threshold (Fraud Cutoff):",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Postings with Fraud Probability exceeding this threshold are flagged as High Risk."
    )

    st.markdown("---")
    st.markdown("#### Engine Diagnostics")
    st.caption("• **Architecture:** Deep Bidirectional Transformer")
    st.caption("• **Max Sequence Length:** 384 tokens")
    st.caption("• **Inference Precision:** FP32 / Automatic Torch Device")
    st.caption("• **Status:** Operational")

# ==============================================================================
# MAIN DASHBOARD HEADER
# ==============================================================================
st.markdown("""
<div class="main-header">
    <div class="main-title">
        <span>🛡️</span> VeriJob AI | Recruitment Fraud & Security Intelligence
    </div>
    <div class="main-subtitle">
        Enterprise-grade screening system for detecting recruitment fraud, phishing schemes, and employment scams.
    </div>
    <div class="status-badge-online">
        ● Security Engine Online
    </div>
</div>
""", unsafe_allow_html=True)

# Load Model
with st.spinner("Connecting to security intelligence engine..."):
    tokenizer, model, detected_thresh, load_err = load_risk_engine(model_target)

if load_err:
    st.error(f"⚠️ Unable to load model weights from `{model_target}`.\n\nError: {load_err}")
    st.info("Tip: If deployed on Hugging Face or Streamlit Cloud, enter your Hugging Face model repository ID in the left sidebar.")
    st.stop()

# ==============================================================================
# PRESET TEMPLATES
# ==============================================================================
col_tpl, col_btn = st.columns([4, 1])

with col_tpl:
    template_selection = st.selectbox(
        "Load a Preset Verification Scenario:",
        [
            "Select an option or input custom posting...",
            "Preset 1: Legitimate Senior Software Engineer (Enterprise Cloud)",
            "Preset 2: High-Risk Work-From-Home Typist ($500/day + WhatsApp)",
            "Preset 3: Fraudulent Executive Assistant (Upfront Software Fee)",
            "Preset 4: Legitimate Research Internship (University Partner)",
            "Preset 5: High-Risk Wire Transfer / Regional Payroll Rep"
        ]
    )

tpl = {
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

if template_selection == "Preset 1: Legitimate Senior Software Engineer (Enterprise Cloud)":
    tpl = {
        "title": "Senior Cloud Infrastructure Engineer (Python/AWS)",
        "company_profile": "Datacore Systems is a SOC-2 certified cloud infrastructure provider founded in 2014, operating data centers across North America and Europe with 450+ enterprise clients.",
        "description": "We are seeking a seasoned Cloud Infrastructure Engineer to design resilient Kubernetes clusters, automate CI/CD release pipelines, and optimize microservices architecture. You will work closely with distributed engineering teams to ensure 99.99% uptime.",
        "requirements": "5+ years of production experience in Python, Go, or Java. Extensive hands-on knowledge of AWS (EKS, IAM, RDS), Terraform, Docker, and Linux kernel internals. BS or MS in Computer Science or equivalent technical experience.",
        "benefits": "Competitive base compensation ($135,000 - $160,000), equity options, comprehensive health/dental/vision coverage, 401(k) retirement matching, and 25 days paid annual leave.",
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "Bachelor's Degree",
        "industry": "Information Technology",
        "function": "Engineering"
    }
elif template_selection == "Preset 2: High-Risk Work-From-Home Typist ($500/day + WhatsApp)":
    tpl = {
        "title": "URGENT: Online Data Entry Typist - Work From Home ($500/Day)",
        "company_profile": "First Horizon Capital Ventures is an international asset management enterprise founded in 2011 with partner offices in Zurich and Singapore.",
        "description": "Earn $500 to $800 daily working just 2 hours from your phone or laptop! Immediate start without formal interview. Send your personal WhatsApp contact to our recruitment supervisor immediately.",
        "requirements": "Must own a smartphone or laptop, active WhatsApp account, and personal bank account to receive immediate wire transfers. No prior experience needed.",
        "benefits": "Daily payouts via Western Union or Wire, guaranteed weekly cash bonuses, zero supervision.",
        "employment_type": "Part-time",
        "required_experience": "Not Applicable",
        "required_education": "Unspecified",
        "industry": "Financial Services",
        "function": "Administrative"
    }
elif template_selection == "Preset 3: Fraudulent Executive Assistant (Upfront Software Fee)":
    tpl = {
        "title": "Executive Virtual Assistant - Immediate Placement",
        "company_profile": "VIP Global Concierge Group provides dedicated executive support to visiting international corporate directors.",
        "description": "Provide dedicated virtual administrative support, coordinate executive flight schedules, and handle confidential client correspondence. Prior to commencing remote duties, selected candidates are required to purchase an onboarding software security license ($150 refundable registration fee).",
        "requirements": "Active CashApp, PayPal, or personal bank account to process corporate disbursements. Ability to purchase gift cards and wire transaction vouchers.",
        "benefits": "$85,000 yearly salary, brand new Apple MacBook Pro and fully paid home office workstation provided upon fee clearance.",
        "employment_type": "Contract",
        "required_experience": "Entry level",
        "required_education": "High School or equivalent",
        "industry": "Management Consulting",
        "function": "Administrative"
    }
elif template_selection == "Preset 4: Legitimate Research Internship (University Partner)":
    tpl = {
        "title": "Machine Learning Research Intern (Summer 2026)",
        "company_profile": "BioAnalytics Research Institute conducts collaborative biomedical data science and NLP research in partnership with regional university medical centers.",
        "description": "We are offering a 3-month paid summer research internship for computer science and data science students. Interns will assist research scientists in evaluating deep learning models for biomedical sequence analysis.",
        "requirements": "Currently enrolled in a BS or MS degree program in Computer Science, Data Science, or related STEM discipline. Basic working knowledge of Python and PyTorch. Familiarity with Git version control.",
        "benefits": "Monthly research stipend ($3,000/month), direct mentorship from senior research scientists, and co-authorship on peer-reviewed research publications.",
        "employment_type": "Internship",
        "required_experience": "Internship",
        "required_education": "Bachelor's Degree",
        "industry": "Research",
        "function": "Research"
    }
elif template_selection == "Preset 5: High-Risk Wire Transfer / Regional Payroll Rep":
    tpl = {
        "title": "Regional Payment Processing Representative",
        "company_profile": "Alliance Global Clearing Services.",
        "description": "Urgent opening for a regional transactions clerk. You will receive corporate wire disbursements into your personal checking account and forward funds to designated overseas suppliers within 24 hours.",
        "requirements": "Must be 18+, active personal bank account in good standing, ability to execute Western Union or Bitcoin transfers rapidly.",
        "benefits": "Keep 10% commission on every transaction processed, instant wire bonuses, no working hours constraints.",
        "employment_type": "Temporary",
        "required_experience": "Not Applicable",
        "required_education": "Unspecified",
        "industry": "Banking",
        "function": "Finance"
    }

# ==============================================================================
# INPUT FORM
# ==============================================================================
with st.form("job_analysis_form"):
    st.markdown("#### 1. Job Role & Core Specifications")
    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input("Job Title *", value=tpl["title"], placeholder="e.g. Senior Software Engineer")
        employment_type = st.selectbox(
            "Employment Type",
            ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"],
            index=["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"].index(tpl["employment_type"]) if tpl["employment_type"] in ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"] else 0
        )
        required_experience = st.text_input("Experience Level", value=tpl["required_experience"], placeholder="e.g. Mid-Senior level, 3+ years")
        
    with col2:
        function = st.text_input("Department / Function", value=tpl["function"], placeholder="e.g. Engineering, Finance")
        required_education = st.text_input("Educational Requirement", value=tpl["required_education"], placeholder="e.g. Bachelor's Degree")
        industry = st.text_input("Industry Sector", value=tpl["industry"], placeholder="e.g. Information Technology, Financial Services")

    st.markdown("#### 2. Position Description & Duties")
    job_description = st.text_area("Job Description *", value=tpl["description"], height=130, placeholder="Detailed job responsibilities, day-to-day duties, and scope...")

    col3, col4 = st.columns(2)
    with col3:
        requirements = st.text_area("Required Qualifications & Skills", value=tpl["requirements"], height=100, placeholder="Technical requirements, soft skills, credentials...")
    with col4:
        benefits = st.text_area("Compensation & Benefits", value=tpl["benefits"], height=100, placeholder="Salary range, health insurance, bonuses, perks...")

    st.markdown("#### 3. Employer Background")
    company_profile = st.text_area("Company Profile / Overview", value=tpl["company_profile"], height=90, placeholder="About the hiring organization...")

    analyze_clicked = st.form_submit_button("🔍 Run Security Risk Assessment", use_container_width=True)

# ==============================================================================
# INFERENCE & RISK ASSESSMENT
# ==============================================================================
if analyze_clicked:
    if not job_title and not job_description:
        st.warning("⚠️ Please provide at least a **Job Title** or **Job Description** to perform verification.")
    else:
        # Preprocess
        payload_text = preprocess_job_payload(
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

        with st.spinner("Analyzing linguistic patterns and structural risk indicators..."):
            inputs = tokenizer(
                payload_text,
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt"
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

            genuine_score = float(probs[0])
            fraud_score = float(probs[1])
            is_fraudulent = fraud_score >= risk_threshold

        # ==============================================================================
        # DASHBOARD RESULTS DISPLAY
        # ==============================================================================
        st.markdown("---")
        st.markdown("### 📊 Security Assessment Report")

        # Top Banner
        if is_fraudulent:
            st.markdown(f"""
            <div class="result-alert-fraud">
                <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.35rem;">
                    🚨 HIGH RISK DETECTED: Potential Recruitment Scam
                </div>
                <div>
                    The automated analysis detected strong linguistic and structural patterns characteristic of fraudulent job postings (Fraud Probability: <strong>{fraud_score * 100:.2f}%</strong> exceeding the {risk_threshold * 100:.1f}% alert threshold).
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-alert-genuine">
                <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.35rem;">
                    ✅ VERIFIED: Standard / Low Risk Job Posting
                </div>
                <div>
                    The listing exhibits standard corporate specifications and language consistent with legitimate employment opportunities (Legitimacy Score: <strong>{genuine_score * 100:.2f}%</strong>).
                </div>
            </div>
            """, unsafe_allow_html=True)

        # KPI Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Fraud Risk Score</div>
                <div class="kpi-value" style="color: {'#DC2626' if is_fraudulent else '#059669'};">
                    {fraud_score * 100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Legitimacy Score</div>
                <div class="kpi-value" style="color: #0F172A;">
                    {genuine_score * 100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            if fraud_score >= 0.75:
                tier_badge = "CRITICAL"
                tier_color = "#DC2626"
            elif fraud_score >= risk_threshold:
                tier_badge = "ELEVATED"
                tier_color = "#EA580C"
            else:
                tier_badge = "LOW"
                tier_color = "#059669"

            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Risk Tier</div>
                <div class="kpi-value" style="color: {tier_color};">
                    {tier_badge}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Recommended Action</div>
                <div class="kpi-value" style="font-size: 1.05rem; color: #334155; padding-top: 0.35rem;">
                    {"FLAG / DO NOT ENGAGE" if is_fraudulent else "PROCEED WITH CAUTION"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Risk breakdown visual bar
        st.markdown("<br>", unsafe_allow_html=True)
        col_bar1, col_bar2 = st.columns(2)
        with col_bar1:
            st.write(f"**Legitimacy Confidence:** `{genuine_score * 100:.2f}%`")
            st.progress(genuine_score)
        with col_bar2:
            st.write(f"**Fraud Probability:** `{fraud_score * 100:.2f}%`")
            st.progress(fraud_score)

        # Automated Diagnostic Red Flags
        full_text_lower = payload_text.lower()
        red_flags_found = []
        if any(w in full_text_lower for w in ["wire transfer", "western union", "moneygram", "bitcoin"]):
            red_flags_found.append("Mentions irreversible money transfer methods (Western Union, Bitcoin, Wire Transfer).")
        if any(w in full_text_lower for w in ["registration fee", "refundable fee", "software license fee", "security license"]):
            red_flags_found.append("Requests upfront payments, security deposits, or software licensing fees prior to work.")
        if any(w in full_text_lower for w in ["personal bank account", "personal checking", "cashier check"]):
            red_flags_found.append("Requests use of candidate's personal bank account for corporate transaction processing.")
        if any(w in full_text_lower for w in ["whatsapp", "telegram", "@gmail.com", "@yahoo.com"]):
            red_flags_found.append("Directs candidates to informal off-platform communication (WhatsApp, Telegram, personal Gmail).")
        if any(w in full_text_lower for w in ["$500 daily", "$600 daily", "$800 daily", "$500/day", "$800/day"]):
            red_flags_found.append("Promises disproportionately high daily earnings ($500+ daily) for entry-level tasks.")

        if red_flags_found:
            st.markdown("#### 🚩 Automated Heuristic Flags Identified:")
            for flag in red_flags_found:
                st.markdown(f"- ⚠️ **Flag:** {flag}")

        # Inspection Accordion
        with st.expander("🔍 View Preprocessed Model Input"):
            st.code(payload_text, language="text")

# ==============================================================================
# ENTERPRISE FOOTER
# ==============================================================================
st.markdown("""
<div class="footer-note">
    <strong>VeriJob AI Risk Advisory:</strong> This automated assessment engine applies deep Transformer self-attention representations to detect deceptive recruitment patterns. Model inferences serve as advisory risk indicators. Always verify employer corporate email domains and never transfer funds or share sensitive credentials during recruitment workflows.
</div>
""", unsafe_allow_html=True)
