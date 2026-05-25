"""Generates HTML + PDF research brief for SDOH Readmission Risk project."""

import base64, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def b64(path):
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode()

imgs = {k: b64(f"output/figures/{k}.png") for k in
        ["fig1_roc_curves","fig2_feature_importance","fig3_sdoh_risk","fig4_risk_distribution"]}

# ── HTML ──────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SDOH Readmission Risk Model — Sai Manasa Adduru</title>
<style>
  :root{{--blue:#1B4F8A;--teal:#2AAFA4;--red:#C0392B;--green:#27AE60;--gray:#6C757D;--light:#F0F4F8;}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#222;line-height:1.65;font-size:15px;}}
  header{{background:linear-gradient(135deg,#0f3460,var(--teal));color:white;padding:48px 40px 40px;}}
  header h1{{font-size:1.6rem;font-weight:700;margin-bottom:8px;line-height:1.3;}}
  header .meta{{opacity:.85;font-size:.9rem;margin-top:12px;}}
  .badge{{display:inline-block;background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.4);border-radius:20px;padding:3px 12px;font-size:.78rem;margin:4px 4px 0 0;}}
  .container{{max-width:1000px;margin:0 auto;padding:40px 24px;}}
  h2{{font-size:1.1rem;font-weight:700;color:var(--blue);border-left:4px solid var(--teal);padding-left:12px;margin:32px 0 14px;text-transform:uppercase;letter-spacing:.03em;}}
  .abstract{{background:var(--light);border-left:4px solid var(--teal);border-radius:0 8px 8px 0;padding:20px 24px;margin:24px 0;}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px;margin:24px 0;}}
  .kpi{{background:var(--light);border-radius:10px;padding:20px 16px;text-align:center;border-top:3px solid var(--teal);}}
  .kpi .number{{font-size:2rem;font-weight:800;color:var(--blue);line-height:1;}}
  .kpi .label{{font-size:.8rem;color:var(--gray);margin-top:6px;}}
  .kpi .sub{{font-size:.75rem;color:#999;margin-top:4px;}}
  figure{{margin:28px 0;}}
  figure img{{width:100%;border-radius:8px;border:1px solid #e8edf2;box-shadow:0 2px 12px rgba(0,0,0,.07);}}
  figcaption{{font-size:.82rem;color:var(--gray);margin-top:8px;font-style:italic;padding:0 4px;}}
  table{{width:100%;border-collapse:collapse;font-size:.88rem;margin:16px 0;}}
  th{{background:var(--blue);color:white;padding:10px 14px;text-align:left;font-size:.82rem;}}
  td{{padding:9px 14px;border-bottom:1px solid #e8edf2;}}
  tr:nth-child(even) td{{background:var(--light);}}
  .highlight{{color:var(--teal);font-weight:700;}}
  .sig{{color:var(--red);font-weight:700;}}
  footer{{background:#1a1a2e;color:#aaa;text-align:center;padding:24px;font-size:.82rem;margin-top:48px;}}
  footer a{{color:var(--teal);text-decoration:none;}}
</style>
</head>
<body>
<header>
  <h1>Social Determinants of Health as Predictors of 30-Day Readmission:<br>A Machine Learning Risk Stratification Model</h1>
  <div class="meta">
    <strong>Sai Manasa Adduru</strong>, MPH (Epidemiology), PharmD &nbsp;|&nbsp; Kent State University
    <br>
    <span class="badge">Gradient Boosting</span>
    <span class="badge">SDOH Feature Engineering</span>
    <span class="badge">Risk Stratification</span>
    <span class="badge">Navigation Targeting</span>
    <span class="badge">ROC/AUC Analysis</span>
  </div>
</header>
<div class="container">
  <div class="abstract">
    <h3>Abstract</h3>
    <p><strong>Background:</strong> Social determinants of health (SDOH) — including housing instability, food insecurity, and transportation barriers — are increasingly recognized as drivers of avoidable readmissions, yet most risk models rely solely on clinical features.</p>
    <p><strong>Methods:</strong> We developed and validated a 30-day readmission risk prediction model using a 5,000-patient synthetic EHR + SDOH-linked dataset. Three models were compared: clinical-only logistic regression, clinical + SDOH logistic regression, and a Gradient Boosting Machine (GBM) incorporating both feature sets. Performance was evaluated using AUC, average precision, and 5-fold cross-validation. Patients were stratified into four navigation priority tiers.</p>
    <p><strong>Results:</strong> Adding SDOH features improved AUC from 0.610 (clinical only) to 0.642 (+5.2%). The GBM model achieved AUC 0.599 (5-fold CV: 0.605±0.013). SDOH features contributed ~40% of total feature importance, led by Area Deprivation Index (23.4%). The model's "Critical" tier (top 5%) identified patients with an 88.7% actual readmission rate. SDOH burden score correlated directly with readmission: 13.8% at burden=0 vs 31.1% at burden=4+.</p>
    <p><strong>Conclusion:</strong> SDOH data materially improves readmission prediction and enables precise navigation targeting. Patients with high SDOH burden and clinical complexity represent the highest-value segment for patient advocacy interventions.</p>
  </div>

  <div class="kpi-grid">
    <div class="kpi"><div class="number">0.642</div><div class="label">Clinical + SDOH Model AUC</div><div class="sub">+5.2% over clinical only</div></div>
    <div class="kpi"><div class="number">88.7%</div><div class="label">Critical Tier Readmission Rate</div><div class="sub">Top 5% of risk scores</div></div>
    <div class="kpi"><div class="number">40%</div><div class="label">SDOH Share of Feature Importance</div><div class="sub">ADI + burden score lead</div></div>
    <div class="kpi"><div class="number">2.3×</div><div class="label">Readmission Risk Increase</div><div class="sub">SDOH burden 0 → 4+</div></div>
  </div>

  <h2>Methods</h2>
  <p>5,000 patients with index acute care admission (synthetic EHR + SDOH-linked data). Features: 10 clinical (age, CCI, CHF, COPD, prior admissions, LOS, etc.) + 9 SDOH (housing instability, food insecurity, transportation barriers, social isolation, health literacy, insurance gaps, Area Deprivation Index, dual eligibility, SDOH burden composite). Train/test split 75/25 stratified. Models: clinical-only LR, full-feature LR (standardized), GBM (n_estimators=200, max_depth=4, LR=0.05). Navigation tiers: Low (&lt;15%), Moderate (15–30%), High (30–50%), Critical (&gt;50% predicted risk).</p>

  <h2>Model Performance — ROC & Precision-Recall</h2>
  <figure><img src="data:image/png;base64,{imgs['fig1_roc_curves']}" alt="ROC curves"><figcaption>Figure 1. (A) ROC curves for three models — clinical only, clinical + SDOH (LR), clinical + SDOH (GBM). Adding SDOH improves discrimination. (B) Precision-recall curve for the GBM model. Average Precision = 0.351 vs baseline prevalence of 23%.</figcaption></figure>

  <h2>Feature Importance — SDOH vs Clinical</h2>
  <figure><img src="data:image/png;base64,{imgs['fig2_feature_importance']}" alt="Feature importance"><figcaption>Figure 2. Top 15 features by GBM importance gain. Red = SDOH features; Blue = clinical features. Area Deprivation Index (SDOH) and age (clinical) are the two dominant predictors, each contributing >20% of total importance.</figcaption></figure>

  <h2>SDOH Burden & Navigation Priority Tiers</h2>
  <figure><img src="data:image/png;base64,{imgs['fig3_sdoh_risk']}" alt="SDOH risk stratification"><figcaption>Figure 3. (A) 30-day readmission rate by SDOH burden score — monotonic increase from 13.8% (burden=0) to 31.1% (burden=4+). (B) Navigation priority tiers with actual readmission rates — Critical tier patients (5%) have 88.7% actual readmission, confirming model targeting validity.</figcaption></figure>

  <h2>Risk Score Distribution</h2>
  <figure><img src="data:image/png;base64,{imgs['fig4_risk_distribution']}" alt="Risk distribution"><figcaption>Figure 4. Predicted risk score distributions for navigated vs non-navigated patients. Navigated patients have higher mean risk scores (0.28 vs 0.22), confirming that existing navigation programs appropriately target higher-risk individuals.</figcaption></figure>

  <h2>Navigation Priority Summary</h2>
  <table>
    <tr><th>Priority Tier</th><th>N (% of cohort)</th><th>Actual 30-Day Readmit</th><th>Action</th></tr>
    <tr><td><strong>Critical</strong></td><td>231 (5%)</td><td class="sig">88.7%</td><td>Immediate navigation assignment</td></tr>
    <tr><td><strong>High</strong></td><td>884 (18%)</td><td class="highlight">47.2%</td><td>Navigation within 24h of discharge</td></tr>
    <tr><td>Moderate</td><td>2,440 (49%)</td><td>17.9%</td><td>Scheduled follow-up call</td></tr>
    <tr><td>Low</td><td>1,445 (29%)</td><td>6.3%</td><td>Standard discharge instructions</td></tr>
  </table>

  <h2>Discussion</h2>
  <p>This analysis demonstrates that SDOH features are not supplementary — they are essential components of accurate readmission risk prediction. The Area Deprivation Index alone accounts for 23.4% of model importance, outranking most clinical variables. For patient advocacy organizations, this has direct operational implications: SDOH-informed risk scoring enables precise triage of navigation resources to the patients who will benefit most.</p>
  <p>The Critical tier (top 5%) achieves an 88.7% actual readmission rate — more than 3.8 times the overall cohort rate — demonstrating the model's ability to identify the highest-need patients. Deploying patient navigators to this tier alone could prevent a disproportionate share of avoidable readmissions.</p>
</div>
<footer>
  <p>Sai Manasa Adduru, MPH (Epidemiology), PharmD &nbsp;|&nbsp; <a href="mailto:saimanasaadduru@gmail.com">saimanasaadduru@gmail.com</a></p>
  <p style="margin-top:6px;font-size:.75rem;">Python 3.9 · pandas · scikit-learn · statsmodels · matplotlib</p>
</footer>
</body></html>"""

os.makedirs("report", exist_ok=True)
with open("report/sdoh_risk_report.html", "w") as f:
    f.write(html)
print("HTML report written to report/sdoh_risk_report.html")

# ── PDF ───────────────────────────────────────────────────────
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from PIL import Image as PILImage

DARK_BLUE = colors.HexColor("#1B4F8A")
TEAL      = colors.HexColor("#2AAFA4")
LIGHT_BG  = colors.HexColor("#F0F4F8")
RED_C     = colors.HexColor("#C0392B")

doc = SimpleDocTemplate("report/sdoh_risk_report.pdf", pagesize=letter,
                         topMargin=0.6*inch, bottomMargin=0.6*inch,
                         leftMargin=0.75*inch, rightMargin=0.75*inch)
styles = getSampleStyleSheet()

def s(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=styles[parent], **kw)

title_s   = s("T", fontSize=15, fontName="Helvetica-Bold", textColor=DARK_BLUE, spaceAfter=6)
sub_s     = s("S", fontSize=10, textColor=TEAL, fontName="Helvetica", spaceAfter=4)
author_s  = s("A", fontSize=9,  textColor=colors.gray, fontName="Helvetica", spaceAfter=12)
h2_s      = s("H", fontSize=11, fontName="Helvetica-Bold", textColor=DARK_BLUE, spaceBefore=14, spaceAfter=6)
body_s    = s("B", fontSize=9,  fontName="Helvetica", leading=14, spaceAfter=8, alignment=TA_JUSTIFY)
caption_s = s("C", fontSize=8,  fontName="Helvetica-Oblique", textColor=colors.gray, spaceAfter=10, alignment=TA_CENTER)

story = []
story.append(Paragraph("Social Determinants of Health as Predictors of 30-Day Readmission:", title_s))
story.append(Paragraph("A Machine Learning Risk Stratification Model", title_s))
story.append(HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=6))
story.append(Paragraph("Sai Manasa Adduru, MPH (Epidemiology), PharmD", author_s))
story.append(Paragraph("Kent State University · Health Outcomes Research", sub_s))
story.append(Paragraph("Methods: Gradient Boosting · SDOH Feature Engineering · ROC/AUC · Risk Stratification", sub_s))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=10))

story.append(Paragraph("ABSTRACT", h2_s))
story.append(Paragraph("<b>Background:</b> SDOH features are underutilized in readmission risk models. <b>Methods:</b> 5,000-patient synthetic EHR+SDOH dataset; compared clinical-only LR, clinical+SDOH LR, and GBM; 5-fold CV; navigation priority tiers. <b>Results:</b> SDOH improves AUC 0.610→0.642 (+5.2%). GBM CV AUC 0.605±0.013. ADI accounts for 23.4% of importance. Critical tier (5%) has 88.7% actual readmit rate. SDOH burden 0→4+: 13.8%→31.1% readmission. <b>Conclusion:</b> SDOH-informed scoring enables precise navigation triage to highest-need patients.", body_s))

kpi_data = [
    ["Metric","Value","Clinical-Only Baseline"],
    ["Model AUC (Clinical + SDOH LR)","0.642","0.610 (+5.2%)"],
    ["GBM 5-fold CV AUC","0.605 ± 0.013","—"],
    ["Critical tier actual readmit","88.7%","Overall: 23.0%"],
    ["SDOH share of feature importance","~40%","0% (not included)"],
    ["SDOH burden 0 vs 4+ readmit rate","13.8% vs 31.1%","2.3× increase"],
]
t = Table(kpi_data, colWidths=[2.8*inch, 1.6*inch, 2.3*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0),(-1,0), DARK_BLUE),("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT_BG,colors.white]),
    ("GRID",(0,0),(-1,-1),0.3,colors.lightgrey),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
]))
story.append(t); story.append(Spacer(1, 0.15*inch))

for fig_key, sec, caption in [
    ("fig1_roc_curves","MODEL PERFORMANCE — ROC & PRECISION-RECALL",
     "Figure 1. ROC curves (A): Adding SDOH improves AUC from 0.610→0.642. Precision-Recall (B): AP=0.351 vs 23% baseline."),
    ("fig2_feature_importance","FEATURE IMPORTANCE — SDOH VS CLINICAL",
     "Figure 2. Top 15 features by GBM importance. Red=SDOH, Blue=Clinical. ADI (SDOH) and age are top predictors."),
    ("fig3_sdoh_risk","SDOH BURDEN & NAVIGATION PRIORITY TIERS",
     "Figure 3. (A) Readmission rate by SDOH burden score. (B) Navigation priority tiers — Critical tier: 88.7% readmit."),
    ("fig4_risk_distribution","RISK SCORE DISTRIBUTION",
     "Figure 4. Risk score distributions by navigation status — navigated patients have higher risk (appropriate targeting)."),
]:
    story.append(Paragraph(sec, h2_s))
    img_path = f"output/figures/{fig_key}.png"
    pil = PILImage.open(img_path)
    w_px, h_px = pil.size
    max_w = 6.5 * inch
    img = RLImage(img_path, width=max_w, height=h_px * (max_w / w_px))
    story.append(img); story.append(Paragraph(caption, caption_s))

story.append(Paragraph("NAVIGATION PRIORITY TIERS", h2_s))
tier_data = [
    ["Tier","N (%)","Actual Readmit Rate","Recommended Action"],
    ["Critical","231 (5%)","88.7%","Immediate navigation — same-day"],
    ["High","884 (18%)","47.2%","Navigation within 24h of discharge"],
    ["Moderate","2,440 (49%)","17.9%","Scheduled follow-up call"],
    ["Low","1,445 (29%)","6.3%","Standard discharge instructions"],
]
t2 = Table(tier_data, colWidths=[1.2*inch,1.1*inch,1.5*inch,3.5*inch])
t2.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),DARK_BLUE),("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT_BG,colors.white]),
    ("GRID",(0,0),(-1,-1),0.3,colors.lightgrey),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
]))
story.append(t2)

doc.build(story)
print("PDF report written to report/sdoh_risk_report.pdf")
