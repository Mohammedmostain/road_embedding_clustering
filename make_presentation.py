"""
Weekly group meeting update — Road Segment Embedding Clustering
Short, plain-language, civil engineering framing. ~7 slides.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pathlib import Path

OUT  = Path("outputs")
PPTX = Path("Weekly_Update_Presentation.pptx")

NAVY    = RGBColor(0x1A, 0x1A, 0x2E)
BLUE    = RGBColor(0x0F, 0x3C, 0x78)
CYAN    = RGBColor(0x00, 0xB4, 0xD8)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
PALE    = RGBColor(0xCA, 0xE9, 0xFF)
GREEN   = RGBColor(0x06, 0xD6, 0x76)
YELLOW  = RGBColor(0xFF, 0xD1, 0x66)
RED     = RGBColor(0xFF, 0x49, 0x5C)

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

# ── helpers ───────────────────────────────────────────────────────────────────
def rect(slide, l, t, w, h, color):
    sh = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.line.fill.background()
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    return sh

def txt(slide, text, l, t, w, h, size=16, bold=False, color=WHITE,
        align=PP_ALIGN.LEFT, italic=False):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tb.word_wrap = True
    tf = tb.text_frame
    tf.word_wrap = True
    p  = tf.paragraphs[0]
    p.alignment = align
    r  = p.add_run()
    r.text = text
    r.font.size   = Pt(size)
    r.font.bold   = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return tb

def bullets(slide, items, l, t, w, h, size=15, gap=0.62):
    for i, (line, color) in enumerate(items):
        txt(slide, f"•   {line}", l, t + i * gap, w, gap + 0.1,
            size=size, color=color)

def header(slide, title):
    rect(slide, 0, 0, 13.33, 7.5, NAVY)
    rect(slide, 0, 1.1, 13.33, 0.05, CYAN)
    txt(slide, title, 0.5, 0.15, 12.5, 0.95, size=30, bold=True, color=WHITE)

def img(slide, path, l, t, w, h):
    if Path(path).exists():
        slide.shapes.add_picture(str(path), Inches(l), Inches(t), Inches(w), Inches(h))

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, 13.33, 7.5, NAVY)
rect(s, 0, 3.0, 13.33, 0.06, CYAN)
txt(s, "Can Satellite Imagery Improve\nTraffic Volume Prediction?",
    1, 0.9, 11.33, 2.0, size=36, bold=True, align=PP_ALIGN.CENTER)
txt(s, "Weekly Research Update  —  March 2026",
    1, 3.3, 11.33, 0.6, size=18, color=PALE, align=PP_ALIGN.CENTER)
txt(s, "Mohammed Mostain",
    1, 4.0, 11.33, 0.5, size=16, color=PALE, align=PP_ALIGN.CENTER)
txt(s, "Dataset: 726 road segments  |  Ottawa region  |  AAWDT as target variable",
    1, 5.0, 11.33, 0.5, size=13, color=PALE, align=PP_ALIGN.CENTER, italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — WHAT I DID THIS WEEK
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "What I Did This Week")

bullets(s, [
    ("Clustered 726 road segments using their satellite images",           WHITE),
    ("Tested 3 clustering methods: K-Means, HDBSCAN, and GMM",           WHITE),
    ("Compared each cluster set against Road Class for predicting AAWDT", WHITE),
    ("Ran a sweep to find the optimal number of clusters per method",      WHITE),
    ("Checked whether results are stable (sensitivity analysis)",          WHITE),
], 0.6, 1.4, 12.0, 4.0, size=17)

txt(s, "Core question: do road segments that look similar from satellite also have similar traffic volumes?",
    0.6, 6.3, 12.0, 0.8, size=14, color=YELLOW, italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — HOW IT WORKS (plain language)
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "How It Works  (Plain Language)")

steps = [
    ("Satellite image → 512 numbers",
     "A pre-trained AI model converts each road's satellite image into a list of 512 numbers\nthat capture what the road and its surroundings look like."),
    ("Reduce to 20 dimensions (UMAP)",
     "512 numbers is too many to cluster directly — distances become meaningless.\nUMAP compresses them to 20 while preserving which roads look similar."),
    ("Group similar roads  (Clustering)",
     "Roads that look alike end up in the same cluster.\nThree algorithms tested — each finds groups differently."),
    ("Test prediction quality  (XGBoost)",
     "Add cluster labels as features to a traffic prediction model.\nMeasure R² improvement vs using Road Class alone."),
]
for i, (title, body) in enumerate(steps):
    y = 1.35 + i * 1.45
    rect(s, 0.5, y, 0.55, 1.2, CYAN)
    txt(s, str(i+1), 0.5, y + 0.3, 0.55, 0.6,
        size=22, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    rect(s, 1.15, y, 11.5, 1.2, BLUE)
    txt(s, title, 1.3, y + 0.05, 11.2, 0.45, size=15, bold=True, color=CYAN)
    txt(s, body,  1.3, y + 0.5,  11.2, 0.65, size=13, color=WHITE)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — WHAT THE CLUSTERS LOOK LIKE
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "What the Clusters Look Like")

img(s, "outputs/03_umap_kmeans_clusters.png", 0.3,  1.3, 3.1, 2.7)
img(s, "outputs/04_umap_aawdt.png",           3.55, 1.3, 3.1, 2.7)
img(s, "outputs/07_bar_kmeans.png",           6.8,  1.3, 6.2, 2.7)

txt(s, "K-Means clusters (left) — each dot is a road segment.\nProximity = visual similarity.",
    0.3, 4.1, 3.1, 0.8, size=11, color=PALE, align=PP_ALIGN.CENTER)
txt(s, "Same map coloured by AAWDT.\nHigher traffic = warmer colour.",
    3.55, 4.1, 3.1, 0.8, size=11, color=PALE, align=PP_ALIGN.CENTER)
txt(s, "Mean AAWDT per cluster.\nClusters do separate roads by traffic volume.",
    6.8, 4.1, 6.2, 0.8, size=11, color=PALE, align=PP_ALIGN.CENTER)

txt(s, "Key observation: satellite image clusters align with traffic volume — "
       "roads that look similar tend to carry similar traffic.",
    0.5, 5.4, 12.33, 0.8, size=14, bold=True, color=YELLOW, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "Results")

# Left: table
rect(s, 0.5, 1.3, 7.0, 5.0, BLUE)
txt(s, "R² (higher = better AAWDT prediction)", 0.55, 1.35, 6.9, 0.5,
    size=13, bold=True, color=CYAN)

rows = [
    ("Road Class + GMM clusters",   "0.801", GREEN,  True),
    ("Road Class + K-Means",        "0.801", GREEN,  True),
    ("Road Class alone",            "0.781", YELLOW, False),
    ("K-Means clusters only",       "0.713", PALE,   False),
    ("HDBSCAN clusters only",       "0.711", PALE,   False),
    ("No clusters (baseline)",      "0.709", PALE,   False),
]
for i, (label, r2, color, highlight) in enumerate(rows):
    y = 1.95 + i * 0.7
    if highlight:
        rect(s, 0.55, y, 6.9, 0.65, RGBColor(0x06, 0x40, 0x60))
    txt(s, label, 0.65, y + 0.08, 5.5, 0.5, size=13, color=color)
    txt(s, r2, 5.9, y + 0.08, 0.9, 0.5, size=14, bold=True,
        color=color, align=PP_ALIGN.RIGHT)

# Right: plain language summary
rect(s, 7.8, 1.3, 5.2, 5.0, BLUE)
txt(s, "What this means", 7.9, 1.35, 5.0, 0.5,
    size=15, bold=True, color=CYAN)
bullets(s, [
    ("Adding satellite clusters ON TOP of Road Class improves R² from 0.781 to 0.801", WHITE),
    ("That +0.020 represents the visual context Road Class does not capture", WHITE),
    ("Clusters alone (~0.71) do not replace Road Class — they supplement it", YELLOW),
    ("Best cluster count: GMM with 21 groups or K-Means with 18 groups", WHITE),
], 7.9, 1.9, 5.0, 4.0, size=13, gap=0.95)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — WHY CLUSTERS HELP (civil eng framing)
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "Why Does Satellite Imagery Add Value?")

rect(s, 0.5, 1.3, 5.7, 4.8, BLUE)
txt(s, "Road Class tells you...", 0.6, 1.38, 5.5, 0.5,
    size=15, bold=True, color=YELLOW)
bullets(s, [
    ("Functional hierarchy: arterial vs collector vs local",     WHITE),
    ("Design standard and speed",                                WHITE),
    ("Does NOT capture surrounding land use",                    RED),
    ("Two Minor Arterials — one in dense city core, one in quiet suburb — get the same label", RED),
], 0.6, 1.95, 5.5, 3.8, size=13, gap=0.9)

rect(s, 6.5, 1.3, 6.4, 4.8, BLUE)
txt(s, "Satellite imagery tells you...", 6.6, 1.38, 6.2, 0.5,
    size=15, bold=True, color=CYAN)
bullets(s, [
    ("What the road environment actually looks like",                WHITE),
    ("Dense urban core vs suburban strip vs industrial estate",      WHITE),
    ("Number of lanes, intersections, parking visible from above",   WHITE),
    ("This visual context directly affects how much traffic uses the road", GREEN),
], 6.6, 1.95, 6.2, 3.8, size=13, gap=0.9)

txt(s, "The +0.020 R² gain = the visual land-use signal that Road Class treats as identical.",
    0.5, 6.35, 12.33, 0.7, size=14, color=YELLOW, italic=True, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — NEXT STEPS
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, "Next Steps")

items = [
    ("Extend cluster sweep to GMM g=40",
     "The benefit curve was still noisy at g=30. Want to confirm whether g=21 is truly optimal or if higher counts help further.", CYAN),
    ("Test on held-out data",
     "Current R² is measured on the same 726 segments used to build the clusters. "
     "Need an independent test set to confirm the +0.020 gain is not overfitting.", YELLOW),
    ("Interpret what each cluster represents",
     "Label each cluster manually — e.g. 'dense urban arterial', 'suburban collector', 'rural highway'. "
     "This would make the clusters meaningful to planners.", GREEN),
    ("Write up methodology section",
     "Document the pipeline formally for the report/thesis chapter.", PALE),
]
for i, (title, body, color) in enumerate(items):
    y = 1.35 + i * 1.4
    rect(s, 0.5, y, 0.12, 1.2, color)
    txt(s, f"{i+1}.  {title}", 0.75, y + 0.05, 12.0, 0.48,
        size=15, bold=True, color=color)
    txt(s, body, 0.75, y + 0.55, 12.0, 0.65, size=13, color=WHITE)

# ══════════════════════════════════════════════════════════════════════════════
prs.save(str(PPTX))
print(f"Saved: {PPTX}  ({len(prs.slides)} slides)")
