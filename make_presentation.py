"""
Weekly group meeting — Road Segment Embedding Clustering
Highly visual, minimal text, speaker notes carry the detail.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree
from pathlib import Path
import copy

PPTX = Path("Weekly_Update_Presentation.pptx")
OUT  = Path("outputs")

# ── palette ───────────────────────────────────────────────────────────────────
NAVY   = RGBColor(0x0D, 0x1B, 0x2A)
DARK   = RGBColor(0x11, 0x28, 0x40)
BLUE   = RGBColor(0x1B, 0x4F, 0x8A)
CYAN   = RGBColor(0x00, 0xC2, 0xE0)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
PALE   = RGBColor(0xB8, 0xD8, 0xF0)
GREEN  = RGBColor(0x00, 0xC9, 0x74)
YELLOW = RGBColor(0xFF, 0xCA, 0x3A)
RED    = RGBColor(0xFF, 0x4D, 0x6D)
GRAY   = RGBColor(0x2C, 0x3E, 0x55)

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

# ── primitives ────────────────────────────────────────────────────────────────
def rect(slide, l, t, w, h, fill, radius=False):
    typ = 5 if radius else 1   # 5 = roundRect, 1 = rect
    sh  = slide.shapes.add_shape(typ, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.line.fill.background()
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if radius:
        sh.adjustments[0] = 0.08
    return sh

def line(slide, l, t, w, h, color, width_pt=2):
    from pptx.util import Pt as UPt
    connector = slide.shapes.add_connector(1, Inches(l), Inches(t), Inches(l+w), Inches(t+h))
    connector.line.color.rgb = color
    connector.line.width = UPt(width_pt)
    return connector

def txt(slide, text, l, t, w, h, size=16, bold=False, color=WHITE,
        align=PP_ALIGN.LEFT, italic=False, wrap=True):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tb.word_wrap = wrap
    tf = tb.text_frame
    tf.word_wrap = wrap
    p  = tf.paragraphs[0]
    p.alignment = align
    r  = p.add_run()
    r.text = text
    r.font.size   = Pt(size)
    r.font.bold   = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return tb

def img(slide, path, l, t, w, h):
    if Path(path).exists():
        slide.shapes.add_picture(str(path), Inches(l), Inches(t), Inches(w), Inches(h))

def add_notes(slide, notes_text):
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = notes_text

def arrow_right(slide, l, t, color=CYAN, size=0.35):
    """Draw a simple right-pointing arrow using a textbox."""
    txt(slide, "➔", l, t, size + 0.1, size + 0.1,
        size=int(size * 60), color=color, align=PP_ALIGN.CENTER)

def icon_card(slide, icon, label, l, t, w=2.3, h=2.0,
              bg=BLUE, icon_size=36, label_size=13, label_color=WHITE):
    rect(slide, l, t, w, h, bg, radius=True)
    txt(slide, icon,  l, t + 0.22, w, 0.9,
        size=icon_size, align=PP_ALIGN.CENTER, color=WHITE)
    txt(slide, label, l, t + h - 0.72, w, 0.65,
        size=label_size, bold=True, color=label_color,
        align=PP_ALIGN.CENTER)

def metric_card(slide, value, label, sublabel, l, t, w=3.5, h=1.9,
                val_color=GREEN, bg=GRAY):
    rect(slide, l, t, w, h, bg, radius=True)
    txt(slide, value,    l, t + 0.1,  w, 0.9,
        size=44, bold=True, color=val_color, align=PP_ALIGN.CENTER)
    txt(slide, label,    l, t + 1.0,  w, 0.5,
        size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txt(slide, sublabel, l, t + 1.48, w, 0.38,
        size=11, color=PALE, align=PP_ALIGN.CENTER, italic=True)

def divider(slide, y=1.12):
    rect(slide, 0, y, 13.33, 0.05, CYAN)

def slide_bg(slide):
    rect(slide, 0, 0, 13.33, 7.5, NAVY)

def slide_title(slide, title, y=0.18):
    txt(slide, title, 0.55, y, 12.0, 0.88, size=28, bold=True, color=WHITE)

# ══════════════════════════════════════════════════════════════════════════════
# 1 — TITLE
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_bg(s)

# accent blocks — top-left and bottom-right decorative
rect(s, 0,    0,    3.0,  0.5, CYAN)
rect(s, 0,    0.5,  1.5,  7.0, BLUE)
rect(s, 10.3, 7.0,  3.03, 0.5, CYAN)
rect(s, 11.8, 0,    1.53, 7.0, BLUE)

# satellite emoji as hero visual
txt(s, "🛰️", 4.5, 0.6, 4.33, 2.8, size=120, align=PP_ALIGN.CENTER, color=WHITE)

txt(s, "Can Satellite Imagery\nImprove Traffic Prediction?",
    1.8, 3.5, 9.73, 2.0,
    size=34, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
rect(s, 1.8, 5.55, 9.73, 0.05, CYAN)
txt(s, "Weekly Research Update  ·  Mohammed Mostain  ·  March 2026",
    1.8, 5.65, 9.73, 0.5, size=14, color=PALE, align=PP_ALIGN.CENTER)

add_notes(s,
"Welcome everyone. This week I focused on testing whether satellite imagery "
"can improve our predictions of Annual Average Weekday Daily Traffic — AAWDT — "
"beyond what Road Class alone tells us. I'll walk through what I did, how it works, "
"and the key results.")

# ══════════════════════════════════════════════════════════════════════════════
# 2 — THIS WEEK (icon cards)
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_bg(s)
slide_title(s, "This Week")
divider(s)

cards = [
    ("🛰️", "Satellite\nClustering",   BLUE),
    ("⚙️", "3 Methods\nTested",        BLUE),
    ("📊", "Road Class\nComparison",   BLUE),
    ("🔍", "Optimal k\nSweep",         BLUE),
    ("✅", "Stability\nChecked",        BLUE),
]
for i, (icon, label, bg_c) in enumerate(cards):
    icon_card(s, icon, label, 0.4 + i * 2.55, 1.5, w=2.3, h=2.2, bg=bg_c)

# one-line summary
rect(s, 0.4, 4.1, 12.53, 1.0, GRAY, radius=True)
txt(s, "Do road segments that look similar from above also carry similar traffic volumes?",
    0.55, 4.2, 12.2, 0.8, size=18, bold=True, color=YELLOW, align=PP_ALIGN.CENTER)

# bottom visual — UMAP teaser
img(s, "outputs/03_umap_kmeans_clusters.png", 0.4,  5.3, 3.2, 2.0)
img(s, "outputs/04_umap_aawdt.png",           3.8,  5.3, 3.2, 2.0)
img(s, "outputs/10_bar_roadclass.png",        7.2,  5.3, 3.0, 2.0)
img(s, "outputs/11_r2_heatmap.png",          10.35, 5.3, 2.85, 2.0)

add_notes(s,
"This week I ran the full pipeline on our 726 road segments. "
"I tested three clustering algorithms — K-Means, HDBSCAN, and GMM — all applied to "
"satellite image embeddings. I then compared each clustering approach against Road Class "
"for predicting AAWDT. I also ran a sweep to find the optimal number of clusters per method, "
"and did a sensitivity analysis to make sure the results are stable.")

# ══════════════════════════════════════════════════════════════════════════════
# 3 — THE PIPELINE (visual flow)
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_bg(s)
slide_title(s, "The Approach")
divider(s)

steps = [
    ("🛰️",  "Satellite\nImage",     "512 numbers\nper road"),
    ("📐",  "Compress\n(UMAP)",      "512 → 20\ndimensions"),
    ("🔵",  "Group\nSimilar Roads",  "K-Means\nHDBSCAN  GMM"),
    ("📈",  "Predict\nAAWDT",        "XGBoost\n5-fold CV"),
]

box_w, box_h = 2.5, 3.2
start_x = 0.45
gap     = 0.5   # space for arrow

for i, (icon, title, sub) in enumerate(steps):
    x = start_x + i * (box_w + gap)

    # step box
    rect(s, x, 1.5, box_w, box_h, BLUE, radius=True)

    # step number pill
    rect(s, x + box_w/2 - 0.25, 1.35, 0.5, 0.5, CYAN, radius=True)
    txt(s, str(i+1), x + box_w/2 - 0.25, 1.35, 0.5, 0.5,
        size=14, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    txt(s, icon,  x, 1.65,       box_w, 1.1,
        size=46, align=PP_ALIGN.CENTER, color=WHITE)
    txt(s, title, x, 2.75,       box_w, 0.85,
        size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    rect(s, x + 0.2, 3.6, box_w - 0.4, 0.04, CYAN)
    txt(s, sub,   x, 3.68,       box_w, 0.85,
        size=13, color=PALE, align=PP_ALIGN.CENTER)

    if i < 3:
        ax = x + box_w + 0.08
        txt(s, "➔", ax, 2.6, gap - 0.08, 0.7,
            size=28, color=CYAN, align=PP_ALIGN.CENTER)

# bottom note
rect(s, 0.45, 5.05, 12.53, 0.75, GRAY, radius=True)
txt(s, "⚠️   Clustering must happen AFTER compression — raw 512-dim distances are nearly uniform (curse of dimensionality)",
    0.65, 5.1, 12.2, 0.65, size=14, color=YELLOW, align=PP_ALIGN.CENTER)

add_notes(s,
"A pre-trained satellite image model converts each road's aerial photo into 512 numbers — "
"essentially a fingerprint of what the road looks like from above. "
"We can't cluster 512 dimensions directly because all points end up equally far apart — "
"this is the curse of dimensionality. "
"UMAP compresses those 512 numbers down to 20 while preserving which roads look similar. "
"We then apply three clustering algorithms to group roads by visual similarity. "
"Finally, we add those cluster labels as features to an XGBoost model and measure "
"whether AAWDT prediction improves.")

# ══════════════════════════════════════════════════════════════════════════════
# 4 — CLUSTER MAPS
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_bg(s)
slide_title(s, "Roads Grouped by Visual Similarity")
divider(s)

img(s, "outputs/03_umap_kmeans_clusters.png", 0.3,  1.3, 4.0, 3.3)
img(s, "outputs/04_umap_aawdt.png",           4.5,  1.3, 4.0, 3.3)
img(s, "outputs/07_bar_kmeans.png",           8.7,  1.3, 4.4, 3.3)

labels = [
    (0.3,  "Clusters", CYAN),
    (4.5,  "AAWDT",    YELLOW),
    (8.7,  "Mean Traffic per Cluster", GREEN),
]
for x, label, color in labels:
    txt(s, label, x, 4.65, 4.0, 0.45, size=13, bold=True,
        color=color, align=PP_ALIGN.CENTER)

# insight callout
rect(s, 0.3, 5.3, 12.73, 1.3, GRAY, radius=True)
txt(s, "💡", 0.5, 5.4, 0.8, 1.1, size=36, align=PP_ALIGN.CENTER)
txt(s, "Clusters that look similar visually also carry similar traffic volumes.\n"
       "The AAWDT heatmap and the cluster map align — this is not a coincidence.",
    1.3, 5.4, 11.6, 1.1, size=15, color=WHITE)

add_notes(s,
"Each dot is one road segment. Dots close together look similar from satellite. "
"On the left you can see the clusters found by K-Means. "
"In the middle, I've coloured the same map by actual AAWDT — warmer colours mean higher traffic. "
"Notice how the warm-coloured dots tend to cluster together. "
"On the right, the bar chart shows mean AAWDT per cluster — there are clear differences between groups. "
"This tells us the clusters are capturing something real about traffic, even though they only used imagery.")

# ══════════════════════════════════════════════════════════════════════════════
# 5 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_bg(s)
slide_title(s, "Results")
divider(s)

# hero metric cards
metric_card(s, "0.801", "Road Class + GMM",  "Best R² achieved",   0.4,  1.4, 3.8, 2.0, GREEN,  GRAY)
metric_card(s, "0.781", "Road Class alone",   "Baseline R²",       4.5,  1.4, 3.8, 2.0, YELLOW, GRAY)
metric_card(s, "+0.020","Improvement",        "From adding clusters",8.6, 1.4, 4.13, 2.0, CYAN,  GRAY)

# visual R2 bar comparison
rect(s, 0.4, 3.7, 12.53, 3.55, DARK, radius=True)
txt(s, "R²  by feature set", 0.7, 3.8, 4.0, 0.5, size=13, bold=True, color=CYAN)

bar_data = [
    ("RC + GMM clusters",   0.801, GREEN,  True),
    ("RC + K-Means",        0.801, GREEN,  True),
    ("Road Class only",     0.781, YELLOW, False),
    ("K-Means only",        0.713, PALE,   False),
    ("HDBSCAN only",        0.711, PALE,   False),
    ("Baseline (no clusters)", 0.709, GRAY, False),
]
bar_min, bar_max = 0.68, 0.815
bar_area_w = 7.5
bar_h_each = 0.44
bar_start_x = 5.2
bar_start_y = 3.8

for i, (label, val, color, highlight) in enumerate(bar_data):
    y = bar_start_y + i * (bar_h_each + 0.07)
    bar_w = bar_area_w * (val - bar_min) / (bar_max - bar_min)

    bg_c = RGBColor(0x1B, 0x3A, 0x55) if highlight else DARK
    rect(s, bar_start_x, y, bar_area_w, bar_h_each, bg_c, radius=False)
    rect(s, bar_start_x, y, max(bar_w, 0.05), bar_h_each, color, radius=False)

    txt(s, label, 0.55, y + 0.05, 4.55, bar_h_each - 0.05,
        size=12, color=WHITE if highlight else PALE, bold=highlight)
    txt(s, f"{val:.3f}", bar_start_x + bar_w + 0.08, y + 0.05, 0.7, bar_h_each - 0.05,
        size=12, bold=highlight, color=color)

# baseline reference line
ref_x = bar_start_x + bar_area_w * (0.781 - bar_min) / (bar_max - bar_min)
rect(s, ref_x, bar_start_y, 0.025, 6 * (bar_h_each + 0.07) - 0.07, YELLOW)

add_notes(s,
"The three big numbers at the top tell the story. "
"Road Class alone gives R² of 0.781. "
"Adding GMM satellite clusters on top brings it to 0.801 — breaking the 0.80 barrier. "
"That's a +0.020 improvement. "
"The bars show all six feature combinations. "
"The yellow vertical line marks the Road Class baseline. "
"Anything to the right of that line means satellite clusters added value. "
"Importantly, clusters alone — without Road Class — only reach about 0.71. "
"The clusters supplement Road Class, they don't replace it.")

# ══════════════════════════════════════════════════════════════════════════════
# 6 — WHY IT HELPS (split visual)
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_bg(s)
slide_title(s, "Why Does Imagery Add Value?")
divider(s)

# left panel — Road Class
rect(s, 0.3, 1.3, 5.9, 5.0, BLUE, radius=True)
txt(s, "🏷️", 0.3, 1.4, 5.9, 1.3, size=60, align=PP_ALIGN.CENTER)
txt(s, "Road Class", 0.3, 2.7, 5.9, 0.6, size=20, bold=True,
    color=YELLOW, align=PP_ALIGN.CENTER)

rc_points = ["Functional hierarchy", "Design standard", "Same label everywhere", "Ignores surroundings"]
for i, pt in enumerate(rc_points):
    txt(s, pt, 0.55, 3.4 + i * 0.65, 5.4, 0.6, size=14, color=WHITE)

# cross icon on "ignores surroundings"
txt(s, "✗", 0.35, 3.4 + 3 * 0.65, 0.4, 0.6, size=16, bold=True, color=RED)

# right panel — Satellite
rect(s, 6.6, 1.3, 6.43, 5.0, BLUE, radius=True)
txt(s, "🛰️", 6.6, 1.4, 6.43, 1.3, size=60, align=PP_ALIGN.CENTER)
txt(s, "Satellite Imagery", 6.6, 2.7, 6.43, 0.6, size=20, bold=True,
    color=CYAN, align=PP_ALIGN.CENTER)

sat_points = ["Dense urban vs suburban", "Land use context", "Road geometry from above", "Visual surroundings"]
for i, pt in enumerate(sat_points):
    txt(s, "✓", 6.6, 3.4 + i * 0.65, 0.5, 0.6, size=16, bold=True, color=GREEN)
    txt(s, pt, 7.1, 3.4 + i * 0.65, 5.7, 0.6, size=14, color=WHITE)

# centre "vs" divider
txt(s, "VS", 5.9, 3.3, 0.9, 0.9, size=22, bold=True, color=PALE, align=PP_ALIGN.CENTER)
rect(s, 6.18, 1.3, 0.04, 5.0, GRAY)

# bottom callout
rect(s, 0.3, 6.5, 12.73, 0.75, GRAY, radius=True)
txt(s, "Two Minor Arterials — one in a dense city core, one in a quiet suburb — "
       "get the same Road Class label but very different AAWDT. Imagery tells them apart.",
    0.55, 6.55, 12.4, 0.65, size=13, color=YELLOW, align=PP_ALIGN.CENTER, italic=True)

add_notes(s,
"Road Class is a label assigned by planners. It tells you the functional purpose of the road — "
"arterial, collector, local — but it applies that label uniformly regardless of context. "
"A Minor Arterial in the dense Ottawa city core and a Minor Arterial at the edge of the suburbs "
"get the same label, but they carry very different amounts of traffic. "
"Satellite imagery can see the difference. It captures the surrounding built environment — "
"how dense the buildings are, whether there's a big intersection, a shopping centre, or just fields. "
"That visual context directly influences traffic demand, and that's the extra signal the clusters bring.")

# ══════════════════════════════════════════════════════════════════════════════
# 7 — NEXT STEPS
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_bg(s)
slide_title(s, "Next Steps")
divider(s)

cards_ns = [
    ("🔢", "Extend GMM\nSweep to g=40",      "Confirm optimal\ncluster count",       BLUE),
    ("📋", "Held-Out\nTest Set",              "Validate on\nnew data",                BLUE),
    ("🗺️", "Label Each\nCluster",             "e.g. 'Dense urban\narterial'",          BLUE),
    ("📝", "Write Up\nMethodology",           "Thesis chapter\nprep",                 BLUE),
]

for i, (icon, title, sub, bg_c) in enumerate(cards_ns):
    col = i % 2
    row = i // 2
    x = 1.2 + col * 5.8
    y = 1.55 + row * 2.65

    rect(s, x, y, 5.0, 2.35, bg_c, radius=True)
    txt(s, icon,  x, y + 0.1,  5.0, 1.0,  size=42, align=PP_ALIGN.CENTER)
    txt(s, title, x, y + 1.1,  5.0, 0.7,  size=16, bold=True,
        color=CYAN, align=PP_ALIGN.CENTER)
    txt(s, sub,   x, y + 1.78, 5.0, 0.52, size=12,
        color=PALE, align=PP_ALIGN.CENTER, italic=True)

add_notes(s,
"Four things I plan to do next. "
"First, the GMM benefit curve was still noisy at g=30 — I want to extend to g=40 to confirm g=21 is truly optimal. "
"Second, the current R² is measured on the same 726 segments used to build the clusters, "
"so I need to validate on a held-out set to confirm the gains aren't from overfitting. "
"Third, I want to manually label what each cluster actually represents in civil engineering terms — "
"dense urban, suburban commercial, rural, and so on — to make the clusters interpretable to planners. "
"Fourth, I'll start writing this up as a methodology section for the thesis.")

# ══════════════════════════════════════════════════════════════════════════════
prs.save(str(PPTX))
print(f"Saved: {PPTX}  ({len(prs.slides)} slides)")
