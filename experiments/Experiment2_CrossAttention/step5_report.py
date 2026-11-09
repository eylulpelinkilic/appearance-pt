"""Experiment 2, step 5: the PDF report, built only from the saved outputs (reports/metrics.json and the figures drawn
from it), in the structure and style of the pilot's report (pilot_concat_report.pdf).

Writes <out>/reports/sonata_cross_attention_report.pdf and <out>/reports/figures/*.png. While metrics.json says
"preliminary" (the smoke run, or a full run that did not finish), the title page and every page footer say
PRELIMINARY and only measured numbers are shown; rerun this step once step 4 has finished to get the final report.
Fonts: DejaVu, shipped with matplotlib, so the same PDF comes out on Kaggle and locally.

Usage: python experiments/Experiment2_CrossAttention/step5_report.py [--out /kaggle/working/Experiment2_CrossAttention]
Needs matplotlib and reportlab (the notebook installs reportlab).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path("/kaggle/working/Experiment2_CrossAttention")
PDF_NAME = "sonata_cross_attention_report.pdf"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # the pilot report's palette (validated categorical slots 1-3)
INK, INK2, MUTED, GRID, FROZEN_FILL, FROZEN_EDGE = "#0b0b0b", "#52514e", "#a9a7a0", "#e6e4df", "#f3f2ee", "#c9c7c0"
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#c9c7c0", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "savefig.dpi": 250,
})


def esc(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def s2(x) -> str:
    return "n/a" if x is None else f"{x:+.2f}".replace("-", "−")


def plural(n, word: str) -> str:
    return word if n == 1 else word + "s"


def f2(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


# ---- figures -----------------------------------------------------------------------------------------------------

def _box(ax, x, y, w, h, head, sub="", kind="frozen"):
    """kind: "trained" (receives gradients), "target", or "frozen" (fixed input or frozen module)."""
    fc, ec, lw = {"trained": ("#e8f0fb", BLUE, 1.6), "target": ("#fdeee6", ORANGE, 1.3), "frozen": (FROZEN_FILL, FROZEN_EDGE, 0.8)}[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h * (0.74 if sub else 0.5), head, ha="center", va="center", fontsize=7.6, color=INK, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.36, sub, ha="center", va="center", fontsize=6.4, color=INK2, linespacing=1.2)
    if kind == "trained":
        ax.text(x + w, y + h + 0.6, "receives gradients", ha="right", va="bottom", fontsize=6.0, color=BLUE, style="italic")


def _arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK2, shrinkA=0, shrinkB=0))


def _band(ax, y, label, note):
    ax.plot([0, 102], [y + 2.5, y + 2.5], color=FROZEN_EDGE, lw=0.5)
    ax.text(0, y, label, fontsize=8.8, color=INK, weight="bold", va="top")
    ax.text(0, y - 4.0, note, fontsize=6.6, color=INK2, va="top")


def fig_pipeline(path: Path) -> None:
    """Teacher training (A) and student training (B) in one figure, point-only evaluation (C) in a second one; drawn at
    the page's text width (6.5 in), so the type is printed at its nominal size."""
    fig, ax = plt.subplots(figsize=(6.5, 5.9))
    ax.set_xlim(-0.5, 102.5)
    ax.set_ylim(-7, 100)
    ax.axis("off")
    _band(ax, 97, "A   Teacher training", "Sonata and DINOv2 frozen · train scenes only, no labels · points seen by ≥ 3 cameras")
    _box(ax, 0, 74, 16, 11, "Input view 1", "DINOv2 patch\nfeature y₁")
    _box(ax, 0, 60, 16, 11, "Input view 2", "DINOv2 patch\nfeature y₂")
    _box(ax, 0, 46, 16, 11, "Point p", "frozen Sonata\nfeature s(p)")
    _box(ax, 21, 55, 23, 27, "Cross-attention", "query W_q s(p)\nkeys W_k y₁, W_k y₂\nvalues y₁, y₂\n(unchanged)", "trained")
    _box(ax, 49, 66, 21, 14, "Weighted DINO", "a₁y₁ + a₂y₂,\na = softmax(q·k)")
    _box(ax, 49, 46, 21, 14, "Held-out view 3", "DINOv2 feature y₃:\nthe teacher's target", "target")
    _box(ax, 74, 54, 28, 18, "Loss", "1 − cos(a₁y₁ + a₂y₂, y₃)\ngradients reach\nW_q and W_k only")
    for a, b in (((16.5, 79.5), (20.5, 75)), ((16.5, 65.5), (20.5, 68)), ((16.5, 51.5), (20.5, 61)), ((44.5, 73), (48.5, 73)),
                 ((70.5, 73), (73.5, 67)), ((70.5, 53), (73.5, 59))):
        _arrow(ax, a, b)
    _band(ax, 37, "B   Student training", "continued training from Sonata · point-only inputs · teacher frozen")
    _box(ax, 0, 11, 15, 14, "Point cloud", "crop:\nxyz, colour,\nnormal")
    _box(ax, 19, 11, 18, 14, "Sonata student", "initialised from\nsonata.pth", "trained")
    _box(ax, 41, 11, 15, 14, "Feature f(p)", "upcast layer\n(1232)")
    _box(ax, 60, 11, 13, 14, "Head h", "1232 → 1024", "trained")
    _box(ax, 77, 11, 25, 14, "Loss", "1 − cos(h(f(p)), t(p))")
    _box(ax, 77, -5.5, 25, 12.5, "Teacher target t(p)", "frozen teacher over all\ncameras of p, detached", "target")
    for a, b in (((15.5, 18), (18.5, 18)), ((37.5, 18), (40.5, 18)), ((56.5, 18), (59.5, 18)), ((73.5, 18), (76.5, 18)),
                 ((89.5, 7.5), (89.5, 10.5))):
        _arrow(ax, a, b)
    ax.text(0, 5, "the teacher, the head and the images\nare used for training only", fontsize=6.6, color=INK2, va="top", linespacing=1.25)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 1.75))
    ax.set_xlim(-0.5, 102.5)
    ax.set_ylim(-2, 27)
    ax.axis("off")
    _band(ax, 24, "C   Point-only evaluation", "teacher and head discarded · no images, cameras, DINOv2 features or teacher outputs")
    for x, w, head, sub, kind in ((0, 15, "Point cloud", "xyz, colour,\nnormal", "frozen"), (19, 14, "Student", "frozen", "frozen"),
                                  (37, 16, "Feature f(p)", "upcast (1232)", "frozen"),
                                  (57, 18, "Linear probe", "pilot's recipe,\nlabelled sample", "trained"),
                                  (79, 23, "Official mIoU", "all vertices of the\nval scenes", "frozen")):
        _box(ax, x, -0.5, w, 12.5, head, sub, kind)
    for a, b in (((15.5, 5.75), (18.5, 5.75)), ((33.5, 5.75), (36.5, 5.75)), ((53.5, 5.75), (56.5, 5.75)), ((75.5, 5.75), (78.5, 5.75))):
        _arrow(ax, a, b)
    fig.savefig(path.with_name(path.stem + "_eval.png"), bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def fig_results(m: dict, path: Path) -> None:
    rows = m["rows"]
    order = [("S", MUTED), ("S+", ORANGE), ("T", BLUE)]
    lo_ = min(rows[c]["top1"] for c, _ in order) - 3
    hi_ = max(rows[c]["top1"] for c, _ in order) + 3
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 1.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for j, (c, col) in enumerate(order):
        r = rows[c]
        a1.errorbar([r["top1"]], [j], xerr=[[r["top1_std"]], [r["top1_std"]]], fmt="o", color=col, ms=6, capsize=2, lw=1)
        a1.text(r["top1"] + r["top1_std"] + 0.06 * (hi_ - lo_), j - 0.22, f"{r['top1']:.2f}", va="bottom", fontsize=6.8, color=INK)
    a1.set_yticks(range(3), [rows[c]["name"].replace(" (concatenation)", "\n(concatenation)") for c, _ in order])
    a1.set_ylim(2.6, -0.6)
    a1.set_xlim(lo_, hi_)
    a1.grid(axis="x", color=GRID, lw=0.6)
    a1.set_title("Official val top-1 mIoU (%)")
    a1.tick_params(axis="y", labelsize=6.6, colors=INK)
    d = m["diffs"]
    pairs = [("S+ - S", "Concatenation − Sonata", ORANGE), ("T - S", "Student − Sonata", BLUE), ("T - S+", "Student − concatenation", AQUA)]
    for j, (k, label, col) in enumerate(pairs):
        x = d[k]
        a2.plot([x["lo"], x["hi"]], [j, j], color=col, lw=1.8, solid_capstyle="round")
        a2.scatter([x["diff"]], [j], s=22, color=col, edgecolor="white", lw=0.7, zorder=3)
        a2.text(x["hi"], j - 0.22, "  " + s2(x["diff"]), va="bottom", fontsize=6.8, color=INK)
    a2.axvline(0, color="#8c8a84", lw=0.9)
    a2.set_yticks(range(3), [p[1] for p in pairs])
    a2.set_ylim(2.6, -0.6)
    a2.grid(axis="x", color=GRID, lw=0.6)
    a2.set_title("Differences (points), 95 % interval")
    a2.tick_params(axis="y", labelsize=6.6, colors=INK)
    xs = [d[k]["lo"] for k, _, _ in pairs] + [d[k]["hi"] for k, _, _ in pairs] + [0]
    a2.set_xlim(min(xs) - 1, max(xs) + 2)
    fig.tight_layout(w_pad=2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def fig_teacher(diag: dict, path: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.1))
    curve = diag.get("curve") or []
    if curve:
        ep = [c["epoch"] for c in curve]
        a1.plot(ep, [c["dev_teacher_cos"] for c in curve], color=BLUE, lw=1.6, marker="o", ms=2.5, label="cross-attention teacher")
        a1.plot(ep, [c["dev_uniform_cos"] for c in curve], color=MUTED, lw=1.4, ls="--", label="uniform average")
        a1.legend(frameon=False, fontsize=6.3)
        a1.set_xlabel("teacher epoch", fontsize=6.8)
    a1.set_title("Dev triples: cosine to the held-out view")
    a1.grid(axis="y", color=GRID, lw=0.6)
    per = sorted(diag.get("per_scene") or [], key=lambda p: p["teacher_cos"] - p["uniform_cos"])
    diffs = [p["teacher_cos"] - p["uniform_cos"] for p in per]
    a2.axvline(0, color="#8c8a84", lw=0.9)
    a2.scatter(diffs, range(len(per)), s=16, color=BLUE, zorder=3)
    a2.set_yticks([])
    a2.set_ylim(len(per) - 0.4, -0.6)
    a2.set_ylabel(f"{len(per)} dev scenes, sorted", fontsize=6.8)
    a2.set_xlabel("teacher − uniform average (cosine)", fontsize=6.8)
    a2.set_title("Per dev scene")
    a2.grid(axis="x", color=GRID, lw=0.6)
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    for axis in (a1.yaxis, a2.xaxis):
        axis.set_major_formatter(FormatStrFormatter("%.4f"))
        axis.set_major_locator(MaxNLocator(5))
    a1.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(w_pad=2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def fig_student(log: dict, path: Path) -> None:
    curve = log.get("curve") or []
    fig, ax = plt.subplots(figsize=(7.0, 1.8))
    if curve:
        steps = np.array([c["step"] for c in curve])
        loss = np.array([c["loss"] for c in curve])
        k = max(1, len(loss) // 60)
        smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
        ax.plot(steps, loss, color="#d9d7d0", lw=0.6)
        ax.plot(steps[k - 1:], smooth, color=BLUE, lw=1.6)
    ax.set_xlabel("student step", fontsize=6.8)
    ax.set_ylabel("1 − cos to target", fontsize=6.8)
    ax.set_title("Student training loss (grey: per step; blue: moving average)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


# ---- the document ------------------------------------------------------------------------------------------------

def build(m: dict, out_dir: Path) -> Path:
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    for name, file in (("Serif", "DejaVuSerif"), ("Serif-Bold", "DejaVuSerif-Bold"), ("Serif-Italic", "DejaVuSerif-Italic"),
                       ("Serif-BoldItalic", "DejaVuSerif-BoldItalic"), ("Sans", "DejaVuSans"), ("Sans-Bold", "DejaVuSans-Bold"),
                       ("Sans-Italic", "DejaVuSans-Oblique"), ("Sans-BoldItalic", "DejaVuSans-BoldOblique")):
        pdfmetrics.registerFont(TTFont(name, str(FONTS / f"{file}.ttf")))
    for fam in ("Serif", "Sans"):
        registerFontFamily(fam, normal=fam, bold=f"{fam}-Bold", italic=f"{fam}-Italic", boldItalic=f"{fam}-BoldItalic")

    final = m["status"] == "final"
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    ink, ink2, rule, shade, blue = (colors.HexColor(c) for c in ("#1b1a18", "#55534e", "#cfcdc6", "#f4f3ef", BLUE))
    W = A4[0] - 4.4 * cm
    body = ParagraphStyle("body", fontName="Serif", fontSize=9.0, leading=13.4, textColor=ink, spaceAfter=6)
    h1 = ParagraphStyle("h1", fontName="Sans-Bold", fontSize=13.5, leading=17, textColor=ink, spaceBefore=12, spaceAfter=6)
    h2 = ParagraphStyle("h2", fontName="Sans-Bold", fontSize=10.4, leading=13.5, textColor=ink, spaceBefore=9, spaceAfter=4)
    caption = ParagraphStyle("caption", fontName="Sans", fontSize=7.4, leading=10, textColor=ink2, spaceBefore=3, spaceAfter=9)
    cell = ParagraphStyle("cell", fontName="Sans", fontSize=7.2, leading=9.3, textColor=ink)
    cellb = ParagraphStyle("cellb", parent=cell, fontName="Sans-Bold")
    boxtext = ParagraphStyle("boxtext", parent=body, fontSize=8.8, leading=12.8, spaceAfter=3)
    story, pending = [], []

    def P(text, style=body):
        return Paragraph(text, style)

    def add(fl):  # keep a heading on the page of what follows it
        if isinstance(fl, Paragraph) and fl.style.name in ("h1", "h2"):
            pending.append(fl)
        elif pending:
            story.append(KeepTogether([*pending, *(fl._content if isinstance(fl, KeepTogether) else [fl])]))
            pending.clear()
        else:
            story.append(fl)

    def bullets(items):
        return ListFlowable([ListItem(P(t), leftIndent=12, value="•") for t in items], bulletType="bullet", start="•", leftIndent=12,
                            bulletFontName="Sans", bulletFontSize=7.5, bulletColor=ink2, spaceAfter=4)

    def table(rows, widths, right=(), bold=()):
        data = [[Paragraph(str(x), ParagraphStyle("a", parent=cellb if i == 0 or i in bold else cell, alignment=2 if j in right else 0))
                 for j, x in enumerate(r)] for i, r in enumerate(rows)]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.4),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4), ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 3.5), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ink),
                               ("LINEBELOW", (0, -1), (-1, -1), 0.6, rule)]
                              + [("BACKGROUND", (0, i), (-1, i), shade) for i in range(2, len(rows), 2)]))
        return t

    def captioned(t, text):
        return KeepTogether([t, P(text, caption)])

    def figure(path, text, width=W):
        with PILImage.open(path) as im:
            w, h = im.size
        return KeepTogether([Image(str(path), width=width, height=width * h / w), P(text, caption)])

    def boxed(flowables):
        t = Table([[flowables]], colWidths=[W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), shade), ("LINEBEFORE", (0, 0), (0, -1), 2.2, blue),
                               ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                               ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        return t

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Sans", 7.2)
        canvas.setFillColor(ink2)
        canvas.drawString(2.2 * cm, 1.2 * cm, ("PRELIMINARY · " if not final else "") + "Point-only Sonata student vs concatenation, ScanNet++ v2")
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"{doc.page}")
        canvas.restoreState()

    rows, diffs = m.get("rows"), m.get("diffs")
    obs, diag, log = m.get("observations") or {}, m.get("teacher_diagnostic"), m.get("student") or {}
    rep, free = m.get("reproducibility") or {}, m.get("image_free") or {}
    fig_pipeline(figs / "pipeline.png")

    # ---------------------------------------------------------------- title and summary
    add(P(esc(m["title"]), ParagraphStyle("title", fontName="Sans-Bold", fontSize=18.5, leading=23, textColor=ink, spaceAfter=5)))
    sub = "Experiment 2: a cross-attention teacher and a point-only Sonata student on ScanNet++ v2"
    add(P(sub, ParagraphStyle("sub", fontName="Serif-Italic", fontSize=11, leading=15, textColor=ink2, spaceAfter=3)))
    meta = f"Project appearance-pt · Experiment2_CrossAttention · generated {esc(m.get('generated_utc', ''))} · mode {m.get('mode')}"
    add(P(meta, ParagraphStyle("meta", fontName="Sans", fontSize=7.8, leading=10.5, textColor=ink2, spaceAfter=8)))
    if not final:
        add(boxed([P("<b>PRELIMINARY.</b> " + (
            f"These numbers come from the smoke run ({m.get('train_scenes')} train scenes, {m.get('val_scenes')} val scenes, a "
            f"{log.get('steps', 'few')}-step student). They test the workflow and are not the experiment's answer. "
            if m.get("mode") == "smoke" else
            "The full evaluation did not complete or did not pass its checks; only measured numbers are shown. ")
            + "Rerun step 5 after step 4 of the full run to regenerate this report.",
            ParagraphStyle("pre", parent=boxtext, textColor=colors.HexColor("#8a3b12")))]))
        add(Spacer(1, 6))

    summary = [P("<b>Summary</b>", ParagraphStyle("bh", parent=boxtext, fontName="Sans-Bold", fontSize=9.6, spaceAfter=5)),
               P("<b>Question.</b> The pilot found that concatenating lifted DINOv2 features to frozen Sonata features raises "
                 "linear-probe mIoU. How much of that gain can Sonata keep when it is instead taught by multi-view images during "
                 "training (a cross-attention teacher and continued training of Sonata as a student) and then evaluated on the "
                 "point cloud alone?", boxtext)]
    if rows and diffs:
        t, s_, c = rows["T"], rows["S"], rows["S+"]
        d_ts, d_tc = diffs["T - S"], diffs["T - S+"]
        verb = "gained" if d_ts["diff"] >= 0 else "lost"
        where = "below" if d_tc["diff"] < 0 else "above"
        rec = m.get("recovered_fraction")
        resolved = "resolved" if (d_ts["lo"] > 0 or d_ts["hi"] < 0) else "not resolved from zero"
        summary.append(P(f"<b>Measured answer.</b> The point-only student reaches <b>{t['top1']:.2f}</b> top-1 mIoU on the "
                         f"{m.get('val_scenes')} val {plural(m.get('val_scenes'), 'scene')}: it {verb} <b>{abs(d_ts['diff']):.2f} points</b> relative to frozen "
                         f"Sonata ({s_['top1']:.2f}; 95 % interval {s2(d_ts['lo'])} to {s2(d_ts['hi'])}, {resolved}) and finished "
                         f"<b>{abs(d_tc['diff']):.2f} points {where}</b> Sonata + DINOv2 concatenation ({c['top1']:.2f}). "
                         + (f"That is {100 * rec:.0f} % of the concatenation gain of {m['concatenation_gain']:.2f} points." if rec is not None else ""),
                         boxtext))
    else:
        summary.append(P("<b>Measured answer.</b> Not measured yet: the evaluation step has not produced its rows.", boxtext))
    ctrl = m.get("control")
    if ctrl and m.get("mode") == "smoke":
        x = ctrl["diffs"]["T - Sr"]
        summary.append(P(f"<b>Like for like (smoke).</b> The pilot rows use the pilot's full probes; the student's probe here is the "
                         f"smoke run's small one. Against frozen Sonata rerun with the same small probe "
                         f"({ctrl['row']['top1']:.2f}), the student differs by {s2(x['diff'])} points ({s2(x['lo'])} to {s2(x['hi'])}).",
                         boxtext))
    summary.append(P("<b>Images at inference.</b> Only Sonata + DINOv2 concatenation needs images: 30 posed DSLR photos per "
                     "scene, run through DINOv2 and lifted onto the points. Frozen Sonata and the student use the point cloud "
                     "(xyz, colour, normal) alone; the student's teacher and head are discarded after training.", boxtext))
    summary.append(P("<b>Main limitation.</b> " + (
        "Smoke-run numbers on a handful of scenes; they say nothing about the answer. " if m.get("mode") == "smoke" else "")
        + "One student was trained, so the intervals cover val scenes and probe seeds but not how much another training run of "
        "the student would differ. The comparison evaluates the complete teacher-student method; it cannot isolate the "
        "contribution of learned attention, because no student of a uniform-average teacher was trained. With 50 val scenes, "
        "differences under about 1 mIoU cannot be resolved.", boxtext))
    add(boxed(summary))
    add(Spacer(1, 6))

    # ---------------------------------------------------------------- background
    add(P("1. Question and background", h1))
    if rows and diffs:
        g = diffs["S+ - S"]
        add(P(f"The pilot (Experiment 1) probed frozen point encoders with one linear layer on ScanNet++ v2. Frozen Sonata reached "
              f"{rows['S']['top1']:.2f} top-1 mIoU; concatenating DINOv2 features lifted from 30 posed photos per scene raised it "
              f"to {rows['S+']['top1']:.2f}, a gain of {g['diff']:.2f} points (95 % interval {g['lo']:.2f} to {g['hi']:.2f}) on "
              f"these {m.get('val_scenes')} val {plural(m.get('val_scenes'), 'scene')}. That gain needs the photos at test time, and it disappears on points no "
              "camera sees."))
    add(P("This experiment asks how much of the gain survives when the images are used only during training. A cross-attention "
          "teacher learns, from the point's own frozen Sonata feature, how to weight the DINOv2 features of two views so that "
          "they predict a third, held-out view. Sonata is then trained further as a student to reproduce the teacher's "
          "prediction from the point cloud alone, and it is probed exactly like the pilot's frozen Sonata. The two pilot rows "
          "are reused from the pilot's saved per-scene results; only the student is new."))

    # ---------------------------------------------------------------- method
    add(P("2. Method", h1))
    add(figure(figs / "pipeline.png", "<b>Figure 1.</b> Teacher training (A) and student training (B). Blue boxes receive "
               "gradients; orange boxes are targets; grey boxes are frozen or fixed. The teacher sees two input views and "
               "is scored on a third, held-out view of the same point."))
    add(figure(figs / "pipeline_eval.png", "<b>Figure 2.</b> Point-only evaluation (C): only the linear probe is trained, "
               "with the pilot's recipe; the student is frozen."))
    teacher_cfg = (m.get("teacher_summary") or {})
    rec = (m.get("probes") or {}).get("recipe") or {}
    add(bullets([
        "<b>Start from the pilot.</b> The same Sonata checkpoint (sonata.pth, facebook/sonata @ df99897), DINOv2 ViT-L/14 with "
        "registers (e4c89a4), ScanNet++ v2 scenes (configs/scannetpp_subset.json: 200 train, 50 val), 30 DSLR views per scene, "
        "occlusion test v2, point inputs (xyz, colour, normal) and the official evaluator. Start commit: "
        f"{esc(rep.get('start_commit', ''))[:12]}.",
        "<b>Per-view observations (step 1).</b> The pilot cached only the average over cameras, and that cache no longer "
        "exists, so the pilot's lifting was rerun and stopped just before averaging: for every camera that sees a point, the "
        "camera, the pixel and the bilinearly sampled DINOv2 patch feature. Checks: the mean of a point's observations "
        "equals the pilot's lifted feature; every observation re-projects to its pixel; a point's cameras are distinct.",
        "<b>Teacher (step 2).</b> For a point seen by ≥ 3 cameras, two input cameras and a held-out third are drawn. Query: "
        "W_q applied to the point's frozen original Sonata feature; keys: W_k applied to the two input DINOv2 features; values: "
        "the two DINOv2 vectors unchanged, so the prediction is their weighted average. Loss: 1 − cos(prediction, held-out "
        "DINOv2 feature). Only W_q and W_k are trained; train scenes only, no labels"
        + (f"; {teacher_cfg.get('epochs')} epochs on {teacher_cfg.get('train_points', 0):,} points." if teacher_cfg else "."),
        "<b>Student (step 3).</b> Continued training from Sonata: the same checkpoint, the pilot's point inputs only. Loss: "
        f"{esc(log.get('loss', 'n/a'))}, where f(p) is the student's per-point upcast feature, h the prediction head "
        "(LayerNorm, 1232 → 1024, GELU, 1024 → 1024) and t(p) the frozen teacher's weighted DINOv2 vector over all of the "
        "point's cameras, detached. "
        + (f"Trainable: {esc(log.get('trainable'))} ({log.get('trainable_parameters', 0):,} parameters), {log.get('steps')} "
           f"steps on crops of {log.get('crop_points', 0):,} vertices." if log else ""),
        f"<b>Evaluation (step 4).</b> Teacher and head discarded; the student frozen; the pilot's feature layer (all five "
        f"stages upcast, 1232) at the pilot's labelled sample ({rec.get('points_per_scene', 0):,} points per train scene, "
        f"{m.get('train_scenes')} train scenes); a fresh linear probe with the pilot's recipe (learning rate from "
        f"{esc(rec.get('lrs'))} on {rec.get('holdout')} held-out train {plural(rec.get('holdout'), 'scene')}, {rec.get('sweep_epochs')} epochs; then seeds "
        f"0–{rec.get('seeds', 1) - 1}, {rec.get('epochs')} epochs, batch {rec.get('batch')}; chosen rate "
        f"{(m.get('probes') or {}).get('lr')}); every vertex of the val scenes, including points seen by fewer than three "
        "cameras; the toolbox's ground truth and metric.",
    ]))

    # ---------------------------------------------------------------- results
    add(P("3. Results", h1))
    if rows and diffs:
        add(P("3.1 Main result", h2))
        tbl = [["Method", "Inference inputs", "Val top-1 mIoU", "Top-3 mIoU", "Difference from frozen Sonata [95 %]", "Provenance"]]
        for c in ("S", "S+", "T"):
            r = rows[c]
            dd = "—" if c == "S" else f"{s2(diffs[f'{c} - S']['diff'])} [{s2(diffs[f'{c} - S']['lo'])}, {s2(diffs[f'{c} - S']['hi'])}]"
            prov = "pilot (reused): pilot_03_probes + pilot_04_eval" if c != "T" else "new run (this experiment)"
            tbl.append([esc(r["name"]), esc(r["inference_inputs"]), f"{r['top1']:.2f} ± {r['top1_std']:.2f}",
                        f"{r['top3']:.2f} ± {r['top3_std']:.2f}", dd, prov])
        add(captioned(table(tbl, [3.4 * cm, 3.2 * cm, 2.0 * cm, 1.8 * cm, 3.2 * cm, 3.3 * cm], right=(2, 3, 4), bold=(3,)),
                      f"<b>Table 1.</b> Official ScanNet++ val results ({m.get('val_scenes')} {plural(m.get('val_scenes'), 'scene')}; %, mean ± standard deviation "
                      f"over {rows['T']['seeds']} probe seeds; pilot rows {rows['S']['seeds']}). The two pilot rows are recomputed, unrounded, from the pilot's saved per-scene "
                      "confusion matrices; nothing of the pilot was rerun. Intervals: paired bootstrap over val scenes."))
        fig_results(m, figs / "results.png")
        add(figure(figs / "results.png", "<b>Figure 3.</b> Left: top-1 mIoU per method (bars: probe-seed standard deviation). "
                   "Right: the differences with their 95 % intervals."))
        dtab = [["Comparison", "Difference (points)", "95 % interval", "Share of resamples ≤ 0"]]
        for k, label in (("T - S", "Student − frozen Sonata"), ("T - S+", "Student − concatenation"),
                         ("S+ - S", "Concatenation − frozen Sonata (pilot's gain)")):
            x = diffs[k]
            dtab.append([label, s2(x["diff"]), f"[{s2(x['lo'])}, {s2(x['hi'])}]", f"{100 * x['p_le_0']:.1f} %"])
        add(captioned(table(dtab, [6.4 * cm, 3.2 * cm, 3.6 * cm, 3.6 * cm], right=(1, 2, 3)),
                      "<b>Table 2.</b> Differences in top-1 mIoU. The observed differences use unrounded means; the interval "
                      "bounds come from 1000 paired scene resamples."))
    else:
        add(P("Not measured yet: step 4 has not written its rows.", body))

    ctrl = m.get("control")
    if ctrl and rows:
        add(P("3.2 Protocol check: frozen Sonata rerun", h2))
        r, d_sr, d_ts = ctrl["row"], ctrl["diffs"]["Sr - S"], ctrl["diffs"]["T - Sr"]
        add(P("The original frozen Sonata went through this experiment's own evaluation: its features at the same labelled "
              "points, a probe with the same recipe and the same initial weights as the pilot's, and the same val vertices. "
              + ("In the full run it should reproduce the pilot's frozen-Sonata row within seed noise, which shows the student "
                 "is measured on the pilot's terms." if m.get("mode") == "full" else
                 "In this smoke run its probe is the small smoke probe, so it is the like-for-like reference for the student; "
                 "the pilot rows above use the pilot's full probes.")))
        ptab = [["Row", "Val top-1 mIoU", "Difference", "95 % interval"],
                ["Frozen Sonata, pilot (reused)", f"{rows['S']['top1']:.2f} ± {rows['S']['top1_std']:.2f}", "—", "—"],
                ["Frozen Sonata, rerun here", f"{r['top1']:.2f} ± {r['top1_std']:.2f}", s2(d_sr["diff"]) + " vs pilot",
                 f"[{s2(d_sr['lo'])}, {s2(d_sr['hi'])}]"],
                ["Point-only student", f"{rows['T']['top1']:.2f} ± {rows['T']['top1_std']:.2f}", s2(d_ts["diff"]) + " vs rerun",
                 f"[{s2(d_ts['lo'])}, {s2(d_ts['hi'])}]"]]
        add(captioned(table(ptab, [6.0 * cm, 3.6 * cm, 3.8 * cm, 3.5 * cm], right=(1, 2, 3)),
                      "<b>Table 2b.</b> The frozen-Sonata rerun against the pilot's recorded row, and the student against the "
                      "rerun (paired bootstrap over val scenes). Same probe recipe within the rerun and the student."))

    add(P("3.3 Teacher diagnostic", h2))
    if diag:
        d = diag["difference"]
        add(P(f"On {diag['dev_points']:,} points of {len(diag['dev_scenes'])} dev {plural(len(diag['dev_scenes']), 'scene')}, held out from teacher training, each "
              f"with a fixed triple of cameras, the teacher's prediction reaches a mean cosine of <b>{diag['teacher_cos']:.4f}</b> "
              f"to the held-out camera's DINOv2 feature; the uniform average of the same two input cameras reaches "
              f"<b>{diag['uniform_cos']:.4f}</b>. Difference {d['mean']:+.4f} (95 % interval {d['lo']:+.4f} to {d['hi']:+.4f}, "
              f"bootstrap over dev scenes); the teacher is closer on {100 * diag['teacher_better_fraction']:.1f} % of triples, "
              f"and its weights move on average {diag['mean_abs_weight_shift']:.3f} away from 0.5. This measures image-feature "
              "prediction only; whether it carried over into better point features is the segmentation result above."))
        fig_teacher(diag, figs / "teacher.png")
        add(figure(figs / "teacher.png", "<b>Figure 4.</b> Teacher versus uniform averaging on identical held-out view triples."))
    else:
        add(P("Not measured yet.", body))

    add(P("3.4 Student training", h2))
    if log.get("curve"):
        add(P(f"Continued training from Sonata for {log['steps']} steps ({esc(log['trainable'])} trainable, crops of "
              f"{log['crop_points']:,} vertices): loss {log['loss_first_50']:.4f} over the first 50 steps, "
              f"{log['loss_last_50']:.4f} over the last 50; median {log['seconds_per_step']:.2f} s per step; peak GPU "
              f"{log['peak_gpu_gb']} GB."))
        fig_student(log, figs / "student.png")
        add(figure(figs / "student.png", "<b>Figure 5.</b> The student's loss, " + esc(log.get("loss", ""))))
    else:
        add(P("Not measured yet.", body))
    smoke = m.get("smoke")
    if smoke and smoke.get("table"):
        stab = [["Trainable", "Crop (vertices)", "Status", "s / step", "Peak GPU (GB)", "Encoder changed"]]
        for r in smoke["table"]:
            stab.append([r["trainable"], f"{r['crop_points']:,}", r["status"], f2(r.get("seconds_per_step")),
                         f2(r.get("peak_gpu_gb")), str(r.get("first_update", {}).get("trainable_changed", "—"))])
        add(captioned(table(stab, [2.4 * cm, 2.8 * cm, 2.8 * cm, 2.2 * cm, 2.8 * cm, 3.4 * cm], right=(1, 3, 4)),
                      "<b>Table 3.</b> The smoke run's measurements behind the trainable-layer and step-budget choice "
                      f"(recommended: {esc(json.dumps(smoke.get('recommendation')))})."))

    # ---------------------------------------------------------------- interpretation
    add(P("4. Interpretation", h1))
    if rows and diffs:
        d_ts, d_tc, gain = diffs["T - S"], diffs["T - S+"], m["concatenation_gain"]
        lines = []
        if d_ts["hi"] < 0:
            lines.append(f"<b>The student lost to frozen Sonata</b> ({s2(d_ts['diff'])} points, interval {s2(d_ts['lo'])} to "
                         f"{s2(d_ts['hi'])}, entirely below zero): continued training on this objective made the features worse "
                         "for the linear probe.")
        elif d_ts["lo"] <= 0:
            lines.append(f"<b>No resolved change from frozen Sonata</b> ({s2(d_ts['diff'])} points, interval {s2(d_ts['lo'])} to "
                         f"{s2(d_ts['hi'])}): the student did not measurably recover any of the {gain:.2f}-point concatenation gain.")
        elif d_tc["lo"] > 0:
            lines.append(f"<b>The student exceeded the concatenation</b>: {s2(d_ts['diff'])} points over frozen Sonata, "
                         f"{s2(d_tc['diff'])} over Sonata + DINOv2 (interval above zero), with no image at inference.")
        else:
            lines.append(f"<b>The student recovered part of the concatenation gain:</b> {s2(d_ts['diff'])} points over frozen "
                         f"Sonata, about {100 * (m['recovered_fraction'] or 0):.0f} % of the pilot's {gain:.2f}-point gain, "
                         "with no image at inference.")
        if d_tc["hi"] < 0:
            lines.append(f"It finished {abs(d_tc['diff']):.2f} points below Sonata + DINOv2 concatenation (interval "
                         f"{s2(d_tc['lo'])} to {s2(d_tc['hi'])}), which still uses the photos at inference.")
        elif d_tc["lo"] <= 0:
            lines.append(f"Its difference to the concatenation ({s2(d_tc['diff'])} points, interval {s2(d_tc['lo'])} to "
                         f"{s2(d_tc['hi'])}) is not resolved.")
        lines.append("The result belongs to the whole method (observations, teacher, targets, continued training); it cannot "
                     "say how much learned attention contributed, since no student of a uniform-average teacher was trained.")
        if diag:
            dd = diag["difference"]
            better = dd["lo"] > 0
            lines.append(f"The teacher's attention changed its targets little: on held-out view triples its cosine to the third view "
                         f"was {diag['teacher_cos']:.4f} against {diag['uniform_cos']:.4f} for the plain average of the two input "
                         f"views ({dd['mean']:+.4f}, interval {dd['lo']:+.4f} to {dd['hi']:+.4f}"
                         + (", resolved but small" if better else ", not resolved") + f"), and its weights moved on average "
                         f"{diag['mean_abs_weight_shift']:.2f} away from 0.5. The student's targets are therefore close to a "
                         "multi-view average of DINOv2 features; whether the attention matters for the student cannot be told "
                         "from this run.")
        lines.append("Only one student was trained (seed 0). The intervals cover val scenes and probe seeds, not the "
                     "variation between student training runs; a second student would show how stable the gain is.")
        lr = (m.get("probes") or {}).get("lr")
        grid = ((m.get("probes") or {}).get("recipe") or {}).get("lrs") or []
        if grid and lr == min(grid):
            lines.append(f"The probe's learning rate ({lr:g}) was the lowest in the grid, as in the pilot, for the student "
                         + ("and for the rerun frozen Sonata alike; " if (m.get("probes") or {}).get("control_lr") == lr else "; ")
                         + "a lower rate might raise absolute scores slightly, but the comparison uses the same recipe.")
        add(bullets(lines))
    else:
        add(P("Pending the evaluation.", body))

    # ---------------------------------------------------------------- checks and reproducibility
    add(P("5. Checks", h1))
    checks = obs.get("checks") or {}
    tchk = m.get("teacher_checks") or {}
    schk = (m.get("student_checks") or {}).get("first_update") or {}
    ctab = [["Check", "Result"],
            ["Observations: mean equals the pilot's lifted feature", f"max |diff| {checks.get('mean_vs_lift_max_abs')}"],
            ["Observations: pixels re-project; cameras distinct; ids unique",
             f"{'n/a' if checks.get('reprojection_max_px') is None else format(checks['reprojection_max_px'], '.1e')} px; "
             f"{checks.get('cameras_distinct')}; {checks.get('vertex_ids_unique')}"],
            ["Training points seen by ≥ 3 distinct cameras", f"{obs.get('fraction_training_points_ge3', 'n/a')} "
             f"({obs.get('training_points_ge3', 0):,} of {obs.get('training_points', 0):,})"],
            ["Teacher: camera separation", str((tchk.get("dev_triples") or {}).get("held_out_not_an_input"))],
            ["Teacher: equal weights reproduce the mean", f"max |diff| {(tchk.get('equal_weights') or {}).get('max_abs_diff_to_mean')}"],
            ["Teacher: one update", f"W_q changed after one update: {((tchk.get('update') or {}).get('update1') or {}).get('changed', {}).get('q.weight')}; "
                                    f"W_k after a second: {((tchk.get('update') or {}).get('update2') or {}).get('changed', {}).get('k.weight')} "
                                    "(W_k gets no gradient while W_q is zero)"],
            ["Student: one update reaches the encoder", f"{schk.get('trainable_changed')} (encoder gradient norm "
                                                        f"{f2(schk.get('encoder_grad_norm'))}; frozen stages unchanged: {schk.get('frozen_unchanged', 'all trainable')})"],
            ["Frozen Sonata rerun reproduces the pilot's row", "not run" if not (m.get("control") or {}).get("diffs") else
             f"{s2(m['control']['diffs']['Sr - S']['diff'])} points vs the pilot ({s2(m['control']['diffs']['Sr - S']['lo'])} to "
             f"{s2(m['control']['diffs']['Sr - S']['hi'])})" + (" (smoke probe: not expected to match)" if m.get("mode") == "smoke" else "")],
            ["Point-only evaluation without images, cameras, DINO, teacher", f"{free.get('ok')}: stripped copy identical "
             f"(max |diff| {free.get('max_abs_feature_diff')}), image modules imported {free.get('image_modules_imported_by_this_step')}"]]
    add(captioned(table(ctab, [8.6 * cm, 8.3 * cm]), "<b>Table 4.</b> The checks each step ran and recorded."))

    add(P("6. Reproducibility", h1))
    st = m.get("storage_gb") or {}
    rtab = [["Item", "Value"],
            ["Repository commit", f"{esc(rep.get('commit'))} (dirty: {rep.get('dirty')}); start commit {esc(rep.get('start_commit'))}"],
            ["Checkpoints", f"{esc(rep.get('checkpoint'))}; {esc(rep.get('dino'))}"],
            ["Scene split", f"{esc(rep.get('subset'))}; teacher dev scenes: {esc(', '.join(rep.get('dev_scenes') or []))}"],
            ["Seeds", esc(json.dumps(rep.get("seeds")))],
            ["Student", f"trainable {esc(rep.get('trainable'))}, {rep.get('steps')} steps; choice recorded from the smoke run: "
                        f"{esc(json.dumps(rep.get('student_choice', {}).get('recorded_from_smoke')))}"],
            ["GPU time", f"{rep.get('gpu_hours')} GPU-hours on one T4 (steps 1–4 of this run)"],
            ["Storage", f"observations {st.get('observations')} GB, student {st.get('student')} GB, probe features {st.get('probe_features')} GB"],
            ["Image-free evaluation", f"{free.get('ok')}; archive reads {esc(', '.join(free.get('archive_reads') or []))}"]]
    add(captioned(table(rtab, [3.6 * cm, 13.3 * cm]), "<b>Table 5.</b> What is needed to reproduce this report; every number "
                  "comes from reports/metrics.json."))

    pdf = out_dir / PDF_NAME
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=1.9 * cm, bottomMargin=2.0 * cm,
                            title=m["title"], author="appearance-pt", subject="Experiment 2: cross-attention teacher, point-only student")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return pdf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--mode", choices=("smoke", "full"), help="accepted for the notebook's uniform call; the status comes from metrics.json")
    args = ap.parse_args()
    reports = args.out / "reports"
    metrics_path = reports / "metrics.json"
    if not metrics_path.exists():
        print(f"!! {metrics_path} missing: run step 4 first")
        return 2
    m = json.loads(metrics_path.read_text())
    pdf = build(m, reports)
    print(f"report ({m['status']}): {pdf} ({os.path.getsize(pdf) / 1e6:.2f} MB); figures in {reports / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
