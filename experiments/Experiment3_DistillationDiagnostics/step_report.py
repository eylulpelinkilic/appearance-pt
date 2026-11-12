"""Experiment 3: reports/metrics.json (every reported number: the summaries of Runs A, B and C) and
reports/distillation_diagnostics_report.pdf, in Experiment 2's report style.

Each run's summary is taken from this output folder or, failing that, from an attached Experiment 3 output of the same
mode (notebook 2 attaches notebook 1's output, so its report covers all three runs). The report is "final" only for a
full run with all three runs present and their checks passed; otherwise every page says PRELIMINARY and names what is
missing. The pre-registered decision rules are printed from config.json.

Usage: python experiments/Experiment3_DistillationDiagnostics/step_report.py --mode smoke|full [--out DIR]
Needs matplotlib and reportlab.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from step5_report import AQUA, BLUE, FONTS, GRID, INK, MUTED, ORANGE, esc, f2, s2  # noqa: E402  (Experiment 2's style)
from xattn import run_mode  # noqa: E402
from xdiag import CONFIG, OUT, find_runs, load_config  # noqa: E402

PDF_NAME = "distillation_diagnostics_report.pdf"
ROW_ORDER = ("S", "S+", "A1", "A2", "A3", "C1", "C2", "C3")


def collect_runs(out: Path, inputs: Path, mode: str) -> dict:
    found = {}
    sources = [out] + [d for d in find_runs(inputs, "Experiment3_DistillationDiagnostics") if run_mode(d) == mode]
    for run in ("A", "B", "C"):
        for root in sources:
            path = root / f"run{run}" / "summary.json"
            if path.exists():
                found[run] = dict(summary=json.loads(path.read_text()), source=str(path),
                                  origin="this output" if root == out else f"attached input {root.parent.name}")
                break
    return found


def run_ok(run: str, s: dict) -> bool:
    c = s["checks"]
    if run == "A":
        return bool(c["reproduction"]["ok"] and c["widths"]["ok"] and c["image_free"]["ok"])
    if run == "B":
        o = c["observations"]
        return bool(c["no_probe"] and o["counts_equal"] and o["cameras_distinct"])
    return bool(c["image_free"]["ok"] and c["training"].get("only_target_differs") and c["training"].get("target_vs_lift_ok"))


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def ci(d: dict, digits: int = 2) -> str:
    f = (lambda x: s2(x)) if digits == 2 else (lambda x: "n/a" if x is None else f"{x:+.{digits}f}".replace("-", "−"))
    return f"{f(d['diff'] if 'diff' in d else d['gap'])} [{f(d['lo'])}, {f(d['hi'])}]"


def s4(x) -> str:
    return f"{x:+.4f}".replace("-", "−")


def gb(x) -> str:
    return "n/a (CPU)" if x is None else f"{x} GB"


def decision_text(run: str, s: dict) -> str:
    """The decision with the numbers it rests on."""
    if run == "A":
        parts = [f"{k.replace(' - ', ' − ')}: {ci(d['top1'])}" for k, d in s["diffs"].items()]
        return f"{cap(s['decision']['text'])} (top-1, {'; '.join(parts)} mIoU)."
    if run == "B":
        return (f"{cap(s['decision']['text'])} (gap student {ci(s['b1'], 4)}, B2 {ci(s['b2'], 4)}, difference "
                f"{ci(s['gap_difference'], 4)}).")
    d = s["diffs"]["C1 - A1"]["top1"]
    if s["decision"]["no_evidence"]:
        return f"No evidence that learned attention matters (uniform − attention student, f(p): {ci(d)} mIoU)."
    who = "uniform" if d["diff"] > 0 else "attention"
    return f"The {who}-teacher student is ahead (uniform − attention student, f(p): {ci(d)} mIoU)."


def fig_rows(rows: dict, path: Path) -> None:
    keys = [k for k in ROW_ORDER if k in rows and k != "S"]
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(keys) + 0.7))
    for j, k in enumerate(keys):
        d = rows[k]["vs_frozen_sonata_top1"]
        col = ORANGE if k == "S+" else (BLUE if k.startswith("A") else AQUA)
        ax.plot([d["lo"], d["hi"]], [j, j], color=col, lw=1.8, solid_capstyle="round")
        ax.scatter([d["diff"]], [j], s=22, color=col, edgecolor="white", lw=0.7, zorder=3)
        ax.text(d["hi"], j - 0.2, "  " + s2(d["diff"]), va="bottom", fontsize=6.6, color=INK)
    ax.axvline(0, color="#8c8a84", lw=0.9)
    ax.set_yticks(range(len(keys)), [rows[k]["name"] for k in keys])
    ax.tick_params(axis="y", labelsize=6.8, colors=INK)
    ax.set_ylim(len(keys) - 0.4, -0.6)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlabel("top-1 mIoU difference from frozen Sonata (points), 95 % interval", fontsize=7)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def fig_cosines(b: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 1.6))
    labels = [("Student h(f(p)) (B1)", b["b1"], BLUE), ("Head on frozen Sonata (B2)", b["b2"], MUTED)]
    for j, (name, g, col) in enumerate(labels):
        ax.plot([g["val"], g["train"]], [j, j], color="#cfcdc6", lw=1.4)
        ax.scatter([g["train"]], [j], s=26, color=col, zorder=3)
        ax.scatter([g["val"]], [j], s=24, facecolor="white", edgecolor=col, lw=1.2, zorder=3)
        ax.text(max(g["train"], g["val"]) + 0.004, j, f"gap {s2(g['gap'])}", va="center", fontsize=6.6, color=INK)
    ax.set_yticks(range(len(labels)), [x[0] for x in labels])
    ax.tick_params(axis="y", labelsize=6.8, colors=INK)
    ax.set_ylim(len(labels) - 0.4, -0.6)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlabel("cosine to the teacher's target (filled: train scenes, open: val scenes)", fontsize=7)
    xs = [g[k] for _, g, _ in labels for k in ("train", "val")]
    ax.set_xlim(min(xs) - 0.02, max(xs) + 0.05)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def build(m: dict, cfg: dict, out_dir: Path) -> Path:
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
    caption = ParagraphStyle("caption", fontName="Sans", fontSize=7.4, leading=10, textColor=ink2, spaceBefore=3, spaceAfter=9)
    cell = ParagraphStyle("cell", fontName="Sans", fontSize=7.0, leading=9.0, textColor=ink)
    cellb = ParagraphStyle("cellb", parent=cell, fontName="Sans-Bold")
    boxtext = ParagraphStyle("boxtext", parent=body, fontSize=8.8, leading=12.8, spaceAfter=3)
    story, pending = [], []

    def P(text, style=body):
        return Paragraph(text, style)

    def add(fl):
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
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.2),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2), ("LEFTPADDING", (0, 0), (-1, -1), 3),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ink),
                               ("LINEBELOW", (0, -1), (-1, -1), 0.6, rule)]
                              + [("BACKGROUND", (0, i), (-1, i), shade) for i in range(2, len(rows), 2)]))
        return t

    def captioned(t, text):
        return KeepTogether([t, P(text, caption)])

    def figure(path, text):
        with PILImage.open(path) as im:
            w, h = im.size
        return KeepTogether([Image(str(path), width=W, height=W * h / w), P(text, caption)])

    def boxed(flowables, colour=blue):
        t = Table([[flowables]], colWidths=[W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), shade), ("LINEBEFORE", (0, 0), (0, -1), 2.2, colour),
                               ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                               ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        return t

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Sans", 7.2)
        canvas.setFillColor(ink2)
        canvas.drawString(2.2 * cm, 1.2 * cm, ("PRELIMINARY · " if not final else "") + "Experiment 3: distillation diagnostics, ScanNet++ v2")
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"{doc.page}")
        canvas.restoreState()

    runs = m["runs"]
    A, B, C = (runs.get(k, {}).get("summary") for k in ("A", "B", "C"))
    rows = {}
    for s in (A, C):
        if s:
            rows.update(s["rows"])
    rules = cfg["decision_rules"]
    bx = ParagraphStyle("bh", parent=boxtext, fontName="Sans-Bold", fontSize=9.6, spaceAfter=5)

    add(P(esc(m["title"]), ParagraphStyle("title", fontName="Sans-Bold", fontSize=17.5, leading=22, textColor=ink, spaceAfter=5)))
    add(P("Experiment 3: three diagnostics of the Experiment 2 student on ScanNet++ v2",
          ParagraphStyle("sub", fontName="Serif-Italic", fontSize=11, leading=15, textColor=ink2, spaceAfter=3)))
    add(P(f"Project appearance-pt · Experiment3_DistillationDiagnostics · generated {esc(m['generated_utc'])} · mode {m['mode']}",
          ParagraphStyle("meta", fontName="Sans", fontSize=7.8, leading=10.5, textColor=ink2, spaceAfter=8)))
    if not final:
        add(boxed([P("<b>PRELIMINARY.</b> " + esc(m["status_reason"]) + " Rerun the report step when every run of the full "
                     "experiment has finished.", ParagraphStyle("pre", parent=boxtext, textColor=colors.HexColor("#8a3b12")))],
                  colors.HexColor(ORANGE)))
        add(Spacer(1, 6))

    summ = [P("Summary", bx)]
    a1 = rows.get("A1")
    if a1:
        summ.append(P(f"<b>Starting point.</b> Experiment 2's point-only student: {a1['top1']:.2f} top-1 mIoU on the val scenes, "
                      f"{s2(a1['vs_frozen_sonata_top1']['diff'])} over frozen Sonata, {100 * a1['recovered_top1']:.0f} % of the "
                      f"concatenation gain (top-3: {100 * a1['recovered_top3']:.0f} %).", boxtext))
    for name, s, q in (("A", A, "Is the distilled information in the student but not linearly readable from f(p)?"),
                       ("B", B, "Does the student's match to its targets hold on val scenes?"),
                       ("C", C, "Does learned attention matter?")):
        ans = esc(decision_text(name, s)) if s else "Not run yet."
        summ.append(P(f"<b>{name}. {esc(q)}</b> {ans}", boxtext))
    summ.append(P("<b>Main limitation.</b> One training seed per student and 50 val scenes: differences under about 1 mIoU are "
                  "not resolvable, and training-seed variance is not measured.", boxtext))
    add(boxed(summ))

    add(P("1. Method and pre-registered decision rules", h1))
    add(bullets([
        "<b>Run A</b> probes the Experiment 2 student's head output: A1 f(p) (Experiment 2's evaluation, reused), A2 h(f(p)) and "
        "A3 [f(p), h(f(p))], with the pilot's recipe (learning rate per feature set by the recipe's own rule), the official "
        "metric on every val vertex, and a paired bootstrap over the 50 val scenes. h is the head Experiment 2 saved in its "
        "final training checkpoint.",
        "<b>Run B</b> measures, on whole scenes, the cosine of the student's h(f(p)) to the teacher's target on train and val "
        "scenes (B1), the same for a head trained on frozen Sonata (B2), and the cosine between the attention and the uniform "
        "targets (B3). It is the only place val images are used, in a separate code path that imports no probe code.",
        "<b>Run C</b> trains a second student identical to Experiment 2's except for the target, the uniform average of the "
        "point's DINOv2 observations, and evaluates it exactly like Experiment 2's step 4.",
    ]))
    add(P("Decision rules, fixed in config.json before the full run:", body))
    add(bullets([f"<b>{k}.</b> {esc(v)}" for k, v in rules.items()]))
    add(P("Check rules, fixed in config.json:", body))
    add(bullets([f"<b>{esc(k.replace('_', ' '))}.</b> {esc(v)}" for k, v in cfg.get("check_rules", {}).items()]))

    add(P("2. Segmentation", h1))
    if rows:
        head = ["Row", "Inputs", "Top-1", "Top-3", "vs frozen Sonata, top-1 [95 %]", "Recovered top-1 · top-3", "LR", "Source"]
        trows = [head]
        for k in [k for k in ROW_ORDER if k in rows]:
            r = rows[k]
            d = r.get("vs_frozen_sonata_top1")
            trows.append([esc(r["name"]), esc(r["inference_inputs"]), f"{r['top1']:.2f} ± {r['top1_std']:.2f}", f"{r['top3']:.2f}",
                          "—" if not d else f"{s2(d['diff'])} [{s2(d['lo'])}, {s2(d['hi'])}]",
                          f"{100 * r['recovered_top1']:.0f} · {100 * r['recovered_top3']:.0f} %", f"{r.get('lr') or '—'}",
                          esc(r["provenance"])])
        add(captioned(table(trows, [3.4 * cm, 2.1 * cm, 2.0 * cm, 1.1 * cm, 2.7 * cm, 1.8 * cm, 1.0 * cm, 2.5 * cm], right=(2, 3, 4, 5, 6)),
                      "<b>Table 1.</b> Official ScanNet++ val results (%, mean ± standard deviation over the probe seeds); recovered: "
                      "the share of the concatenation gain (Sonata + DINOv2 − Sonata) at top-1 and top-3; LR: the probe learning "
                      "rate chosen by the recipe (A1's from Experiment 2)."))
        fig_rows(rows, figs / "rows.png")
        add(figure(figs / "rows.png", "<b>Figure 1.</b> Every row's top-1 difference from frozen Sonata with its 95 % interval "
                   "(paired bootstrap over val scenes)."))
        dt = [["Comparison", "Top-1 [95 %]", "Share ≤ 0", "Top-3 [95 %]"]]
        for s in (A, C):
            for k, d in (s or {}).get("diffs", {}).items():
                dt.append([k.replace(" - ", " − "), f"{s2(d['top1']['diff'])} [{s2(d['top1']['lo'])}, {s2(d['top1']['hi'])}]", f"{100 * d['top1']['p_le_0']:.1f} %",
                           f"{s2(d['top3']['diff'])} [{s2(d['top3']['lo'])}, {s2(d['top3']['hi'])}]"])
        add(captioned(table(dt, [4.2 * cm, 4.4 * cm, 2.4 * cm, 4.4 * cm], right=(1, 2, 3)),
                      "<b>Table 2.</b> The comparisons the decision rules use (A2/A3 against A1; the uniform student against the "
                      "attention student)."))
    else:
        add(P("Not measured yet.", body))
    if A:
        add(P(f"<b>Decision A.</b> {esc(decision_text('A', A))}", body))
    if C:
        add(P(f"<b>Decision C.</b> {esc(decision_text('C', C))} <b>Proposed next (not run):</b> "
              f"{esc(C['second_seed_proposal']['text'])} About {C['second_seed_proposal']['estimate_hours']} GPU-hours "
              f"({C['second_seed_proposal']['per_student_hours']} h per student with its evaluation).", body))

    add(P("3. Run B: cosine to the targets on train and val scenes", h1))
    if B:
        g1, g2, gd = B["b1"], B["b2"], B["gap_difference"]
        bt = [["Measure", "Train scenes", "Val scenes", "Gap (train − val) [95 %]"],
              ["B1 student h(f(p)) vs t_attn", f"{g1['train']:.4f}", f"{g1['val']:.4f}", f"{s2(g1['gap'])} [{s2(g1['lo'])}, {s2(g1['hi'])}]"],
              ["B2 head on frozen Sonata vs t_attn", f"{g2['train']:.4f}", f"{g2['val']:.4f}", f"{s2(g2['gap'])} [{s2(g2['lo'])}, {s2(g2['hi'])}]"],
              ["Gap difference (B1 − B2)", "", "", f"{s2(gd['diff'])} [{s2(gd['lo'])}, {s2(gd['hi'])}]"],
              ["B3 cos(t_attn, t_mean)", f"{B['b3_train']:.4f}", f"{B['b3_val']['mean']:.4f}", "—"]]
        add(captioned(table(bt, [5.6 * cm, 2.6 * cm, 2.6 * cm, 5.0 * cm], right=(1, 2, 3)),
                      f"<b>Table 3.</b> Mean cosines over {B['sampling']['points_per_scene']:,} points per scene seen by ≥ 1 camera "
                      f"({len(B['train_scenes'])} train, {len(B['val_scenes'])} val scenes; whole scenes); intervals: bootstrap over "
                      "scenes. The B2 head trained "
                      f"{B['b2_head']['epochs']} epochs ({B['b2_head']['steps']:,} steps) to a train loss of {B['b2_head']['final_loss']}."))
        fig_cosines(B, figs / "cosines.png")
        add(figure(figs / "cosines.png", "<b>Figure 2.</b> Cosine to the targets on train (filled) and val (open) scenes."))
        cr = B["crops"]
        add(P(f"Training-style crops on the train scenes ({cr['points']:,} points covered): B1 {f2(cr['b1_crop'])} in crops "
              f"(points with ≥ 3 cameras: {f2(cr['b1_crop_ge3'])}) against {f2(cr['b1_whole'])} on whole scenes, for the same points. "
              "The student's training loss ended at 1 − cos ≈ 0.134 on its own targets (≥ 3 cameras, crops).", body))
        ct = [["Cameras", "Val points", "B1", "B2", "B3"]] + [[k, f"{v['points']:,}", f2(v["b1"]), f2(v["b2"]), f2(v["b3"])]
                                                              for k, v in B["by_cameras"].items()]
        add(captioned(table(ct, [3.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm], right=(1, 2, 3, 4)),
                      "<b>Table 4.</b> Val cosines by the number of cameras that see a point (the student was trained on points "
                      "with ≥ 3 cameras only; with one camera t_attn equals t_mean)."))
        cl = [["Class", "Val vertices %", "Points", "B1", "B2", "IoU change student − Sonata"]]
        for c in B["classes"]:
            cl.append([esc(c["cls"]), f"{100 * c['vertex_share']:.2f}", f"{c['points']:,}", f2(c["b1"]), f2(c["b2"]),
                       s2(c["iou_change_student_minus_sonata"])])
        add(captioned(table(cl, [4.0 * cm, 2.4 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 4.0 * cm], right=(1, 2, 3, 4, 5)),
                      "<b>Table 5.</b> The most frequent val classes: val cosines and the per-class top-1 IoU change of Experiment "
                      "2's student over frozen Sonata."))
        add(P(f"<b>Decision B.</b> {esc(decision_text('B', B))}", body))
    else:
        add(P("Not run yet.", body))

    if C:
        add(P("4. Run C: the uniform-teacher student", h1))
        tl, ck = C["training"], C["checks"]["training"]
        av = tl["attn_vs_mean_train"]
        lifts = "; ".join(f"scene {x['scene']}: {x['points']:,} points, max |difference| {x['max_abs_diff']:.1e} (relative "
                          f"{x['max_rel_diff']:.1e}), min cosine {x['min_cos']:.6f}" for x in ck["target_vs_lift"])
        add(bullets([
            f"The two students' configurations differ only in: {esc(', '.join(ck['config_diff']))}.",
            f"The target t_mean against the pilot's lift(), recomputed from the images: {esc(lifts)}.",
            f"cos(t_attn, t_mean) over {av['points']:,} training targets: mean {av['mean']:.4f}, 5th percentile {av['p05']:.4f}; by "
            f"camera count " + ", ".join(f"{k}: {v:.4f}" for k, v in av["by_cameras"].items()) + ".",
            f"Training: {tl['steps']:,} steps, {esc(tl['trainable'])} trainable ({tl['trainable_parameters']:,} parameters), loss "
            f"{tl['loss_first_50']:.4f} → {tl['loss_last_50']:.4f}, {tl['session_hours']:.2f} h, peak GPU {gb(tl['peak_gpu_gb'])}.",
        ]))

    add(P("5. Checks", h1))
    ch = [["Check", "Result"]]
    if A:
        a = A["checks"]
        rp_, w = a["reproduction"], a["widths"]
        ch += [["A: h is Experiment 2's saved head, at the end of training",
                f"SHA-256 {a['head']['head_sha256'][:16]}…, step {a['head']['step']:,} of {a['head']['steps']:,}"],
               ["A: f(p) recomputed now, through Experiment 2's saved probes on every val scene (A1′), reproduces A1",
                f"{'passed' if rp_['ok'] else 'FAILED'}: top-1 {s4(rp_['diff_top1'])} (tolerance {rp_['tolerance_top1']:.4f}), "
                f"top-3 {s4(rp_['diff_top3'])} (tolerance {rp_['tolerance_top3']:.4f}); {100 * rp_['moved_share_top1']:.3f} % of "
                f"top-1 counts moved (max {rp_['max_abs_count_diff']} in a cell), {rp_['identical_scenes']} of {rp_['scenes']} "
                f"scenes identical; tile counts {'as in Experiment 2' if rp_['tiles_match'] else 'DIFFER: ' + esc(json.dumps(rp_['tiles_mismatch']))}"],
               ["A: feature widths f, h, [f, h]", f"{w['f']}, {w['h']}, {w['f_plus_h']}: {'ok' if w['ok'] else 'MISMATCH'}"],
               ["A: image-free evaluation", "passed" if a["image_free"]["ok"] else "FAILED"]]
    if B:
        b = B["checks"]
        o = b["observations"]
        flags = [n for n, k in (("counts equal", "counts_equal"), ("ids unique", "vertex_ids_unique"), ("cameras distinct", "cameras_distinct"))
                 if o[k]]
        ch += [["B: val observations pass step 1's checks",
                f"mean vs lift() max |difference| {o['mean_vs_lift_max_abs']:.1e}, re-projection max {o['reprojection_max_px']:.1e} px; "
                + ", ".join(flags)],
               ["B: no probe code imported (its outputs never reach a probe)",
                "passed" if b["no_probe"] else f"FAILED: {', '.join(b['probe_modules_imported'])}"]]
    if C:
        c = C["checks"]
        fu = c["training"].get("first_update") or {}
        yes = lambda x: "passed" if x else "FAILED"  # noqa: E731
        ch += [["C: only the target differs from Experiment 2's student", yes(c["training"]["only_target_differs"])],
               ["C: the target equals the pilot's lifted feature", yes(c["training"]["target_vs_lift_ok"])],
               ["C: one update reaches the encoder",
                f"{yes(fu.get('trainable_changed'))} ({esc(fu.get('trainable_parameter'))} changed; encoder gradient norm "
                f"{fu.get('encoder_grad_norm', float('nan')):.2e})" if fu else "n/a (resumed run)"],
               ["C: image-free evaluation", yes(c["image_free"]["ok"])]]
    add(captioned(table(ch, [8.0 * cm, 8.9 * cm]), "<b>Table 6.</b> The checks each run recorded."))

    add(P("6. Reproducibility", h1))
    rp = [["Item", "Value"], ["Experiment 3 commit / start commit", f"{esc(m['git'].get('commit'))} / {esc(cfg['start_commit'])}"],
          ["Experiment 2 run", esc(cfg["experiment2"]["commit"])]]
    for k, v in runs.items():
        s = v["summary"]
        rp.append([f"Run {k}", f"commit {esc(str(s['git'].get('commit'))[:12])}; {s.get('minutes')} min; peak GPU "
                               f"{gb(s.get('peak_gpu_gb'))}; from {esc(v.get('origin', v['source']))}"])
    if C:
        rp.append(["Run C student", f"{C['training']['session_hours']:.2f} h training; seeds: student 0, probes "
                                    f"0–{C['rows']['C1']['seeds'] - 1}"])
    add(captioned(table(rp, [4.6 * cm, 12.3 * cm]), "<b>Table 7.</b> Every number in this report is in reports/metrics.json."))

    pdf = out_dir / PDF_NAME
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=1.9 * cm, bottomMargin=2.0 * cm,
                            title=m["title"], author="appearance-pt")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return pdf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    cfg = load_config(args.config, args.mode)
    runs = collect_runs(args.out, args.input, args.mode)
    missing = [r for r in ("A", "B", "C") if r not in runs]
    failed = [r for r, v in runs.items() if not run_ok(r, v["summary"])]
    final = args.mode == "full" and not missing and not failed
    reason = ("" if final else
              ("A smoke run. " if args.mode == "smoke" else "") + (f"Runs missing: {', '.join(missing)}. " if missing else "")
              + (f"Runs with failed checks: {', '.join(failed)}. " if failed else ""))
    from appearance_pt.envinfo import collect
    m = dict(status="final" if final else "preliminary", status_reason=reason.strip(), mode=args.mode, title=cfg["title"],
             generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), git=collect()["git"],
             decision_rules=cfg["decision_rules"], runs=runs)
    reports = args.out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(m, indent=1, default=str))
    pdf = build(m, cfg, reports)
    print(f"report ({m['status']}{': ' + m['status_reason'] if reason else ''}): {pdf}; metrics {reports / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
