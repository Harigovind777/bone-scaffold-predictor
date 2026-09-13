"""
Build the shareable accuracy report as a static site - one HTML page plus its figures.

    .venv/bin/python tools/build_site.py                  # -> site/
    .venv/bin/python tools/build_site.py --repo-url URL   # add a link to the source

Every number on the page is read from files the pipeline and the ledger produce -
`results/metrics.json` for where the models stand now, `results/accuracy_baseline.json`
for where they stood before - and the long-form write-up is `ACCURACY.md`, rendered. Nothing
is typed in by hand, so the page cannot drift from the results it reports: re-run the
pipeline, re-run this, and the page follows.

Standard library only, like the web app. Static on purpose, too: a link sent to someone
checking the work has to keep working without a Python process behind it, which is also
why the live predictor is not part of it.
"""

import argparse
import datetime as dt
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
OUT = ROOT / "site"

# Figures placed into the write-up, each after the paragraph that starts with its anchor.
FIGURE_SLOTS = [
    ("The ceiling:", "fig2_model_ladder.png",
     "Model ladder under leave-one-architecture-out validation (left), and how much a "
     "random split would have inflated printability accuracy (right). The best model beats "
     "the textbook power law by {margin:.3f} R², under the 0.02 the figure counts as a "
     "meaningful margin, which is why its title says no model meaningfully beats it."),
    ("**Read that table honestly.**", "fig4_multifidelity.png",
     "Fused model against the coarse mesh alone and the fine mesh alone, as the number "
     "of trusted fine-mesh points grows. The coarse-mesh line is dashed on top because "
     "the fused curve now sits exactly on it."),
]


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

def load_numbers():
    now = json.loads((RESULTS / "metrics.json").read_text())
    was = json.loads((RESULTS / "accuracy_baseline.json").read_text())

    ladder = {r["model"]: r for r in now["model_ladder"]}
    ga_now = max(r["r2"] for m, r in ladder.items() if m.startswith("gibson_ashby"))
    ga_was = max(r["r2"] for m, r in was["ladder"].items() if m.startswith("gibson_ashby"))
    textbook = ladder["gibson_ashby_global"]["r2"]

    fused = {int(r["n_high"]): r for r in now["fusion"]["table"] if r["model"] == "fused"}
    low_only = next(r["r2"] for r in now["fusion"]["table"] if r["model"] == "low_only")
    lo_n, hi_n = min(fused), max(fused)

    ml = now["leakage"]["mlate"]
    p_now, p_was = ml["grouped_full"], was["printability"]["expected_readout"]
    majority = ml["baseline_full"]

    groups = [
        dict(name="Stiffness of topologies the model never saw",
             note="Each model is scored on a scaffold shape held out of training entirely.",
             rows=[
                 dict(label="Physics-informed model", measure="R²",
                      was=was["ladder"]["physics_informed"]["r2"],
                      now=ladder["physics_informed"]["r2"],
                      ref=textbook, ref_label="textbook power law"),
                 dict(label="Best power law", measure="R²",
                      was=ga_was, now=ga_now,
                      ref=textbook, ref_label="textbook power law"),
             ]),
        dict(name="Fusing coarse and fine simulations",
             note=("Fusion now matches the coarse mesh at every budget. It no longer loses "
                   "to it, and on this data the fine mesh has nothing to add."),
             rows=[
                 dict(label=f"Fusion, {lo_n} trusted points", measure="R²",
                      was=was["fusion"]["fused"][str(lo_n)], now=fused[lo_n]["r2"],
                      ref=low_only, ref_label="coarse mesh alone"),
                 dict(label=f"Fusion, {hi_n} trusted points", measure="R²",
                      was=was["fusion"]["fused"][str(hi_n)], now=fused[hi_n]["r2"],
                      ref=low_only, ref_label="coarse mesh alone"),
             ]),
        dict(name="Printability of real inks, grouped by publication",
             note=(f"{majority['accuracy']:.1%} of inks are rated 3, so accuracy is nearly "
                   "won by always answering 3. Weighted kappa scores that answer 0."),
             rows=[
                 dict(label="Printability", measure="weighted kappa",
                      was=p_was["qwk"], now=p_now["qwk"],
                      ref=majority["qwk"], ref_label="answering the most common level every time"),
                 dict(label="Printability", measure="accuracy",
                      was=p_was["accuracy"], now=p_now["accuracy"],
                      ref=majority["accuracy"], ref_label="answering the most common level every time"),
             ]),
    ]
    return groups


# --------------------------------------------------------------------------
# Markdown - the subset ACCURACY.md actually uses, and nothing more
# --------------------------------------------------------------------------

def _inline(text):
    text = html.escape(text, quote=False)
    code = []

    def stash(m):
        code.append(f"<code>{m.group(1)}</code>")
        return f"\x00{len(code) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)
    return re.sub(r"\x00(\d+)\x00", lambda m: code[int(m.group(1))], text)


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", text).lower()).strip("-")


def markdown_to_html(md, figure_html=None):
    """Headings, paragraphs, tables, fenced code, ordered lists, rules. Figures by anchor."""
    figure_html = figure_html or {}
    lines = md.splitlines()
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            body = html.escape("\n".join(lines[i + 1:j]), quote=False)
            out.append(f'<pre class="code"><code>{body}</code></pre>')
            i = j + 1
            continue
        m = re.match(r"^(#{2,3})\s+(.*)$", line)
        if m:
            level, inner = len(m.group(1)), _inline(m.group(2))
            out.append(f'<h{level} id="{_slug(inner)}">{inner}</h{level}>')
            i += 1
            continue
        if line.strip() == "---":
            out.append("<hr>")
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[1:] if not set("".join(r)) <= set("-: ")]
            numeric = [all(re.fullmatch(r"[−+\-]?[\d.]+|—", _plain(r[k])) for r in body)
                       for k in range(len(head))]
            th = "".join(f'<th class="{"num" if numeric[k] else ""}">{_inline(c)}</th>'
                         for k, c in enumerate(head))
            trs = "".join("<tr>" + "".join(
                f'<td class="{"num" if numeric[k] else ""}">{_inline(c)}</td>'
                for k, c in enumerate(r)) + "</tr>" for r in body)
            out.append(f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead>'
                       f'<tbody>{trs}</tbody></table></div>')
            continue
        if re.match(r"^\d+\.\s", line):
            items = []
            while i < len(lines) and (re.match(r"^\d+\.\s", lines[i])
                                      or (items and lines[i].startswith("   "))):
                if re.match(r"^\d+\.\s", lines[i]):
                    items.append(re.sub(r"^\d+\.\s+", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append("<ol>" + "".join(f"<li>{_inline(t)}</li>" for t in items) + "</ol>")
            continue
        para = []
        while (i < len(lines) and lines[i].strip() and not lines[i].startswith(("|", "```", "#"))
               and lines[i].strip() != "---" and not re.match(r"^\d+\.\s", lines[i])):
            para.append(lines[i].strip())
            i += 1
        text = " ".join(para)
        out.append(f"<p>{_inline(text)}</p>")
        for anchor, fig in figure_html.items():
            if text.startswith(anchor):
                out.append(fig)
    return "\n".join(out)


def _plain(cell):
    return re.sub(r"[*`]", "", cell).strip()


def prepare_writeup(md):
    """Drop the H1 and the before/after table the chart at the top already shows."""
    md = re.sub(r"^# .*\n", "", md, count=1)
    start = md.index("## Where it stands")
    table_start = md.index("\n|", start)
    table_end = md.index("\n\n", table_start + 1)
    md = md[:table_start] + md[table_end:]
    orphan = "Not one of those came from a bigger model."
    if orphan not in md:
        raise SystemExit(f"expected sentence not found in ACCURACY.md: {orphan!r}")
    return md.replace(orphan, "Not one of the gains in the chart above came from a bigger model.")


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

def _pct(v):
    return f"{max(0.0, min(1.0, v)) * 100:.2f}%"


def ledger_html(groups):
    ticks = "".join(f'<span class="grid" style="left:{t * 100:.0f}%"></span>'
                    for t in (0, 0.25, 0.5, 0.75, 1))
    parts = []
    for g in groups:
        rows = []
        for r in g["rows"]:
            delta = r["now"] - r["was"]
            tip = (f"{r['label']}, {r['measure']}: {r['was']:.3f} before, {r['now']:.3f} now "
                   f"({delta:+.3f}). {r['ref_label'].capitalize()} scores {r['ref']:.3f}.")
            lo, hi = sorted((r["was"], r["now"]))
            rows.append(f"""
      <div class="row" tabindex="0" data-tip="{html.escape(tip)}">
        <div class="row-label"><span class="row-name">{html.escape(r['label'])}</span>
          <span class="row-measure">{html.escape(r['measure'])}</span></div>
        <div class="track" aria-hidden="true">{ticks}
          <span class="ref" style="left:{_pct(r['ref'])}"></span>
          <span class="bar" style="left:{_pct(lo)};width:calc({_pct(hi)} - {_pct(lo)})"></span>
          <span class="dot was" style="left:{_pct(r['was'])}"></span>
          <span class="dot now" style="left:{_pct(r['now'])}"></span>
        </div>
        <div class="row-values"><span class="v-was">{r['was']:.3f}</span>
          <span class="v-now">{r['now']:.3f}</span></div>
        <p class="sr-only">{html.escape(tip)}</p>
      </div>""")
        parts.append(f"""
    <section class="group">
      <h3 class="group-name">{html.escape(g['name'])}</h3>
      {''.join(rows)}
      <p class="group-note">{html.escape(g['note'])}</p>
    </section>""")
    axis = "".join(f'<span style="left:{t * 100:.0f}%">{t:g}</span>' for t in (0, 0.25, 0.5, 0.75, 1))
    return f"""
  <figure class="ledger" aria-labelledby="ledger-title">
    <div class="ledger-head">
      <h2 id="ledger-title">Before and after</h2>
      <ul class="legend">
        <li><span class="key key-was"></span>before this work</li>
        <li><span class="key key-now"></span>now</li>
        <li><span class="key key-ref"></span>what a trivial answer scores</li>
      </ul>
    </div>
    {''.join(parts)}
    <div class="axis" aria-hidden="true"><div></div><div class="axis-track">{axis}</div><div></div></div>
    <div class="tip" role="status" hidden></div>
  </figure>"""


def figure_block(name, caption):
    return (f'<figure class="plot"><img src="figures/{name}" alt="{html.escape(caption)}" '
            f'loading="lazy"><figcaption>{html.escape(caption)}</figcaption></figure>')


CSS = """
:root{
  color-scheme: light;
  --ground:#E8EBE9; --surface:#FAFBFA; --surface-2:#F1F3F1;
  --line:#D2D7D3; --line-strong:#B7BFB9;
  --ink:#15181A; --ink-2:#39413E; --muted:#5C6663;
  --accent:#5B44CC;
  --was:#8F79E6; --now:#3F2A9E; --ref:#8B9591;
  --display:"Avenir Next Condensed","Avenir Next","Segoe UI","Arial Narrow",system-ui,sans-serif;
  --sans:"Avenir Next","Avenir","Segoe UI",-apple-system,BlinkMacSystemFont,system-ui,sans-serif;
  --prose:Charter,"Bitstream Charter","Iowan Old Style",Palatino,Georgia,serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html{background:var(--ground)}
body{margin:0;background:var(--ground);color:var(--ink);font:17px/1.62 var(--prose);
  -webkit-font-smoothing:antialiased}
a{color:var(--accent);text-underline-offset:.18em}
a:focus-visible,.row:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:4px}
.page{max-width:1040px;margin:0 auto;padding:56px 28px 80px}
header.masthead{max-width:44rem;margin-bottom:40px}
.project{font:600 15px/1.3 var(--sans);color:var(--muted);margin:0 0 14px}
h1{font:700 clamp(38px,6vw,64px)/1.02 var(--display);letter-spacing:-.01em;margin:0 0 18px}
.lede{font-size:19px;color:var(--ink-2);margin:0}

.ledger{background:var(--surface);border:1px solid var(--line);border-radius:14px;
  margin:0 0 64px;padding:28px 32px 20px;position:relative}
.ledger-head{display:flex;flex-wrap:wrap;gap:12px 32px;align-items:baseline;
  justify-content:space-between;margin-bottom:8px}
.ledger h2{font:700 28px/1.1 var(--display);margin:0}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;list-style:none;margin:0;padding:0;
  font:500 14px/1.4 var(--sans);color:var(--ink-2)}
.legend li{display:flex;align-items:center;gap:8px}
.key{display:inline-block;width:10px;height:10px;border-radius:50%}
.key-was{border:2.5px solid var(--was);background:var(--surface)}
.key-now{background:var(--now)}
.key-ref{width:2px;height:14px;border-radius:1px;background:var(--ref)}

.group{padding-top:22px}
.group + .group{border-top:1px solid var(--line);margin-top:6px}
.group-name{font:600 16px/1.3 var(--sans);color:var(--ink);margin:0 0 6px}
.group-note{font:400 14px/1.5 var(--sans);color:var(--muted);margin:6px 0 14px;max-width:62ch}
.row{display:grid;grid-template-columns:minmax(170px,230px) 1fr 132px;gap:0 24px;
  align-items:center;min-height:46px;cursor:default}
.row-label{display:flex;flex-direction:column;line-height:1.25}
.row-name{font:500 15px/1.3 var(--sans)}
.row-measure{font:400 13px/1.3 var(--sans);color:var(--muted)}
.row-values{display:flex;justify-content:flex-end;gap:14px;font:14px/1 var(--sans);
  font-variant-numeric:tabular-nums}
.v-was{color:var(--muted)}
.v-now{color:var(--ink);font-weight:700}
.track{position:relative;height:30px}
.grid{position:absolute;top:0;bottom:0;width:1px;background:var(--line)}
.ref{position:absolute;top:4px;bottom:4px;width:2px;margin-left:-1px;border-radius:1px;background:var(--ref)}
.bar{position:absolute;top:50%;height:2px;margin-top:-1px;background:var(--line-strong)}
.dot{position:absolute;top:50%;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;
  box-shadow:0 0 0 2px var(--surface)}
.dot.was{background:var(--surface);border:2.5px solid var(--was)}
.dot.now{background:var(--now)}
.row:hover .dot.now,.row:focus-visible .dot.now{box-shadow:0 0 0 2px var(--surface),0 0 0 4px var(--was)}
.axis{display:grid;grid-template-columns:minmax(170px,230px) 1fr 132px;gap:0 24px;margin-top:4px}
.axis-track{position:relative;height:20px;font:12px/1 var(--sans);color:var(--muted);
  font-variant-numeric:tabular-nums}
.axis-track span{position:absolute;top:4px;transform:translateX(-50%)}
.tip{position:absolute;z-index:2;max-width:300px;padding:9px 12px;border-radius:8px;
  background:var(--ink);color:#fff;font:14px/1.45 var(--sans);pointer-events:none}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}

.writeup{max-width:44rem}
.writeup h2{font:700 34px/1.1 var(--display);margin:64px 0 16px}
.writeup h3{font:600 21px/1.3 var(--sans);margin:40px 0 10px}
.writeup p{margin:0 0 18px}
.writeup hr{border:0;margin:0}
.writeup ol{padding-left:1.3em}
.writeup li{margin-bottom:8px}
.writeup code{font:.84em/1 var(--mono);background:var(--surface-2);padding:.12em .36em;border-radius:4px}
pre.code{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:14px 18px;overflow-x:auto;font:14px/1.6 var(--mono);margin:0 0 22px}
pre.code code{background:none;padding:0;font:inherit}
.table-wrap{overflow-x:auto;margin:6px 0 24px}
.table-wrap,pre.code,figure.plot{width:min(960px, calc(100vw - 56px))}
table{border-collapse:collapse;width:100%;font:14.5px/1.45 var(--sans)}
th{font-weight:600;text-align:left;border-bottom:1.5px solid var(--line-strong);padding:8px 14px 8px 0}
td{border-bottom:1px solid var(--line);padding:8px 14px 8px 0;vertical-align:top}
th.num,td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table code{font-size:.9em}
figure.plot{margin:8px 0 34px}
figure.plot img{display:block;width:100%;height:auto;background:#fcfcfb;border:1px solid var(--line);
  border-radius:10px}
figure.plot figcaption{font:14px/1.5 var(--sans);color:var(--muted);margin-top:10px;max-width:62ch}
footer{margin-top:72px;padding-top:22px;border-top:1px solid var(--line-strong);
  font:14px/1.6 var(--sans);color:var(--muted);max-width:44rem}
footer p{margin:0 0 8px}

@media (max-width:720px){
  .page{padding:36px 18px 64px}
  .table-wrap,pre.code,figure.plot{width:calc(100vw - 36px)}
  .ledger{padding:22px 18px 16px}
  .row,.axis{grid-template-columns:1fr 104px;gap:4px 12px}
  .row-label{grid-column:1 / -1;flex-direction:row;gap:8px;align-items:baseline;margin-top:8px}
  .axis > div:first-child{display:none}
}
"""

JS = """
(() => {
  const ledger = document.querySelector('.ledger');
  const tip = ledger.querySelector('.tip');
  const show = row => {
    tip.textContent = row.dataset.tip;
    tip.hidden = false;
    const box = ledger.getBoundingClientRect(), r = row.querySelector('.track').getBoundingClientRect();
    const left = Math.min(Math.max(r.left - box.left, 12), box.width - tip.offsetWidth - 12);
    tip.style.left = left + 'px';
    tip.style.top = (r.top - box.top - tip.offsetHeight - 6) + 'px';
  };
  const hide = () => { tip.hidden = true; };
  ledger.querySelectorAll('.row').forEach(row => {
    row.addEventListener('mouseenter', () => show(row));
    row.addEventListener('mouseleave', hide);
    row.addEventListener('focus', () => show(row));
    row.addEventListener('blur', hide);
  });
})();
"""


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def build(repo_url=None):
    groups = load_numbers()
    now = json.loads((RESULTS / "metrics.json").read_text())
    ladder = {r["model"]: r["r2"] for r in now["model_ladder"]}
    margin = max(ladder.values()) - ladder["gibson_ashby_global"]
    fill = dict(margin=margin)
    figures = {anchor: figure_block(name, caption.format(**fill))
               for anchor, name, caption in FIGURE_SLOTS}
    writeup_md = prepare_writeup((ROOT / "ACCURACY.md").read_text())
    for anchor in figures:
        if anchor not in writeup_md:
            raise SystemExit(f"figure anchor not found in ACCURACY.md: {anchor!r}")
    writeup = markdown_to_html(writeup_md, figures)

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    built = dt.date.today().strftime("%-d %B %Y")
    source = (f'<p>The code behind every number: <a href="{html.escape(repo_url)}">'
              f'the repository at this commit</a>.</p>' if repo_url else "")
    state = " with uncommitted changes" if dirty else ""

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bone scaffold predictor accuracy</title>
<meta name="description" content="How accurate the bone scaffold property predictor is, what was changed to improve it, and how each number was measured.">
<style>{CSS}</style>
</head>
<body>
<main class="page">
  <header class="masthead">
    <p class="project">Bone scaffold property prediction and inverse design</p>
    <h1>How accurate are the models, and how do we know?</h1>
    <p class="lede">Three models predict how a biopolymer bone scaffold will behave: how
    stiff it is, whether its ink will print, and how far cheap simulations can stand in for
    expensive ones. Each was scored on data it never trained on, before and after a round
    of accuracy work. Hover over a row, or tab to it, for its exact change.</p>
  </header>
  {ledger_html(groups)}
  <article class="writeup">
  {writeup}
  </article>
  <footer>
    <p>Every number on this page is generated from the pipeline's own output files by
    <code>tools/build_site.py</code>, built from commit <code>{commit}</code>{state} on {built}.</p>
    {source}
  </footer>
</main>
<script>{JS}</script>
</body>
</html>
"""
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)
    (OUT / "index.html").write_text(page)
    (OUT / ".nojekyll").write_text("")          # serve files as-is, no Jekyll pass
    for _, name, _ in FIGURE_SLOTS:
        shutil.copy2(FIGURES / name, OUT / "figures" / name)
    return OUT / "index.html"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-url", help="link to the source repository in the footer")
    args = ap.parse_args()
    print(f"wrote {build(args.repo_url).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
