#!/usr/bin/env python3
"""Draw a tree result (fit.py's JSON) as a standalone SVG: the run's facts at
the top, the decision regions on the chess board at the top right, the tree
below. No dependencies.

  render.py RESULT.json [...] --out DIR

Small trees (at most 64 leaves) get a box per node with the split, the rows
and the class mix; bigger ones a compact layout (dots, colored leaves, labels
only where they fit). Every node has a tooltip with its full numbers. The SVG
follows the viewer's light or dark mode.
"""
import argparse, json, sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import treeio  # noqa: E402

FULL_MAX_LEAVES = 64
HEADER_H = 250
MARGIN = 28
BOARD = 3.0  # the chess board spans [0, 3] in x and y

LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "faint": "#8a8984", "grid": "#e4e3df",
         "edge": "#b9b8b2", "box": "#ffffff", "a": "#2a78d6", "b": "#eb6834", "a_tint": "#d9e7f8",
         "b_tint": "#fbe0d4"}
DARK = {"surface": "#1a1a19", "ink": "#f0efec", "muted": "#c3c2b7", "faint": "#8f8e86", "grid": "#3a3936",
        "edge": "#5d5c57", "box": "#242422", "a": "#3987e5", "b": "#d95926", "a_tint": "#1f3550",
        "b_tint": "#4a2a1c"}


def colors(c):
    """The color rules for one mode, with plain values (no CSS variables, which
    some SVG renderers such as librsvg ignore)."""
    return f"""
.bg {{ fill: {c['surface']}; }} text {{ fill: {c['ink']}; }}
.muted, .n, .vs, .k {{ fill: {c['muted']}; }} .faint, .lab {{ fill: {c['faint']}; }}
.edge {{ stroke: {c['edge']}; }} .box {{ fill: {c['box']}; stroke: {c['edge']}; }}
.leaf-A {{ fill: {c['a_tint']}; stroke: {c['a']}; }} .leaf-B {{ fill: {c['b_tint']}; stroke: {c['b']}; }}
.ca {{ fill: {c['a']}; }} .cb {{ fill: {c['b']}; }} .rA {{ fill: {c['a_tint']}; }} .rB {{ fill: {c['b_tint']}; }}
.rline {{ stroke: {c['surface']}; }} .board {{ stroke: {c['ink']}; }} .frame {{ stroke: {c['grid']}; }}
.dot {{ fill: {c['muted']}; }} .rule {{ stroke: {c['grid']}; }}
"""


STYLE = """
text { font-family: Inter, "Helvetica Neue", Arial, sans-serif; }
.mono { font-family: "JetBrains Mono", Menlo, Consolas, monospace; }
.title { font-size: 22px; font-weight: 700; } .sub { font-size: 13px; }
.k { font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }
.v { font-size: 20px; font-weight: 650; } .vs { font-size: 11.5px; }
.edge { stroke-width: 1.4; fill: none; } .box { stroke-width: 1; }
.leaf-A, .leaf-B { stroke-width: 1.2; }
.rline { stroke-width: 0.6; fill: none; }
.board { stroke-width: 1; fill: none; stroke-dasharray: 3 3; opacity: .55; }
.frame { fill: none; } .rule { stroke-width: 1; }
.split { font-size: 12.5px; font-weight: 650; } .n { font-size: 11px; } .lab { font-size: 10px; }
""" + colors(LIGHT) + "@media (prefers-color-scheme: dark) {" + colors(DARK) + "}\n"


def fmt_time(s):
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1e3:.1f} ms"
    return f"{s:.2f} s"


def fmt_thr(t):
    return f"{t:.6g}"


def backend_text(m):
    if m["impl"] == "tree":
        return {"serial": "serial, 1 thread", "parallel": f"parallel, {m['threads']} threads",
                "cuda": f"CUDA + {m['threads']} CPU threads"}[m["backend"]]
    if m["backend"] == "cuda":
        return "CUDA"
    return f"{m['threads']} thread{'s' if m['threads'] != 1 else ''}"


def node_tip(node, depth):
    lines = [f"depth {depth}"]
    if not treeio.is_leaf(node):
        lines.append(f"{node['feature']} {node['op']} {node['threshold']!r}")
    if node.get("n") is not None:
        lines.append(f"rows {node['n']:,}")
    if node.get("dist"):
        lines.append("  ".join(f"{c}: {k:,}" for c, k in node["dist"].items()))
    if treeio.is_leaf(node):
        lines.append(f"class {node['class']}")
    return escape("\n".join(lines))


def layout(tree):
    """x slot (leaves in order 0, 1, ...; a split sits midway over its leaves)
    and depth for every node."""
    pos = []
    counter = [0]

    def visit(node, depth):
        if treeio.is_leaf(node):
            x = counter[0]
            counter[0] += 1
        else:
            a = visit(node["left"], depth + 1)
            b = visit(node["right"], depth + 1)
            x = (a + b) / 2
        pos.append((node, x, depth))
        node["_x"], node["_d"] = x, depth
        return x
    visit(tree, 0)
    return pos, counter[0]


def regions(tree):
    """Leaf rectangles on the board: (x0, x1, y0, y1, class)."""
    out = []

    def visit(node, box):
        if treeio.is_leaf(node):
            out.append((*box, node["class"]))
            return
        x0, x1, y0, y1 = box
        t = node["threshold"]
        if node["feature"] == "x":
            visit(node["left"], (x0, min(x1, t), y0, y1))
            visit(node["right"], (max(x0, t), x1, y0, y1))
        else:
            visit(node["left"], (x0, x1, y0, min(y1, t)))
            visit(node["right"], (x0, x1, max(y0, t), y1))
    visit(tree, (0.0, BOARD, 0.0, BOARD))
    return out


def mix_bar(out, x, y, w, dist, h=5):
    total = sum(dist.values()) if dist else 0
    if not total:
        return
    a = w * dist.get("A", 0) / total
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{a:.1f}" height="{h}" class="ca"/>')
    out.append(f'<rect x="{x + a:.1f}" y="{y:.1f}" width="{w - a:.1f}" height="{h}" class="cb"/>')


def draw_tree(out, tree, x0, y0, slot, level, full):
    pos, _ = layout(tree)
    bw, bh = slot - 14, 58

    def cx(node):
        return x0 + node["_x"] * slot + slot / 2

    def cy(node):
        return y0 + node["_d"] * level

    # Edges first, under the nodes.
    for node, _, _ in pos:
        if treeio.is_leaf(node):
            continue
        for child, word in ((node["left"], "yes"), (node["right"], "no")):
            x1, y1 = cx(node), cy(node) + (bh / 2 if full else 0)
            x2, y2 = cx(child), cy(child) - (bh / 2 if full else 0)
            ym = (y1 + y2) / 2
            out.append(f'<path class="edge" d="M{x1:.1f},{y1:.1f} C{x1:.1f},{ym:.1f} {x2:.1f},{ym:.1f} '
                       f'{x2:.1f},{y2:.1f}"/>')
            if full:
                out.append(f'<text class="lab" x="{(x1 + x2) / 2 + (-4 if word == "yes" else 4):.1f}" '
                           f'y="{ym - 3:.1f}" text-anchor="{"end" if word == "yes" else "start"}">'
                           f'{word}</text>')
    for node, _, depth in pos:
        x, y = cx(node), cy(node)
        tip = f"<title>{node_tip(node, depth)}</title>"
        leaf = treeio.is_leaf(node)
        n = node.get("n")
        if full:
            cls = f'leaf-{node["class"]}' if leaf else "box"
            out.append(f'<g>{tip}<rect x="{x - bw / 2:.1f}" y="{y - bh / 2:.1f}" width="{bw}" '
                       f'height="{bh}" rx="7" class="{cls}"/>')
            if leaf:
                purity = ""
                if node.get("dist") and n:
                    purity = f" · {100 * node['dist'][node['class']] / n:.1f}%"
                out.append(f'<text class="split" x="{x:.1f}" y="{y - 6:.1f}" text-anchor="middle">'
                           f'class {node["class"]}</text>')
                out.append(f'<text class="n" x="{x:.1f}" y="{y + 12:.1f}" text-anchor="middle">'
                           f'{"" if n is None else f"{n:,} rows"}{purity}</text>')
            else:
                op = "≤" if node["op"] == "<=" else "&lt;"
                out.append(f'<text class="split mono" x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle">'
                           f'{node["feature"]} {op} {fmt_thr(node["threshold"])}</text>')
                out.append(f'<text class="n" x="{x:.1f}" y="{y + 6:.1f}" text-anchor="middle">'
                           f'{"" if n is None else f"{n:,} rows"}</text>')
                mix_bar(out, x - bw / 2 + 10, y + 14, bw - 20, node.get("dist"))
            out.append("</g>")
        else:
            if leaf:
                out.append(f'<g>{tip}<rect x="{x - slot / 2 + 1:.1f}" y="{y - 5:.1f}" '
                           f'width="{slot - 2:.1f}" height="12" rx="2" class="leaf-{node["class"]}"/></g>')
            else:
                out.append(f'<g>{tip}<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" class="dot"/></g>')
                span = slot * (count_leaves(node))
                if span >= 110:
                    op = "≤" if node["op"] == "<=" else "&lt;"
                    out.append(f'<text class="lab mono" x="{x:.1f}" y="{y - 7:.1f}" text-anchor="middle">'
                               f'{node["feature"]} {op} {fmt_thr(node["threshold"])}</text>')


def count_leaves(node):
    return sum(1 for n in treeio.iter_nodes(node) if treeio.is_leaf(n))


def draw_board(out, tree, x, y, size):
    s = size / BOARD
    out.append(f'<text class="k" x="{x}" y="{y - 10}">decision regions</text>')
    for x0, x1, y0, y1, c in regions(tree):
        if x1 <= x0 or y1 <= y0:
            continue
        # y grows upwards on the board.
        out.append(f'<rect x="{x + x0 * s:.2f}" y="{y + (BOARD - y1) * s:.2f}" width="{(x1 - x0) * s:.2f}" '
                   f'height="{(y1 - y0) * s:.2f}" class="r{c}"/>')
        out.append(f'<rect x="{x + x0 * s:.2f}" y="{y + (BOARD - y1) * s:.2f}" width="{(x1 - x0) * s:.2f}" '
                   f'height="{(y1 - y0) * s:.2f}" class="rline"/>')
    for k in (1, 2):
        out.append(f'<line class="board" x1="{x + k * s:.1f}" y1="{y}" x2="{x + k * s:.1f}" y2="{y + size}"/>')
        out.append(f'<line class="board" x1="{x}" y1="{y + k * s:.1f}" x2="{x + size}" y2="{y + k * s:.1f}"/>')
    out.append(f'<rect class="frame" x="{x}" y="{y}" width="{size}" height="{size}"/>')
    out.append(f'<text class="lab" x="{x + size / 2}" y="{y + size + 13}" text-anchor="middle">x  (dashed: '
               f'the true 3×3 board)</text>')
    out.append(f'<text class="lab" x="{x - 6}" y="{y + size / 2}" text-anchor="end">y</text>')


def render(result):
    m, tree = result["meta"], result["tree"]
    _, leaves = layout(tree)
    full = leaves <= FULL_MAX_LEAVES
    slot = 132 if full else max(6.0, min(18.0, 2400 / leaves))
    level = 104 if full else 56
    tree_w = leaves * slot
    board = 170
    width = max(1180, tree_w + 2 * MARGIN)
    tree_y = HEADER_H + 40 + (34 if full else 20)
    height = tree_y + m["depth"] * level + (56 if full else 30)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
           f'viewBox="0 0 {width:.0f} {height:.0f}" role="img">',
           f"<title>{escape(m['label'])} {escape(m['case'].upper())} tree, chess 1M, depth limit "
           f"{m['depth_limit']}</title>", f"<style>{STYLE}</style>",
           f'<rect class="bg" width="{width:.0f}" height="{height:.0f}"/>']
    case = "CART" if m["case"] == "cart" else "C4.5"
    where = m["gpu"] or m["cpu"]
    out.append(f'<text class="title" x="{MARGIN}" y="44">{escape(m["label"])} · {case} · '
               f'{escape(backend_text(m))}</text>')
    out.append(f'<text class="sub muted" x="{MARGIN}" y="66">{escape(m["algorithm"])} · '
               f'{escape(m["host"])} ({escape(where)}) · {escape(m["version"])}</text>')
    tiles = [
        ("training time", fmt_time(m["median_seconds"]),
         f"median of {len(m['times'])}, {fmt_time(m['min_seconds'])} – {fmt_time(m['max_seconds'])}"),
        ("test accuracy", f"{100 * m['test_accuracy']:.2f}%", f"{m['n_test']:,} rows (last 20%)"),
        ("train accuracy", f"{100 * m['train_accuracy']:.2f}%", f"{m['n_train']:,} rows (first 80%)"),
        ("depth", f"{m['depth']}", f"limit {m['depth_limit']}" +
         (f", grown {m['grown_depth']}, cut after" if m.get("splits_cut_at_depth_limit") else "")),
        ("nodes", f"{m['nodes']:,}", f"{m['leaves']:,} leaves"),
    ]
    tx = MARGIN
    tile_w = (min(width, 1180) - 2 * MARGIN - board - 40) / len(tiles)
    for k, v, s in tiles:
        out.append(f'<text class="k" x="{tx:.0f}" y="106">{escape(k)}</text>')
        out.append(f'<text class="v" x="{tx:.0f}" y="134">{escape(v)}</text>')
        out.append(f'<text class="vs" x="{tx:.0f}" y="153">{escape(s)}</text>')
        tx += tile_w
    notes = [f"{m['dataset']} · no pruning · timed: {m['timer']}", f"run: {m['command']}"]
    if m.get("test_export_mismatches") is not None:
        notes.append(f"check: the drawn tree predicts all {m['n_test']:,} test rows as "
                     f"{escape(m['label'])}'s own predict() does" if m["test_export_mismatches"] == 0 else
                     f"warning: the drawn tree differs from {m['label']}'s predict() on "
                     f"{m['test_export_mismatches']:,} test rows")
    if m.get("depth_limit_note"):
        notes.append(m["depth_limit_note"] + f" ({m['splits_cut_at_depth_limit']} splits cut)")
    y = 182
    for note in notes:
        out.append(f'<text class="vs" x="{MARGIN}" y="{y}">{escape(note)}</text>')
        y += 16
    bx = min(width, 1180) - MARGIN - board
    draw_board(out, tree, bx, 30, board)
    # legend
    lx = MARGIN
    ly = HEADER_H + 4
    out.append(f'<line class="rule" x1="{MARGIN}" y1="{HEADER_H - 12}" x2="{width - MARGIN:.0f}" '
               f'y2="{HEADER_H - 12}"/>')
    for c in treeio.CLASSES:
        out.append(f'<rect x="{lx}" y="{ly - 9}" width="12" height="12" rx="2" class="leaf-{c}"/>')
        out.append(f'<text class="vs" x="{lx + 18}" y="{ly + 1}">class {c}</text>')
        lx += 80
    out.append(f'<text class="vs" x="{lx + 10}" y="{ly + 1}">left branch: the condition holds (yes). '
               f'{"Bar: the class mix of the rows reaching the node." if full else "Hover a node for its numbers."}'
               f'</text>')
    x0 = (width - tree_w) / 2
    draw_tree(out, tree, x0, tree_y, slot, level, full)
    out.append("</svg>")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for path in a.results:
        result = json.load(open(path))
        target = Path(a.out) / (Path(path).stem + ".svg")
        target.write_text(render(result))
        print(target)


if __name__ == "__main__":
    main()
