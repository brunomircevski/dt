#!/usr/bin/env python3

import math
import re
import sys
from dataclasses import dataclass, field
from html import escape


H_SPACING = 190
V_SPACING = 130
MARGIN_X = 40
MARGIN_Y = 40
BOX_WIDTH = 150
BOX_HEIGHT = 56
CORNER_RADIUS = 14

NODE_RE = re.compile(r"^(?P<edge>\w+) \[n=(?P<count>\d+)\]: (?P<body>.+)$")
LEAF_RE = re.compile(r"^Leaf -> (?P<label>.+)$")
DECISION_RE = re.compile(r"^if (?P<feature>.+) <= (?P<threshold>-?\d+(?:\.\d+)?)$")


@dataclass
class LayoutNode:
    is_leaf: bool
    sample_count: int
    edge_label: str
    leaf_label: str | None = None
    feature_name: str | None = None
    threshold: float | None = None
    children: list["LayoutNode"] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    subtree_left: float = 0.0
    subtree_right: float = 0.0


def parse_tree(lines: list[str]) -> LayoutNode:
    stack: list[tuple[int, LayoutNode]] = []
    root: LayoutNode | None = None

    for raw_line in lines:
        if not raw_line.strip():
            continue

        depth = (len(raw_line) - len(raw_line.lstrip(" "))) // 2
        line = raw_line.strip()

        match = NODE_RE.match(line)
        if not match:
            raise ValueError(f"Unsupported tree line: {line}")

        edge_label = match.group("edge")
        sample_count = int(match.group("count"))
        body = match.group("body")

        leaf_match = LEAF_RE.match(body)
        if leaf_match:
            node = LayoutNode(
                is_leaf=True,
                sample_count=sample_count,
                edge_label=edge_label,
                leaf_label=leaf_match.group("label"),
            )
        else:
            decision_match = DECISION_RE.match(body)
            if not decision_match:
                raise ValueError(f"Unsupported decision text: {body}")
            node = LayoutNode(
                is_leaf=False,
                sample_count=sample_count,
                edge_label=edge_label,
                feature_name=decision_match.group("feature"),
                threshold=float(decision_match.group("threshold")),
            )

        while stack and stack[-1][0] >= depth:
            stack.pop()

        if stack:
            stack[-1][1].children.append(node)
        else:
            root = node

        stack.append((depth, node))

    if root is None:
        raise ValueError("Printed tree is empty.")

    return root


def assign_positions(node: LayoutNode, depth: int, next_leaf_x: list[float]) -> None:
    node.y = MARGIN_Y + depth * V_SPACING

    if not node.children:
        node.x = next_leaf_x[0]
        next_leaf_x[0] += H_SPACING
        node.subtree_left = node.x
        node.subtree_right = node.x
        return

    for child in node.children:
        assign_positions(child, depth + 1, next_leaf_x)

    node.x = (node.children[0].x + node.children[-1].x) / 2.0
    node.subtree_left = node.children[0].subtree_left
    node.subtree_right = node.children[-1].subtree_right


def gather_edges(node: LayoutNode, edges: list[tuple[LayoutNode, LayoutNode]]) -> None:
    for child in node.children:
        edges.append((node, child))
        gather_edges(child, edges)


def gather_nodes(node: LayoutNode, nodes: list[LayoutNode]) -> None:
    nodes.append(node)
    for child in node.children:
        gather_nodes(child, nodes)


def node_label(node: LayoutNode) -> list[str]:
    if node.is_leaf:
        return [f"Leaf: {node.leaf_label}", f"n={node.sample_count}"]

    return [f"{node.feature_name} <= {node.threshold:.3f}", f"n={node.sample_count}"]


def render_svg(root: LayoutNode) -> str:
    edges: list[tuple[LayoutNode, LayoutNode]] = []
    nodes: list[LayoutNode] = []
    gather_edges(root, edges)
    gather_nodes(root, nodes)

    width = int(math.ceil(root.subtree_right - root.subtree_left + 2 * MARGIN_X + BOX_WIDTH))
    height = int(math.ceil(max(node.y for node in nodes) + BOX_HEIGHT + MARGIN_Y))

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: 'DejaVu Sans Mono', monospace; fill: #14213d; }",
        ".edge { stroke: #6c7a89; stroke-width: 2; fill: none; }",
        ".edge-label { font-size: 12px; font-weight: 700; fill: #415a77; }",
        ".decision { fill: #e0fbfc; stroke: #1d3557; stroke-width: 2; }",
        ".leaf { fill: #fefae0; stroke: #bc6c25; stroke-width: 2; }",
        ".node-title { font-size: 12px; font-weight: 700; }",
        ".node-meta { font-size: 11px; }",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8f9fa" />',
    ]

    for parent, child in edges:
        x1 = parent.x + BOX_WIDTH / 2
        y1 = parent.y + BOX_HEIGHT
        x2 = child.x + BOX_WIDTH / 2
        y2 = child.y
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 - 8
        svg_parts.append(
            f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" />'
        )
        svg_parts.append(
            f'<text class="edge-label" x="{mid_x:.1f}" y="{mid_y:.1f}" text-anchor="middle">{escape(child.edge_label)}</text>'
        )

    for node in nodes:
        css_class = "leaf" if node.is_leaf else "decision"
        svg_parts.append(
            f'<rect class="{css_class}" x="{node.x:.1f}" y="{node.y:.1f}" width="{BOX_WIDTH}" height="{BOX_HEIGHT}" rx="{CORNER_RADIUS}" ry="{CORNER_RADIUS}" />'
        )
        title, meta = node_label(node)
        center_x = node.x + BOX_WIDTH / 2
        svg_parts.append(
            f'<text class="node-title" x="{center_x:.1f}" y="{node.y + 23:.1f}" text-anchor="middle">{escape(title)}</text>'
        )
        svg_parts.append(
            f'<text class="node-meta" x="{center_x:.1f}" y="{node.y + 41:.1f}" text-anchor="middle">{escape(meta)}</text>'
        )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: render_tree_svg.py input.txt output.svg", file=sys.stderr)
        return 1

    input_path, output_path = sys.argv[1], sys.argv[2]
    with open(input_path, "r", encoding="utf-8") as input_file:
        root = parse_tree(input_file.read().splitlines())

    assign_positions(root, depth=0, next_leaf_x=[MARGIN_X])
    svg = render_svg(root)

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(svg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
