"""Rebuild the bilingual figures for the neural-network training article.

Requires Python 3, NumPy, Matplotlib and fontTools. Diagrams are exact SVG
drawings; numeric plots are made with Matplotlib. Example:
    python generate-figures.py --font /path/to/Regular.ttf --bold-font /path/to/Bold.ttf

The font arguments supply complete Korean fonts. All visible text is stored as
SVG paths, so the published figures do not require fonts on the reader's device.
"""

from argparse import ArgumentParser
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen


OUT = Path(__file__).resolve().parent
WIDTH = 760
BG, INK, MUTED, LINE = "#f6f8fc", "#16243b", "#52627b", "#cbd5e1"
BLUE, BLUE_BG = "#1d4ed8", "#dbeafe"
TEAL, TEAL_BG = "#0f766e", "#ccfbf1"
ORANGE, ORANGE_BG = "#b45309", "#ffedd5"
PURPLE, PURPLE_BG = "#6d28d9", "#ede9fe"
RED, RED_BG = "#be123c", "#ffe4e6"
OUTLINE_FONTS = {}


def outline_diagram_text(path):
    """Embed reusable glyph outlines, preserving each label as accessible text.

    Read the complete glyph table before writing anything. A truncated font
    can expose names and character maps while containing no usable outlines.
    Loading its glyph set here fails explicitly instead of drawing blank text.
    """
    ns = "http://www.w3.org/2000/svg"
    xlink = "http://www.w3.org/1999/xlink"
    ET.register_namespace("", ns)
    ET.register_namespace("xlink", xlink)
    tree = ET.parse(path)
    root = tree.getroot()
    defs = root.find(f"{{{ns}}}defs")
    glyph_cache = {}
    for label in list(root.findall(f"{{{ns}}}text")):
        text = label.text or ""
        mono = label.get("class") == "num"
        bold = label.get("font-weight") == "700"
        key = (mono, bold)
        font = OUTLINE_FONTS[key]
        cmap = font.getBestCmap()
        font_tag = f"{int(mono)}-{int(bold)}"
        if any(ord(c) not in cmap for c in text):
            font = OUTLINE_FONTS[("fallback", bold)]
            cmap = font.getBestCmap()
            font_tag = f"math-{int(bold)}"
        glyph_set = font.getGlyphSet()
        metrics = font["hmtx"].metrics
        scale = float(label.get("font-size")) / font["head"].unitsPerEm
        missing = [c for c in text if ord(c) not in cmap]
        if missing:
            raise ValueError(f"Font has no glyphs for {missing!r} in {text!r}")
        advance = sum(metrics[cmap[ord(c)]][0] for c in text)
        x, y = float(label.get("x")), float(label.get("y"))
        anchor = label.get("text-anchor", "start")
        if anchor == "middle":
            x -= advance * scale / 2
        elif anchor == "end":
            x -= advance * scale
        group = ET.Element(f"{{{ns}}}g", {
            "fill": label.get("fill", INK),
            "transform": f"translate({x:.5f} {y:.5f}) scale({scale:.8f} {-scale:.8f})",
            "data-text": text,
            "aria-label": text,
        })
        cursor = 0
        for char in text:
            code = ord(char)
            glyph_key = (font_tag, code)
            glyph_name = cmap[code]
            glyph_id = f"glyph-{font_tag}-{code:x}"
            if glyph_key not in glyph_cache:
                pen = SVGPathPen(glyph_set)
                glyph_set[glyph_name].draw(pen)
                ET.SubElement(defs, f"{{{ns}}}path", {"id": glyph_id, "d": pen.getCommands()})
                glyph_cache[glyph_key] = glyph_id
            ET.SubElement(group, f"{{{ns}}}use", {
                f"{{{xlink}}}href": f"#{glyph_id}", "x": str(cursor),
            })
            cursor += metrics[glyph_name][0]
        index = list(root).index(label)
        root.remove(label)
        root.insert(index, group)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def pick(lang, ko, en):
    return ko if lang == "ko" else en


class Diagram:
    """A small SVG writer with explicit positions and accessible descriptions."""

    def __init__(self, lang, height, title, desc):
        self.lang, self.height = lang, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height}" '
            f'viewBox="0 0 760 {height}" role="img" aria-labelledby="title desc" xml:lang="{lang}">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>',
            '<style>text{font-family:"Noto Sans KR","Noto Sans CJK KR",'
            '"Pretendard","Apple SD Gothic Neo","Malgun Gothic",sans-serif;'
            'font-variant-numeric:tabular-nums}.num{font-family:ui-monospace,'
            'SFMono-Regular,Consolas,"DejaVu Sans Mono",monospace}</style>',
            f'<rect width="760" height="{height}" rx="18" fill="{BG}"/>',
        ]
        self.text(30, 44, title, 25, bold=True)

    def raw(self, value):
        self.parts.append(value)

    def rect(self, x, y, w, h, fill="white", stroke=LINE, radius=12, width=1):
        self.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                 f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def text(self, x, y, value, size=19, color=INK, bold=False, anchor="start", mono=False):
        css_class = ' class="num"' if mono else ''
        self.raw(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
                 f'font-weight="{700 if bold else 400}" text-anchor="{anchor}" xml:space="preserve"'
                 f'{css_class}>{escape(str(value))}</text>')

    def lines(self, x, y, values, size=18, color=MUTED, gap=28, bold=False, anchor="start"):
        for i, value in enumerate(values):
            self.text(x, y + i * gap, value, size, color, bold, anchor)

    def line(self, x1, y1, x2, y2, color=MUTED, width=2, arrow=False):
        arrow_attr = ' marker-end="url(#arrow)"' if arrow else ''
        self.raw(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                 f'stroke="{color}" stroke-width="{width}"{arrow_attr}/>')

    def path(self, path, color=MUTED, width=2, arrow=False):
        arrow_attr = ' marker-end="url(#arrow)"' if arrow else ''
        self.raw(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}"{arrow_attr}/>')

    def badge(self, x, y, text, color=INK, width=116):
        self.rect(x - width / 2, y - 24, width, 34, "white", LINE, 8)
        self.text(x, y, text, 18, color, True, "middle", True)

    def matrix(self, x, y, rows, color=BLUE, fill=BLUE_BG, cw=62, ch=48,
               rowlabels=None, collabels=None):
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                self.rect(x + j * cw, y + i * ch, cw, ch, fill, LINE, 0)
                self.text(x + (j + .5) * cw, y + (i + .5) * ch + 8,
                          str(value).replace("-", "−"), 23, color, True, "middle", True)
            if rowlabels:
                self.text(x - 14, y + (i + .5) * ch + 7, rowlabels[i], 18,
                          MUTED, True, "end")
        if collabels:
            for j, label in enumerate(collabels):
                self.text(x + (j + .5) * cw, y - 12, label, 18, MUTED, anchor="middle")

    def save(self, stem):
        self.parts.append("</svg>")
        path = OUT / f"{stem}-{self.lang}.svg"
        path.write_text("\n".join(self.parts) + "\n", encoding="utf-8")
        outline_diagram_text(path)


def workflow(lang):
    t = lambda ko, en: pick(lang, ko, en)
    d = Diagram(lang, 1040,
                t("배치 하나를 학습하는 전체 순서", "One training step, from input to update"),
                t("입력 X로 예측을 계산하고 정답 y로 손실을 계산한다. 순전파, 손실 계산, 역전파 동안 파라미터는 고정되어 있다. optimizer.step에서만 수정한 뒤 다음 배치로 돌아간다.",
                  "Inputs X produce predictions; targets y enter the loss. Parameters stay fixed during forward, loss, and backward. optimizer.step updates them before the next batch."))
    d.text(30, 78, t("학습 준비: 모델 구조와 파라미터 초기값을 정한다.",
                     "Preparation: define the model and initialize its parameters."), 18, MUTED)
    d.rect(66, 111, 428, 86, BLUE_BG, BLUE)
    d.text(86, 143, t("배치의 입력 X", "Batch inputs X"), 23, BLUE, True)
    d.text(86, 174, "[[2, 1], [1, 1]]", 21, INK, mono=True)
    d.rect(528, 111, 203, 86, PURPLE_BG, PURPLE)
    d.text(548, 143, t("실제 정답 y", "Targets y"), 22, PURPLE, True)
    d.text(548, 174, "[[6], [4]]", 20, INK, mono=True)
    d.text(86, 224, "zero_grad(): " + t("이전 기울기를 비운다.", "clear old gradients."), 18, MUTED)
    d.line(280, 229, 280, 246, arrow=True)
    d.rect(51, 250, 554, 451, "#eef2f7", "#b6c2d3", 14)
    d.text(75, 279, t("이 구간에서는 파라미터 값을 바꾸지 않는다.",
                     "Parameters stay fixed throughout this region."), 18, INK, True)
    for y, label, code, fill, color in [
        (300, t("① 순전파: 예측값을 계산한다.", "1. Forward: compute predictions."),
         "pred = model(X)", BLUE_BG, BLUE),
        (435, t("② 손실: 예측을 정답과 비교한다.", "2. Loss: compare predictions and targets."),
         "L = mean(0.5 × (pred − y)²)", PURPLE_BG, PURPLE),
        (570, t("③ 역전파: 기울기를 계산한다.", "3. Backward: compute gradients."),
         "loss.backward()", ORANGE_BG, ORANGE),
    ]:
        d.rect(71, y, 514, 93, fill, color)
        d.text(92, y + 33, label, 20, color, True)
        d.text(92, y + 67, code, 20, INK, mono=True)
    d.line(325, 398, 325, 430, arrow=True)
    d.line(325, 533, 325, 565, arrow=True)
    d.path("M 630 201 V 482 H 591", PURPLE, arrow=True)
    d.text(645, 360, "y", 23, PURPLE, True)
    d.line(325, 706, 325, 743, arrow=True)
    d.rect(66, 750, 665, 102, TEAL_BG, TEAL)
    d.text(86, 786, t("④ 업데이트: 파라미터 값을 실제로 바꾼다.",
                     "4. Update: change the parameter values."), 22, TEAL, True)
    d.text(86, 822, "optimizer.step()", 22, INK, mono=True)
    d.line(398, 857, 398, 893, arrow=True)
    d.rect(66, 900, 665, 74, BLUE_BG, BLUE)
    d.text(86, 931, t("수정된 파라미터로 다음 배치를 처리한다.",
                     "Use the updated parameters for the next batch."), 21, BLUE, True)
    d.text(86, 957, t("다시 기울기 비우기 → 순전파 → 손실 → 역전파 → 업데이트", 
                     "Clear gradients → forward → loss → backward → update"), 17, MUTED)
    d.path("M 63 938 H 24 V 153 H 61", BLUE, 2, True)
    d.text(30, 1010, t("배치당 역전파 1회와 업데이트 1회를 수행하는 기본 설정이다.",
                      "Basic setup: one backward pass and one update per batch."), 18, MUTED)
    d.save("workflow")


def network(lang):
    t = lambda ko, en: pick(lang, ko, en)
    d = Diagram(lang, 930,
                t("두 입력 특징이 두 은닉 뉴런에 연결된다", "Both input features connect to both hidden neurons"),
                t("입력 x1과 x2는 각각 은닉 뉴런 두 개에 연결된다. 가중치는 w1=1,w2=2,w3=-1,w4=1, 편향 b1=0,b2=2다. 출력 뉴런은 h1,h2에 v1=2,v2=-1을 곱하고 b3=1을 더한다.",
                  "Each of x1 and x2 connects to both hidden neurons. Weights are 1,2,-1,1, hidden biases 0,2. The output uses v1=2, v2=-1, and b3=1."))
    d.text(30, 79, t("같은 층의 두 뉴런은 서로의 출력을 입력으로 받지 않는다.",
                     "The two hidden neurons do not feed into each other."), 18, MUTED)
    for x, name, meaning in [(150, "x1", t("작은 상자 수", "Small boxes")),
                             (470, "x2", t("큰 상자 수", "Large boxes"))]:
        d.rect(x, 121, 140, 83, BLUE_BG, BLUE)
        d.text(x + 70, 155, name, 26, BLUE, True, "middle", True)
        d.text(x + 70, 184, meaning, 17, MUTED, anchor="middle")
    # Draw edges first; badges then sit over their own line segments.
    d.line(195, 208, 195, 338, BLUE, arrow=True)
    d.line(565, 208, 565, 338, TEAL, arrow=True)
    d.line(265, 208, 515, 338, TEAL, arrow=True)
    d.line(495, 208, 245, 338, BLUE, arrow=True)
    d.badge(155, 281, "w1 = 1", BLUE, 114)
    d.badge(605, 281, "w4 = 1", TEAL, 114)
    d.badge(465, 310, "w3 = −1", TEAL, 123)
    d.badge(310, 313, "w2 = 2", BLUE, 114)
    for x, j, fill, color, bias in [(40, 1, BLUE_BG, BLUE, 0), (410, 2, TEAL_BG, TEAL, 2)]:
        d.rect(x, 348, 310, 160, fill, color)
        d.text(x + 18, 381, t(f"은닉 뉴런 {j}", f"Hidden neuron {j}"), 23, color, True)
        w_a, w_b = (1, 2) if j == 1 else (3, 4)
        d.text(x + 18, 420, f"z{j} = x1·w{w_a} + x2·w{w_b} + b{j}", 17, INK, mono=True)
        d.text(x + 18, 454, f"h{j} = ReLU(z{j})", 20, INK, mono=True)
        d.text(x + 18, 487, f"b{j} = {bias}", 20, color, True, mono=True)
    d.line(195, 513, 296, 622, BLUE, arrow=True)
    d.line(565, 513, 464, 622, TEAL, arrow=True)
    d.badge(230, 575, "v1 = 2", BLUE, 114)
    d.badge(530, 575, "v2 = −1", TEAL, 123)
    d.rect(159, 632, 442, 137, ORANGE_BG, ORANGE)
    d.text(180, 667, t("출력 뉴런", "Output neuron"), 23, ORANGE, True)
    d.text(180, 705, "ŷ = h1·v1 + h2·v2 + b3", 23, INK, mono=True)
    d.text(180, 744, "b3 = 1", 21, ORANGE, True, mono=True)
    d.text(380, 811, t("예측할 값: 포장 시간(분)", "Prediction: packing time in minutes"),
           22, ORANGE, True, "middle")
    d.rect(30, 847, 700, 57, "white")
    d.text(380, 882, t("가중치 6개 + 편향 3개 = 학습할 숫자 9개", 
                      "6 weights + 3 biases = 9 trainable numbers"), 21, INK, True, "middle")
    d.save("network")


def batch_matrix(lang):
    t = lambda ko, en: pick(lang, ko, en)
    d = Diagram(lang, 1110,
                t("배치 계산에서도 A와 B의 행을 유지한다", "Batch calculation preserves the A and B rows"),
                t("X의 A행 [2,1]과 B행 [1,1]에 같은 W_hidden과 편향 [0,2]를 적용한다. 은닉 출력은 A [4,1], B [3,2]이며 출력 가중치 [2,-1]과 편향 1을 적용하면 예측은 A 8, B 5다.",
                  "Rows A=[2,1] and B=[1,1] use the same hidden weights and bias [0,2]. Hidden outputs are [4,1] and [3,2]. Output weights [2,-1] and bias 1 give predictions 8 and 5."))
    d.text(30, 79, t("모든 샘플이 같은 가중치와 편향을 사용한다.",
                     "Every sample uses the same weights and biases."), 19, MUTED)
    d.text(30, 126, "1. Z = X @ W_hidden + b_hidden", 22, BLUE, True, mono=True)
    for x, name, shape in [(151, "X", "(2, 2)"), (391, "W_hidden", "(2, 2)"), (631, "X @ W_hidden", "(2, 2)")]:
        d.text(x, 169, name, 19, BLUE, True, "middle", True)
        d.text(x, 194, shape, 17, MUTED, anchor="middle", mono=True)
    d.matrix(89, 231, [[2, 1], [1, 1]], BLUE, BLUE_BG, rowlabels=["A", "B"], collabels=["x1", "x2"])
    d.matrix(329, 231, [[1, -1], [2, 1]], ORANGE, ORANGE_BG, collabels=["z1", "z2"])
    d.matrix(569, 231, [[4, -1], [3, 0]], TEAL, TEAL_BG, rowlabels=["A", "B"])
    d.text(264, 289, "@", 31, MUTED, True, "middle")
    d.text(504, 289, "=", 31, MUTED, True, "middle")
    d.rect(30, 360, 700, 125, "white")
    d.text(49, 392, t("각 행에 같은 편향 [0, 2]를 더한다.", "Add the same bias [0, 2] to each row."), 19, ORANGE, True)
    d.text(49, 429, "A: [4, −1] + [0, 2] = [4, 1]", 23, INK, mono=True)
    d.text(49, 465, "B: [3,  0] + [0, 2] = [3, 2]", 23, INK, mono=True)
    d.text(30, 537, "2. H = ReLU(Z)", 22, TEAL, True, mono=True)
    d.text(30, 570, t("Z의 네 값이 모두 양수이므로 H에도 그대로 남는다.",
                      "All four entries of Z are positive, so H keeps them."), 19, MUTED)
    d.text(30, 623, "3. pred = H @ W_out + b_out", 22, PURPLE, True, mono=True)
    for x, name, shape in [(151, "H", "(2, 2)"), (326, "W_out", "(2, 1)"), (638, "pred", "(2, 1)")]:
        d.text(x, 667, name, 21, PURPLE, True, "middle", True)
        d.text(x, 693, shape, 18, MUTED, anchor="middle", mono=True)
    d.matrix(89, 718, [[4, 1], [3, 2]], TEAL, TEAL_BG, rowlabels=["A", "B"])
    d.matrix(295, 718, [[2], [-1]], ORANGE, ORANGE_BG)
    d.matrix(607, 718, [[8], [5]], PURPLE, PURPLE_BG, rowlabels=["A", "B"])
    d.text(254, 775, "@", 28, MUTED, True, "middle")
    d.text(428, 774, "+ [1]", 26, INK, mono=True, anchor="middle")
    d.text(535, 775, "=", 29, MUTED, True, "middle")
    d.text(427, 809, "b_out", 17, MUTED, anchor="middle", mono=True)
    d.rect(30, 854, 700, 122, "white")
    d.text(49, 891, "A: 4×2 + 1×(−1) + 1 = 8", 24, INK, mono=True)
    d.text(49, 929, "B: 3×2 + 2×(−1) + 1 = 5", 24, INK, mono=True)
    d.text(49, 958, t("실제 정답: A는 6분, B는 4분", "Actual targets: A = 6 min, B = 4 min"), 18, MUTED)
    d.lines(30, 1021, [t("입력 (N, 2) → 은닉 출력 (N, 2) → 예측 (N, 1)",
                       "Input (N, 2) → hidden output (N, 2) → prediction (N, 1)"),
                       t("N은 샘플 수다. 샘플이 늘어도 가중치 행렬의 크기는 그대로다.",
                         "N is the sample count. More samples do not add weights.")], 19, MUTED, gap=34)
    d.save("batch-matrix")


def backward(lang):
    t = lambda ko, en: pick(lang, ko, en)
    d = Diagram(lang, 990,
                t("같은 w1을 쓴 A와 B의 기여를 더한다", "Add the contributions from the shared w1"),
                t("평균 손실 L의 w1 기울기는 A 기여 1×2×1×2=4와 B 기여 0.5×2×1×1=1의 합 5다. 예측값 기울기 1과 0.5에는 이미 배치 평균 계수 1/2이 들어 있다.",
                  "The w1 gradient is A's contribution 1×2×1×2=4 plus B's 0.5×2×1×1=1, giving 5. The prediction gradients already include the batch-mean factor 1/2."))
    d.text(30, 79, "L = (ℓA + ℓB) / 2", 22, INK, mono=True)
    d.text(30, 114, t("역전파 계산 순서: 손실에서 시작해 앞쪽으로 연결한다.",
                      "Backward order: begin at the loss and move to earlier values."), 18, MUTED)
    labels = ["L", "ŷ", "h1", "z1", "w1"]
    for i, label in enumerate(labels):
        x = 30 + i * 146
        d.rect(x, 145, 114, 54, ORANGE_BG, ORANGE, 9)
        d.text(x + 57, 180, label, 25, ORANGE, True, "middle", True)
        if i < 4:
            d.line(x + 119, 172, x + 140, 172, arrow=True)
    columns = [119, 293, 467, 641]
    descriptors = [
        t("예측값의 기울기", "Prediction gradient"),
        t("출력 가중치", "Output weight"),
        t("ReLU 변화율", "ReLU derivative"),
        t("입력 특징값", "Input feature"),
    ]
    for y, name, vals, equation, contribution, fill, color in [
        (239, "A", ["1", "2", "1", "2"], "(8 − 6) / 2 = 1", "4", BLUE_BG, BLUE),
        (489, "B", ["0.5", "2", "1", "1"], "(5 − 4) / 2 = 0.5", "1", TEAL_BG, TEAL),
    ]:
        d.rect(30, y, 700, 216, fill, color)
        d.text(50, y + 34, t(f"샘플 {name}", f"Sample {name}"), 23, color, True)
        d.text(710, y + 34, equation, 22, INK, anchor="end", mono=True)
        for i, (x, value, description) in enumerate(zip(columns, vals, descriptors)):
            d.text(x, y + 79, description, 17, MUTED, anchor="middle")
            d.text(x, y + 126, value, 29, color, True, "middle", True)
            if i < 3:
                d.text(x + 87, y + 124, "×", 24, MUTED, anchor="middle")
        d.line(49, y + 150, 711, y + 150, color=LINE, width=1)
        d.text(380, y + 188,
               t(f"평균 손실의 w1 기울기에 {contribution}만큼 기여한다.",
                 f"Contribution to the w1 gradient of mean loss: {contribution}"),
               21, color, True, "middle")
    d.path("M 27 350 H 12 V 763 H 246 V 786", BLUE, 2, True)
    d.path("M 733 599 H 747 V 763 H 514 V 786", TEAL, 2, True)
    d.rect(104, 797, 552, 79, ORANGE_BG, ORANGE)
    d.text(380, 847, "∂L/∂w1 = 4 + 1 = 5", 30, ORANGE, True, "middle", True)
    d.lines(30, 922, [t("1과 0.5에 평균 계수 ÷2가 이미 들어 있다.",
                       "The factor ÷2 is already included in 1 and 0.5."),
                       t("합계 5를 다시 2로 나누지 않는다.",
                         "Do not divide the total 5 by 2 again.")], 19, INK, gap=32)
    d.save("backward")


def accessible_plot(path, lang, title, description):
    """Add text alternatives to the Matplotlib SVG without changing the plot."""
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r'(width|height)="([\d.]+)pt"', r'\1="\2"', svg, count=2)
    start = svg.index("<svg ")
    close = svg.index(">", start)
    svg = (svg[:close] + f' role="img" aria-labelledby="chart-title chart-desc" xml:lang="{lang}"' +
           svg[close:close + 1] + f'\n<title id="chart-title">{escape(title)}</title>' +
           f'\n<desc id="chart-desc">{escape(description)}</desc>' + svg[close + 1:])
    path.write_text(svg, encoding="utf-8")


def chart_base(title, subtitle, height=620):
    fig = plt.figure(figsize=(760 / 72, height / 72), dpi=72, facecolor=BG)
    fig.text(.045, .938, title, fontsize=23, weight="bold", color=INK)
    fig.text(.045, .887, subtitle, fontsize=17, color=MUTED)
    ax = fig.add_axes([.12, .21, .83, .59], facecolor="white")
    ax.grid(alpha=.21, color=MUTED)
    ax.tick_params(labelsize=16, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LINE)
    ax.set_axisbelow(True)
    return fig, ax


def loss_slope(lang):
    t = lambda ko, en: pick(lang, ko, en)
    title = t("현재 손실값과 기울기는 서로 다른 값이다", "The current loss and its gradient are different")
    subtitle = t("w1만 바꾼다. 다른 파라미터와 데이터 A, B는 고정한다.",
                 "Only w1 changes; all other parameters and both samples stay fixed.")
    fig, ax = chart_base(title, subtitle, 640)
    w = np.linspace(0, 1.4, 281)
    ax.plot(w, 5 * (w - .5) ** 2, color=BLUE, lw=3,
            label=t("실제 손실 곡선", "Actual loss"))
    near = np.linspace(.77, 1.2, 81)
    ax.plot(near, 1.25 + 5 * (near - 1), color=ORANGE, lw=2.5,
            linestyle="--", label=t("현재 점의 접선", "Tangent at current point"))
    ax.scatter([.9, 1, 1.1], [.8, 1.25, 1.8], s=[54, 115, 54],
               color=[TEAL, ORANGE, TEAL], edgecolor="white", zorder=4)
    ax.annotate("(1.0, 1.25)", xy=(1, 1.25), xytext=(.55, 2.57),
                fontsize=19, weight="bold", color=ORANGE,
                arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.6})
    ax.annotate("(0.9, 0.80)", xy=(.9, .8), xytext=(.22, .80),
                fontsize=16, color=TEAL,
                arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.1})
    ax.annotate("(1.1, 1.80)", xy=(1.1, 1.8), xytext=(1.05, 3.22),
                fontsize=16, color=TEAL,
                arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.1})
    ax.set_xlim(0, 1.4)
    ax.set_ylim(-.12, 4.25)
    ax.set_xlabel(t("가중치 w1", "Weight w1"), fontsize=18, color=INK, labelpad=11)
    ax.set_ylabel(t("평균 손실 L", "Mean loss L"), fontsize=18, color=INK, labelpad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=16)
    fig.text(.045, .09, t("현재 값: L = 1.25    현재 기울기: ∂L/∂w1 = 5", 
                         "Current value: L = 1.25    Current gradient: ∂L/∂w1 = 5"),
             fontsize=18, color=INK, weight="bold")
    fig.text(.045, .042, t("접선은 가까운 변화만 근사한다: ΔL ≈ 5 × Δw1", 
                          "The tangent approximates nearby changes: ΔL ≈ 5 × Δw1"),
             fontsize=17, color=MUTED)
    path = OUT / f"loss-slope-{lang}.svg"
    fig.savefig(path, format="svg", facecolor=BG)
    plt.close(fig)
    accessible_plot(path, lang, title,
                    t("다른 조건을 고정하면 표시 구간의 손실은 L=5(w1-0.5)^2이다. w1=0.9에서 손실0.8, w1=1에서1.25, w1=1.1에서1.8이다. 현재 점의 접선 기울기는5다.",
                      "With other conditions fixed, L=5(w1-0.5)^2 on the plotted interval. Losses at w1=0.9,1,1.1 are 0.8,1.25,1.8. The current tangent has slope 5."))


def learning_rate(lang):
    t = lambda ko, en: pick(lang, ko, en)
    title = t("학습률만 바꾸면 손실의 경로가 달라진다", "Changing only the learning rate changes the path")
    subtitle = t("별도 실험: J(u) = (u − 3)², 시작 u = 0, 업데이트 3회", 
                 "Separate example: J(u) = (u − 3)², start u = 0, 3 updates")
    fig, ax = chart_base(title, subtitle, 650)
    trajectories = {}
    for lr, color in [(.05, BLUE), (.25, TEAL), (1.1, RED)]:
        u = 0.0
        values = [(u - 3) ** 2]
        for _ in range(3):
            grad = 2 * (u - 3)
            u -= lr * grad
            values.append((u - 3) ** 2)
        trajectories[lr] = values
        ax.plot(range(4), values, marker="o", ms=7, lw=2.7, color=color,
                label=f"lr = {lr}")
        offset = 0
        ax.annotate(f"{values[-1]:.3f}", (3, values[-1]),
                    xytext=(2.62, values[-1] + 1.3 + offset),
                    fontsize=17, color=color, weight="bold")
    assert np.allclose(trajectories[.05], [9, 7.29, 5.9049, 4.782969])
    assert np.allclose(trajectories[.25], [9, 2.25, .5625, .140625])
    assert np.allclose(trajectories[1.1], [9, 12.96, 18.6624, 26.873856])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(-.6, 30.5)
    ax.set_xlim(-.12, 3.2)
    ax.set_xlabel(t("업데이트를 마친 횟수", "Number of completed updates"),
                  fontsize=18, color=INK, labelpad=11)
    ax.set_ylabel(t("손실 J", "Loss J"), fontsize=18, color=INK, labelpad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=17)
    fig.text(.045, .091, t("0은 업데이트 전의 시작 상태다. 기록 4개, 업데이트 3회.",
                          "Step 0 is the starting state: 4 records, but only 3 updates."),
             fontsize=18, color=INK, weight="bold")
    fig.text(.045, .044, t("이 함수에서는 lr = 1.1일 때 최소점에서 점점 멀어진다.",
                          "For this function, lr = 1.1 moves farther from the minimum."),
             fontsize=17, color=MUTED)
    path = OUT / f"learning-rate-{lang}.svg"
    fig.savefig(path, format="svg", facecolor=BG)
    plt.close(fig)
    accessible_plot(path, lang, title,
                    t("손실 J=(u-3)^2에 대해 같은 시작점 u=0에서 세 번 업데이트한다. lr0.05의 손실은9,7.29,5.9049,4.782969, lr0.25는9,2.25,0.5625,0.140625, lr1.1은9,12.96,18.6624,26.873856이다.",
                      "Starting at u=0 for J=(u-3)^2: lr0.05 losses are9,7.29,5.9049,4.782969; lr0.25 losses9,2.25,0.5625,0.140625; lr1.1 losses9,12.96,18.6624,26.873856. Each series contains the initial state and three updates."))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, help="Korean TTF or OTF for chart labels")
    parser.add_argument("--bold-font", type=Path, help="Complete Korean bold TTF or OTF")
    args = parser.parse_args()
    if args.font:
        font_manager.fontManager.addfont(str(args.font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(args.font)).get_name()
        if args.bold_font:
            font_manager.fontManager.addfont(str(args.bold_font))
    else:
        plt.rcParams["font.family"] = ["Noto Sans KR", "Noto Sans CJK KR", "sans-serif"]
    plt.rcParams.update({"svg.fonttype": "path", "axes.unicode_minus": False})
    regular = args.font or Path(font_manager.findfont(font_manager.FontProperties(family="Noto Sans KR")))
    bold = args.bold_font or regular
    for mono, is_bold in [(False, False), (False, True), (True, False), (True, True)]:
        if mono:
            font_path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans Mono", weight="bold" if is_bold else "normal"))
        else:
            font_path = bold if is_bold else regular
        font = TTFont(font_path)
        font.getGlyphSet()  # Validate the complete glyph table, not only its metadata.
        OUTLINE_FONTS[(mono, is_bold)] = font
    for is_bold in [False, True]:
        font_path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if is_bold else "normal"))
        OUTLINE_FONTS[("fallback", is_bold)] = TTFont(font_path)
    OUT.mkdir(parents=True, exist_ok=True)
    for lang in ["ko", "en"]:
        for make in [workflow, network, batch_matrix, backward, loss_slope, learning_rate]:
            make(lang)
    for path in sorted(OUT.glob("*.svg")):
        ET.parse(path)  # Fail immediately if a generated SVG is not valid XML.
        print(path.name)


if __name__ == "__main__":
    main()
