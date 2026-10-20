"""Generate the article's exact, bilingual SVG diagrams (Python standard library only).

Run from any directory with Python 3. The SVG files are written next to this file.
All matrices use the fixed XOR calculation explained in the article.
"""

from html import escape
from pathlib import Path


OUT = Path(__file__).resolve().parent
BG = "#f6f8fc"
INK = "#16243b"
MUTED = "#52627b"
LINE = "#cbd5e1"
BLUE = "#1d4ed8"
BLUE_BG = "#dbeafe"
ORANGE = "#b45309"
ORANGE_BG = "#ffedd5"
TEAL = "#0f766e"
TEAL_BG = "#ccfbf1"
PURPLE = "#6d28d9"
PURPLE_BG = "#ede9fe"
RED = "#be123c"
RED_BG = "#ffe4e6"

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
W1 = [[1, -1], [-1, 1]]
Z1 = [[0, 0], [-1, 1], [1, -1], [0, 0]]
H = [[0, 0], [0, 1], [1, 0], [0, 0]]
OUTPUT_SUM = [[0, 0], [-1, 1], [-1, 1], [0, 0]]
SCORES = [[1, 0], [0, 1], [0, 1], [1, 0]]
PROBS = [["0.7311", "0.2689"], ["0.2689", "0.7311"],
         ["0.2689", "0.7311"], ["0.7311", "0.2689"]]
TARGETS = [0, 1, 1, 0]


def pick(lang, ko, en):
    return ko if lang == "ko" else en


class Diagram:
    def __init__(self, lang, height, title, desc):
        self.lang = lang
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="680" height="{height}" '
            f'viewBox="0 0 680 {height}" role="img" aria-labelledby="title desc" xml:lang="{lang}">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>',
            '<style>text{font-family:"Noto Sans KR","Noto Sans CJK KR",'
            '"Pretendard","Apple SD Gothic Neo","Malgun Gothic",sans-serif;'
            'font-variant-numeric:tabular-nums} .num{font-family:ui-monospace,SFMono-Regular,'
            'Consolas,"DejaVu Sans Mono",monospace}</style>',
            f'<rect width="680" height="{height}" rx="18" fill="{BG}"/>',
        ]
        self.text(28, 41, title, 24, bold=True)

    def raw(self, html):
        self.parts.append(html)

    def rect(self, x, y, w, h, fill="white", stroke=LINE, radius=12, width=1):
        self.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                 f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def text(self, x, y, value, size=18, color=INK, bold=False, anchor="start", mono=False):
        css_class = ' class="num"' if mono else ''
        self.raw(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
                 f'font-weight="{700 if bold else 400}" text-anchor="{anchor}" xml:space="preserve"'
                 f'{css_class}>{escape(str(value))}</text>')

    def lines(self, x, y, values, size=17, color=MUTED, gap=27, bold=False, anchor="start"):
        for i, value in enumerate(values):
            self.text(x, y + i * gap, value, size, color, bold, anchor)

    def line(self, x1, y1, x2, y2, color=LINE, width=1.5, dash=None, arrow=False):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
        arrow_attr = ' marker-end="url(#arrow)"' if arrow else ''
        self.raw(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                 f'stroke="{color}" stroke-width="{width}"'
                 f'{dash_attr}{arrow_attr}/>')

    def path(self, d, color=MUTED, width=2, arrow=False):
        arrow_attr = ' marker-end="url(#arrow)"' if arrow else ''
        self.raw(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'
                 f'{arrow_attr}/>')

    def matrix(self, x, y, data, cw=60, ch=42, highlights=None,
               rowlabels=None, collabels=None, fontsize=22):
        """Draw one cell per value; highlights map (row, column) to (fill, ink)."""
        highlights = highlights or {}
        for r, row in enumerate(data):
            for c, value in enumerate(row):
                fill, ink = highlights.get((r, c), ("white", INK))
                self.rect(x + c * cw, y + r * ch, cw, ch, fill, LINE, 0)
                self.text(x + (c + .5) * cw, y + (r + .5) * ch + fontsize * .35,
                          str(value).replace("-", "−"), fontsize, ink,
                          (r, c) in highlights, "middle", mono=True)
            if rowlabels:
                self.text(x - 12, y + (r + .5) * ch + 6, rowlabels[r], 15, MUTED, anchor="end")
        if collabels:
            for c, label in enumerate(collabels):
                self.text(x + (c + .5) * cw, y - 12, label, 16, MUTED, anchor="middle")

    def footer(self, values, y=None):
        y = self.height - 26 - 25 * (len(values) - 1) if y is None else y
        self.lines(28, y, values, 16, MUTED, gap=25)

    def save(self, stem):
        self.parts.append("</svg>")
        (OUT / f"{stem}-{self.lang}.svg").write_text("\n".join(self.parts) + "\n", encoding="utf-8")


def workflow(lang):
    t = lambda ko, en: pick(lang, ko, en)
    d = Diagram(lang, 980, t("네 샘플의 입력에서 예측과 손실까지", "From four samples to predictions and loss"),
                t("입력 X, 가중합 Z1, ReLU 출력 H, 출력 점수 S를 순서대로 계산한다. S에서 argmax로 예측을 고르거나 Softmax와 정답 y로 손실을 계산한다.",
                  "Compute X, Z1, H, and S in order. Scores branch into argmax predictions or Softmax probabilities and target-based loss."))
    d.lines(28, 74, [t("모든 샘플에 같은 가중치와 편향을 적용한다.",
                       "Every sample uses the same weights and biases.")], 17)
    stages = [
        (110, "X", t("원본 입력 · 샘플 4개 × 특징 2개", "Input · 4 samples × 2 features"), BLUE_BG, BLUE),
        (238, "Z1", t("은닉층의 가중합 + 편향", "Hidden weighted sums + biases"), ORANGE_BG, ORANGE),
        (366, "H", t("ReLU를 적용한 은닉층 출력", "Hidden outputs after ReLU"), TEAL_BG, TEAL),
        (494, "S", t("‘같다’와 ‘다르다’의 출력 점수", "Output scores for same / different"), PURPLE_BG, PURPLE),
    ]
    for y, name, meaning, fill, color in stages:
        d.rect(52, y, 576, 77, fill, color)
        d.text(75, y + 31, name, 26, color, True)
        d.text(593, y + 31, "(4, 2)", 20, color, True, "end", mono=True)
        d.text(75, y + 59, meaning, 17, INK)
    for y, label in [(203, "X @ W1 + b1"), (331, "maximum(0, Z1)"), (459, "H @ W2 + b2")]:
        d.line(340, y - 8, 340, y + 23, MUTED, arrow=True)
        d.text(363, y + 12, label, 17, MUTED, mono=True)
    d.path("M 340 574 V 603 H 183 V 622", arrow=True)
    d.path("M 340 603 H 497 V 622", arrow=True)
    d.rect(28, 638, 310, 96, BLUE_BG, BLUE)
    d.text(49, 668, "argmax(axis=1)", 19, BLUE, True, mono=True)
    d.text(49, 698, "pred = [0, 1, 1, 0]", 18, INK, mono=True)
    d.text(49, 721, t("샘플마다 클래스 하나", "One class per sample"), 16, MUTED)
    d.rect(363, 638, 289, 96, PURPLE_BG, PURPLE)
    d.text(385, 668, "Softmax", 21, PURPLE, True)
    d.text(385, 698, t("클래스별 확률 (4, 2)", "Class probabilities (4, 2)"), 17)
    d.text(385, 722, t("각 행의 확률 합 = 1", "Each row sums to 1"), 16, MUTED)
    d.line(507, 738, 507, 759, MUTED, arrow=True)
    d.rect(363, 773, 289, 76)
    d.text(385, 800, t("y로 정답 확률 4개 선택", "Use y to select 4 probabilities"), 16, PURPLE, True)
    d.text(385, 830, "−log → mean → 0.3133", 17, INK, mono=True)
    d.lines(42, 791, [t("예측은 최고 점수의 위치를 고른다.", "Prediction selects the top score."),
                     t("손실은 실제 정답의 확률을 평가한다.", "Loss evaluates the true class.")], 15, gap=28)
    d.rect(28, 880, 624, 70, "#fff7ed", "#fed7aa")
    d.lines(45, 906, [t("이 예제의 가중치는 사람이 미리 지정했다.", "The weights in this example were set by hand."),
                     t("가중치를 수정하지 않으므로 학습을 수행한 것은 아니다.", "These calculations do not update weights or train a model.")], 16, gap=25)
    d.save("workflow")


def matmul(lang):
    t = lambda ko, en: pick(lang, ko, en)
    d = Diagram(lang, 620, t("한 행과 한 열을 계산하면 결과 한 칸이 된다", "One row and one column produce one cell"),
                t("X의 행 번호 1과 W1의 열 번호 1을 선택한다. 같은 순서의 숫자를 곱해 더하면 Z1의 행 1, 열 1 값인 1이 된다.",
                  "Pair row 1 of X with column 1 of W1. Multiply matching entries and add them to get Z1[1,1] = 1."))
    d.text(28, 75, t("행·열 번호는 0부터 시작한다. b1 = [0, 0]이다.",
                     "Row and column indices start at 0. b1 = [0, 0]."), 17, MUTED)
    for x, label, shape, color in [(96, "X", "(4, 2)", BLUE), (336, "W1", "(2, 2)", ORANGE), (568, "Z1", "(4, 2)", TEAL)]:
        d.text(x, 122, label, 24, color, True, "middle")
        d.text(x, 149, shape, 17, MUTED, anchor="middle", mono=True)
    d.matrix(40, 201, X, 56, 44, {(1, c): (BLUE_BG, BLUE) for c in range(2)},
             rowlabels=["0", "1", "2", "3"], collabels=["x1", "x2"])
    d.matrix(280, 201, W1, 56, 44, {(r, 1): (ORANGE_BG, ORANGE) for r in range(2)},
             rowlabels=["x1", "x2"], collabels=["j=0", "j=1"])
    d.matrix(512, 201, Z1, 56, 44, {(1, 1): (TEAL_BG, TEAL)},
             rowlabels=["0", "1", "2", "3"], collabels=["j=0", "j=1"])
    d.text(207, 273, "@", 32, MUTED, True, "middle")
    d.text(451, 273, "=", 31, MUTED, True, "middle")
    d.text(96, 402, t("샘플 한 개", "One sample"), 17, BLUE, True, "middle")
    d.text(337, 324, t("뉴런 한 개의 가중치", "One neuron's weights"), 16, ORANGE, True, "middle")
    d.text(568, 402, t("결과 한 칸", "One result"), 17, TEAL, True, "middle")
    d.rect(28, 434, 624, 101, "white", LINE)
    d.text(46, 465, "Z1[1, 1] = 0 × (−1) + 1 × 1 + 0", 20, INK, mono=True)
    d.text(46, 506, "             = 0 + 1 + 0 = 1", 20, TEAL, True, mono=True)
    d.footer([t("왼쪽의 특징 2개와 오른쪽의 가중치 2개가 순서대로 짝을 이룬다.",
                "Two features pair with two weights in the same order."),
              t("다른 칸도 같은 방법으로 계산한다. (4, 2) @ (2, 2) → (4, 2)",
                "Repeat for each cell: (4, 2) @ (2, 2) → (4, 2).")])
    d.save("matmul")


def relu(lang):
    t = lambda ko, en: pick(lang, ko, en)
    d = Diagram(lang, 528, t("ReLU는 각 칸의 음수만 0으로 바꾼다", "ReLU changes each negative entry to zero"),
                t("Z1의 음수 두 칸은 H에서 0이 된다. 0과 양수는 그대로이며 행 4개와 열 2개가 유지된다.",
                  "The two negative entries of Z1 become zero in H. Zero and positive entries stay unchanged. Both shapes remain (4,2)."))
    d.text(28, 76, "H = np.maximum(0, Z1)", 21, MUTED, mono=True)
    d.text(171, 132, "Z1  (4, 2)", 23, ORANGE, True, "middle", mono=True)
    d.text(518, 132, "H  (4, 2)", 23, TEAL, True, "middle", mono=True)
    negatives = {(1, 0): (RED_BG, RED), (2, 1): (RED_BG, RED)}
    zeros = {(1, 0): (TEAL_BG, TEAL), (2, 1): (TEAL_BG, TEAL)}
    d.matrix(104, 186, Z1, 68, 48, negatives, rowlabels=["0", "1", "2", "3"], collabels=["z1", "z2"])
    d.matrix(450, 186, H, 68, 48, zeros, rowlabels=["0", "1", "2", "3"], collabels=["h1", "h2"])
    for r in range(4):
        d.line(250, 210 + r * 48, 401, 210 + r * 48, MUTED, width=1.5, arrow=True)
    d.rect(28, 412, 624, 78)
    d.lines(47, 442, [t("−1 → 0     0 → 0     1 → 1", "−1 → 0     0 → 0     1 → 1"),
                     t("숫자 8개를 각각 처리한다. 샘플을 합치거나 지우지 않는다.",
                       "All 8 entries are processed. No samples are merged or removed.")], 17, gap=28)
    d.save("relu")


def xor_spaces(lang):
    t = lambda ko, en: pick(lang, ko, en)
    d = Diagram(lang, 720, t("같은 샘플을 서로 다른 좌표로 표시한다", "The same samples in two coordinate systems"),
                t("원본 X에는 네 점이 있다. 은닉층 H에는 원본의 샘플 0과 3이 같은 좌표 (0,0)에 겹친다. 원본의 두 경계는 은닉좌표에서 직선 하나로 표현된다.",
                  "X has four input points. In H, samples 0 and 3 share (0,0). The two boundaries in input coordinates become one line in hidden coordinates."))
    d.lines(28, 77, [t("원본 X를 덮어쓰지 않고, H에 계산한 새 값을 표시한다.",
                       "X stays unchanged; H contains newly calculated values.")], 17)
    panels = [(30, "X", "x1", "x2"), (353, "H", "h1", "h2")]
    def coords(left, x, y):
        return left + 41 + x * 185, 382 - y * 185
    for left, name, xlabel, ylabel in panels:
        d.rect(left, 110, 297, 328, "white", LINE)
        d.text(left + 148, 142, name, 23, BLUE if name == "X" else TEAL, True, "middle")
        for v in [0, .5, 1]:
            x, _ = coords(left, v, 0)
            _, y = coords(left, 0, v)
            d.line(x, 166, x, 391, "#e2e8f0", width=1)
            d.line(left + 27, y, left + 262, y, "#e2e8f0", width=1)
            if v in [0, 1]:
                d.text(x, 408, str(v), 15, MUTED, anchor="middle")
                d.text(left + 24, y + 5, str(v), 15, MUTED, anchor="end")
        d.line(left + 27, 382, left + 263, 382, MUTED, arrow=True)
        d.line(left + 41, 395, left + 41, 164, MUTED, arrow=True)
        d.text(left + 268, 409, xlabel, 18, INK, True, "middle")
        d.text(left + 40, 162, ylabel, 18, INK, True, "middle")
    # The displayed boundaries are exact consequences of |x1-x2| = 0.5.
    for start, end in [((-.07, .43), (.62, 1.12)), ((.43, -.07), (1.12, .62))]:
        x1, y1 = coords(30, *start)
        x2, y2 = coords(30, *end)
        d.line(x1, y1, x2, y2, PURPLE, width=2, dash="6 5")
    x1, y1 = coords(353, -.07, .57)
    x2, y2 = coords(353, .57, -.07)
    d.line(x1, y1, x2, y2, PURPLE, width=2, dash="6 5")

    def point(left, x, y, cls, label, dx, dy, overlap=False):
        px, py = coords(left, x, y)
        if cls == 0:
            if overlap:
                d.raw(f'<circle cx="{px}" cy="{py}" r="12" fill="white" stroke="{BLUE}" stroke-width="2"/>')
            d.raw(f'<circle cx="{px}" cy="{py}" r="7" fill="{BLUE}" stroke="white" stroke-width="1.5"/>')
        else:
            d.line(px - 7, py - 7, px + 7, py + 7, ORANGE, 3)
            d.line(px + 7, py - 7, px - 7, py + 7, ORANGE, 3)
        d.text(px + dx, py + dy, label, 17, BLUE if cls == 0 else ORANGE, True)
    point(30, 0, 0, 0, "0", 12, -12)
    point(30, 0, 1, 1, "1", 13, -8)
    point(30, 1, 0, 1, "2", 12, -12)
    point(30, 1, 1, 0, "3", 12, -8)
    point(353, 0, 0, 0, "0, 3", 16, -15, True)
    point(353, 0, 1, 1, "1", 13, -8)
    point(353, 1, 0, 1, "2", 12, -12)
    d.text(178, 468, "x2 = x1 ± 0.5", 19, PURPLE, True, "middle", mono=True)
    d.text(501, 468, "h1 + h2 = 0.5", 19, PURPLE, True, "middle", mono=True)
    d.rect(28, 493, 624, 87, "white", LINE)
    d.lines(47, 522, [t("점 옆의 숫자는 샘플 번호다. 0과 3은 H의 (0, 0)에 겹친다.",
                       "Point labels are sample indices. Samples 0 and 3 overlap in H."),
                     t("H의 점이 세 곳에 보이지만 샘플은 여전히 네 개다.",
                       "There are three visible locations, but still four samples.")], 16, gap=29)
    d.raw(f'<circle cx="43" cy="614" r="7" fill="{BLUE}"/>')
    d.text(61, 620, t("클래스 0: 같다", "Class 0: same"), 17)
    d.line(270, 607, 284, 621, ORANGE, 3)
    d.line(284, 607, 270, 621, ORANGE, 3)
    d.text(295, 620, t("클래스 1: 다르다", "Class 1: different"), 17)
    d.footer([t("점선은 두 클래스의 점수가 같은 경계다. 색과 정답은 바뀌지 않는다.",
                "Dashed lines mark equal class scores. Labels and classes stay the same."),
              t("0.5는 미리 지정한 이 예제의 가중치·편향에서 나오는 기준이다.",
                "The threshold 0.5 follows from this example's fixed weights and biases.")])
    d.save("xor-spaces")


def output_bias(lang):
    t = lambda ko, en: pick(lang, ko, en)
    d = Diagram(lang, 695, t("편향 두 개를 모든 샘플 행에 똑같이 더한다", "Add the same two biases to every sample row"),
                t("H @ W2의 각 행에 b2=[1,0]을 더한다. 첫 열에는 1, 둘째 열에는 0을 더한 결과가 S다.",
                  "Add b2=[1,0] to each row of H @ W2. The first column receives 1 and the second receives 0, yielding S."))
    d.text(28, 76, "S = H @ W2 + b2", 21, MUTED, mono=True)
    d.text(154, 127, "H @ W2", 22, ORANGE, True, "middle", mono=True)
    d.text(496, 127, "S", 24, TEAL, True, "middle")
    d.text(154, 157, "(4, 2)", 17, MUTED, anchor="middle", mono=True)
    d.text(496, 157, "(4, 2)", 17, MUTED, anchor="middle", mono=True)
    d.matrix(91, 196, OUTPUT_SUM, 63, 46, rowlabels=["0", "1", "2", "3"])
    hi = {(r, 0): (TEAL_BG, TEAL) for r in range(4)}
    d.matrix(432, 196, SCORES, 63, 46, hi, rowlabels=["0", "1", "2", "3"],
             collabels=[t("같다", "same"), t("다르다", "different")])
    for r in range(4):
        d.text(312, 226 + r * 46, "+ [1, 0]", 19, BLUE, True, "middle", mono=True)
    d.text(154, 414, t("가중합", "Weighted sums"), 17, MUTED, anchor="middle")
    d.text(496, 414, t("가중합 + 편향", "Sums + biases"), 17, MUTED, anchor="middle")
    d.rect(28, 443, 624, 134, "white", LINE)
    d.text(48, 475, "b2.shape = (2,)", 19, BLUE, True, mono=True)
    d.text(48, 510, t("첫 열: −1 + 1 = 0     둘째 열: 1 + 0 = 1", "First column: −1 + 1 = 0    Second: 1 + 0 = 1"), 18)
    d.text(48, 548, "(4, 2) + (2,) → (4, 2)", 21, TEAL, True, mono=True)
    d.footer([t("이렇게 크기가 다른 배열의 대응 위치를 맞추는 규칙이 브로드캐스팅이다.",
                "Broadcasting determines how arrays with different shapes align."),
              t("여기서는 b2의 같은 두 값을 모든 행에 적용한다.",
                "Here, the same two b2 values apply to every row."),
              t("b2를 실제로 네 번 복사한 배열을 미리 만들 필요는 없다.",
                "No separate array with four copies of b2 needs to be created.")])
    d.save("output-bias")


def softmax(lang):
    t = lambda ko, en: pick(lang, ko, en)
    d = Diagram(lang, 1120, t("Softmax: 한 행의 점수를 확률로 계산한다", "Softmax: turn one row of scores into probabilities"),
                t("행 번호 1의 점수 [0,1]에서 최댓값 1을 빼면 [-1,0]이다. exp를 적용한 [0.3679,1]의 각 값을 행 합 1.3679로 나누면 [0.2689,0.7311]이 된다.",
                  "For row 1, subtract 1 from scores [0,1], exponentiate [-1,0], then divide [0.3679,1] by 1.3679 to obtain [0.2689,0.7311]."))
    d.text(28, 77, t("둘째 샘플, S의 행 번호 1을 끝까지 계산한다.",
                     "Follow the second sample: row index 1 of S."), 17, MUTED)
    blocks = [
        (105, 130, "1", t("행의 최댓값을 구한다", "Find the row maximum"), "[0, 1]  →  max = 1", "S.max(axis=1, keepdims=True)", "(4, 2) → (4, 1)", BLUE_BG, BLUE),
        (260, 130, "2", t("그 행의 두 점수에서 1을 뺀다", "Subtract 1 from both scores"), "[0 − 1, 1 − 1] = [−1, 0]", "shifted = S − row_max", "(4, 2) − (4, 1) → (4, 2)", ORANGE_BG, ORANGE),
        (415, 130, "3", t("각 숫자에 exp를 적용한다", "Apply exp to each entry"), "[exp(−1), exp(0)] ≈ [0.3679, 1]", "exp_scores = np.exp(shifted)", "(4, 2) → (4, 2)", TEAL_BG, TEAL),
        (570, 145, "4", t("그 행의 합을 구하고 각각 나눈다", "Sum the row, then divide each entry"), "0.3679 + 1 = 1.3679", "[0.3679 / 1.3679, 1 / 1.3679]", "(4, 2) / (4, 1) → (4, 2)", PURPLE_BG, PURPLE),
    ]
    for y, h, n, label, calculation, code, shape, fill, color in blocks:
        d.rect(28, y, 624, h, "white", LINE)
        d.rect(44, y + 17, 30, 30, fill, color, 8)
        d.text(59, y + 39, n, 18, color, True, "middle")
        d.text(90, y + 39, label, 20, color, True)
        d.text(48, y + 76, calculation, 19, INK, mono=True)
        d.text(48, y + 103, code, 16, MUTED, mono=True)
        d.text(627, y + h - 9, shape, 16, color, True, "end", mono=True)
    for y in [242, 397, 552]:
        d.line(340, y, 340, y + 13, MUTED, arrow=True)
    d.line(340, 721, 340, 742, MUTED, arrow=True)
    d.text(28, 777, t("네 샘플의 최종 확률", "Final probabilities for all four samples"), 21, PURPLE, True)
    d.text(627, 777, "(4, 2)", 18, MUTED, anchor="end", mono=True)
    hi = {(1, c): (PURPLE_BG, PURPLE) for c in range(2)}
    d.matrix(186, 823, PROBS, 124, 43, hi, rowlabels=["0", "1", "2", "3"],
             collabels=[t("같다", "same"), t("다르다", "different")], fontsize=23)
    d.text(470, 891, "←", 26, PURPLE, True)
    d.text(510, 890, t("이 행을 계산했다", "This row"), 16, PURPLE, True)
    d.footer([t("각 행의 확률 합은 1이다. 다른 샘플의 점수를 함께 더하지 않는다.",
                "Each row sums to 1. Scores from different samples are not combined."),
              t("표시는 소수 넷째 자리까지 반올림했으며, 계산은 원래 정밀도로 수행한다.",
                "Values shown are rounded to four decimals; calculations retain precision.")])
    d.save("softmax")


def indexing_loss(lang):
    t = lambda ko, en: pick(lang, ko, en)
    d = Diagram(lang, 800, t("정답 y가 가리키는 칸을 행마다 하나씩 고른다", "Use y to select one true-class cell from each row"),
                t("행 번호 [0,1,2,3]와 정답 [0,1,1,0]을 짝지어 probabilities의 네 칸을 선택한다. 각 정답 확률은 0.7311, 각 손실은 약 0.3133이다.",
                  "Pair row indices [0,1,2,3] with targets [0,1,1,0]. Each selected probability is about 0.7311 and each loss about 0.3133."))
    d.text(28, 79, "correct_p = probabilities[row_indices, y]", 20, MUTED, mono=True)
    d.text(180, 134, "probabilities (4, 2)", 19, PURPLE, True, "middle", mono=True)
    d.text(437, 134, "y", 23, BLUE, True, "middle")
    d.text(583, 134, t("선택 위치", "Selected cell"), 17, TEAL, True, "middle")
    hi = {(r, c): (TEAL_BG, TEAL) for r, c in enumerate(TARGETS)}
    d.matrix(74, 193, PROBS, 112, 49, hi, rowlabels=["0", "1", "2", "3"],
             collabels=[t("열 0: 같다", "col 0: same"), t("열 1: 다르다", "col 1: different")], fontsize=22)
    d.matrix(415, 193, [[x] for x in TARGETS], 45, 49,
             {(r, 0): (BLUE_BG, BLUE) for r in range(4)}, fontsize=22)
    for r, c in enumerate(TARGETS):
        d.text(583, 225 + r * 49, f"({r}, {c})", 21, TEAL, True, "middle", mono=True)
    d.text(30, 174, t("행", "row"), 15, MUTED)
    d.rect(28, 425, 624, 112, "white", LINE)
    d.text(46, 456, "row_indices:  [0, 1, 2, 3]", 20, BLUE, True, mono=True)
    d.text(46, 489, "y:            [0, 1, 1, 0]", 20, TEAL, True, mono=True)
    d.text(46, 518, t("같은 위치의 행 번호와 열 번호를 한 쌍으로 사용한다.",
                     "Match the row and column indices at the same position."), 16, MUTED)
    d.line(340, 545, 340, 565, MUTED, arrow=True)
    d.rect(28, 580, 624, 129, PURPLE_BG, "#c4b5fd")
    d.text(46, 611, "correct_p ≈ [0.7311, 0.7311, 0.7311, 0.7311]", 17, PURPLE, True, mono=True)
    d.text(46, 649, "−log → [0.3133, 0.3133, 0.3133, 0.3133]", 18, INK, mono=True)
    d.text(46, 686, t("네 손실의 평균 ≈ 0.3133", "Mean of the four losses ≈ 0.3133"), 21, PURPLE, True)
    d.footer([t("정답 평가에는 y를 쓴다. pred는 모델이 고른 클래스다.",
                "Use y to evaluate the true class. pred is the predicted class."),
              t("모델이 틀리게 예측한 행에서는 두 값이 서로 다른 칸을 가리킨다.",
                "For incorrect predictions, the two indices select different cells.")])
    d.save("indexing-loss")


def layer_shapes(lang):
    t = lambda ko, en: pick(lang, ko, en)
    d = Diagram(lang, 970, t("층을 지나도 샘플 수는 그대로 유지된다", "The sample count stays unchanged across layers"),
                t("샘플 2개, 입력 특징 4개에서 은닉 뉴런 5개, 6개와 출력 뉴런 3개를 통과한다. 가중치는 (4,5), (5,6), (6,3)이며 편향 포함 파라미터는 82개다.",
                  "Two samples with four features pass through hidden layers of 5 and 6 neurons and an output layer of 3 neurons. The model has 82 parameters including biases."))
    d.text(28, 78, t("앞의 XOR와 별개로, 층의 크기를 확인하는 예제다.",
                     "This separate example checks layer sizes, apart from XOR."), 17, MUTED)
    stages = [
        (114, "1", t("첫 번째 은닉층: 뉴런 5개", "Hidden layer 1: 5 neurons"),
         "X_demo       W1_demo       H1_demo", "(2, 4)   @   (4, 5)   →   (2, 5)", "4 × 5 + 5 = 25", BLUE_BG, BLUE),
        (295, "2", t("두 번째 은닉층: 뉴런 6개", "Hidden layer 2: 6 neurons"),
         "H1_demo      W2_demo       H2_demo", "(2, 5)   @   (5, 6)   →   (2, 6)", "5 × 6 + 6 = 36", TEAL_BG, TEAL),
        (476, "3", t("출력층: 뉴런 3개", "Output layer: 3 neurons"),
         "H2_demo      W3_demo       S_demo", "(2, 6)   @   (6, 3)   →   (2, 3)", "6 × 3 + 3 = 21", PURPLE_BG, PURPLE),
    ]
    for y, n, title, names, shape, count, fill, color in stages:
        d.rect(28, y, 624, 152, "white", LINE)
        d.rect(44, y + 15, 31, 31, fill, color, 8)
        d.text(59, y + 38, n, 18, color, True, "middle")
        d.text(89, y + 39, title, 21, color, True)
        inputs = names.split()
        sizes = {"1": ["(2, 4)", "(4, 5)", "(2, 5)"],
                 "2": ["(2, 5)", "(5, 6)", "(2, 6)"],
                 "3": ["(2, 6)", "(6, 3)", "(2, 3)"]}[n]
        for center, name, dims in zip([132, 337, 548], inputs, sizes):
            d.text(center, y + 77, name, 18, MUTED, anchor="middle", mono=True)
            d.text(center, y + 110, dims, 21, color, True, "middle", mono=True)
        d.text(230, y + 110, "@", 21, color, True, "middle", mono=True)
        d.text(441, y + 110, "→", 21, color, True, "middle", mono=True)
        d.text(49, y + 138, t("가중치 + 편향: ", "Weights + biases: ") + count, 16, MUTED)
    for y in [272, 453]:
        d.line(340, y, 340, y + 15, MUTED, arrow=True)
    d.rect(28, 658, 624, 124, ORANGE_BG, "#fed7aa")
    d.text(47, 691, t("이전 결과의 열 k가 다음 입력의 특징 k가 된다.", "Column k becomes feature k for the next layer."), 18, ORANGE, True)
    d.text(47, 728, "H1_demo[i, 2] × W2_demo[2, j]", 18, INK, mono=True)
    d.text(47, 759, t("이 곱은 H2_demo[i, j]를 만드는 합의 한 항이다.",
                     "This product is one term used to compute H2_demo[i, j]."), 16, MUTED)
    d.text(28, 827, t("편향까지 세면 전체 파라미터는 82개다.",
                     "Including biases, the model has 82 parameters."), 21, INK, True)
    d.text(28, 866, "25 + 36 + 21 = 82", 24, TEAL, True, mono=True)
    d.footer([t("행렬곱 뒤에는 각 층의 편향을 더한다. 두 은닉층에는 ReLU도 적용한다.",
                "Add a bias after each product. Both hidden layers also apply ReLU."),
              t("샘플은 계속 2개다. 각 단계의 열 수는 그 층의 뉴런 수다.",
                "There are always 2 samples. Each output width is that layer's neuron count.")])
    d.save("layer-shapes")


def row_broadcast(lang):
    t = lambda ko, en: pick(lang, ko, en)
    d = Diagram(lang, 906, t("keepdims=True는 행마다 기준값 한 칸을 남긴다", "keepdims=True retains one reference value per row"),
                t("별도 점수 예제 scores3의 크기는 (2,3)이다. axis=1로 최댓값을 구하고 keepdims=True로 (2,1)을 유지한다. 각 행은 자신의 최댓값 5.0 또는 4.2를 뺀다.",
                  "The separate scores3 example has shape (2,3). Reducing with axis=1 and keepdims=True returns (2,1). Each row subtracts its own maximum, 5.0 or 4.2."))
    d.lines(28, 77, [t("행마다 다른 최댓값을 갖는 별도 점수 예제로 방향을 확인한다.",
                       "Use a separate score example whose row maxima are different.")], 17)
    d.text(181, 134, "scores3 (2, 3)", 22, BLUE, True, "middle", mono=True)
    d.text(518, 134, "row_max3 (2, 1)", 21, ORANGE, True, "middle", mono=True)
    colors = {(0, c): (BLUE_BG, BLUE) for c in range(3)}
    colors.update({(1, c): (TEAL_BG, TEAL) for c in range(3)})
    d.matrix(58, 190, [["2.1", "0.4", "5.0"], ["1.3", "4.2", "0.7"]], 82, 52,
             colors, rowlabels=["0", "1"], collabels=["0", "1", "2"], fontsize=23)
    d.matrix(477, 190, [["5.0"], ["4.2"]], 82, 52,
             {(0, 0): (BLUE_BG, BLUE), (1, 0): (TEAL_BG, TEAL)}, fontsize=23)
    for row in range(2):
        d.line(320, 216 + row * 52, 458, 216 + row * 52, MUTED, arrow=True)
    d.text(389, 174, "max", 17, ORANGE, True, "middle")
    d.text(28, 340, "row_max3 = scores3.max(axis=1, keepdims=True)", 19, MUTED, mono=True)
    d.rect(28, 367, 624, 84, "white", LINE)
    d.lines(46, 396, [t("axis=1: 각 행 안의 점수 3개를 최댓값 하나로 줄인다.",
                       "axis=1 reduces the 3 scores within each row to one maximum."),
                     t("keepdims=True: 줄어든 두 번째 축을 크기 1로 유지한다.",
                       "keepdims=True retains the reduced second axis with size 1.")], 17, gap=28)
    d.text(28, 497, t("각 행은 자신의 최댓값을 뺀다", "Each row subtracts its own maximum"), 22, INK, True)
    for y, label, formula, result, fill, color in [
        (520, t("행 0", "Row 0"), "[2.1 − 5.0, 0.4 − 5.0, 5.0 − 5.0]", "[−2.9, −4.6, 0.0]", BLUE_BG, BLUE),
        (653, t("행 1", "Row 1"), "[1.3 − 4.2, 4.2 − 4.2, 0.7 − 4.2]", "[−2.9, 0.0, −3.5]", TEAL_BG, TEAL),
    ]:
        d.rect(28, y, 624, 111, fill, color)
        d.text(46, y + 30, label, 19, color, True)
        d.text(46, y + 65, formula, 20, INK, mono=True)
        d.text(46, y + 95, "= " + result, 20, color, True, mono=True)
    d.text(28, 813, "(2, 3) − (2, 1) → (2, 3)", 23, PURPLE, True, mono=True)
    d.footer([t("크기 (2, 1)의 한 칸은 같은 행의 세 열에 적용된다.",
                "The single value in each (2,1) row applies across that row's 3 columns."),
              t("행마다 계산한 기준값을 다른 샘플의 행과 섞지 않는다.",
                "Reference values are not mixed between samples.")])
    d.save("row-broadcast")


if __name__ == "__main__":
    for language in ["ko", "en"]:
        for make in [workflow, matmul, relu, xor_spaces, output_bias, softmax, indexing_loss, layer_shapes, row_broadcast]:
            make(language)
    print("Generated 18 SVG figures.")
