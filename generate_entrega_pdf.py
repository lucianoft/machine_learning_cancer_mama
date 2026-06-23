"""Gera ENTREGA_FASE_1.pdf com layout profissional."""
from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from fpdf import FPDF
from PIL import Image

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Tech_Challenge_Breast_Cancer.ipynb"
RELATORIO = ROOT / "RELATORIO_TECNICO.md"
OUTPUT = ROOT / "ENTREGA_FASE_1.pdf"
ASSETS = ROOT / ".pdf_assets"
REPO = "https://github.com/lucianoft/machine_learning_cancer_mama"

# Paleta
PRIMARY = (26, 54, 93)       # azul escuro
ACCENT = (0, 102, 153)       # azul médio
LIGHT_BG = (245, 247, 250)   # fundo suave
BORDER = (210, 218, 226)
TEXT = (33, 37, 41)
MUTED = (108, 117, 125)

IMAGE_CAPTIONS = [
    "Figura 1 — Distribuição da variável alvo (benigno vs maligno)",
    "Figura 2 — Histogramas das variáveis numéricas",
    "Figura 3 — Pairplot das medidas mean_* por diagnóstico",
    "Figura 4 — Matriz de correlação entre features",
    "Figura 5 — Matriz de confusão (Regressão Logística)",
    "Figura 6 — Comparação de modelos por F1-Score",
]


class EntregaPDF(FPDF):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.section_title = "Tech Challenge — Câncer de Mama"

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_fill_color(*PRIMARY)
        self.rect(0, 0, self.w, 14, style="F")
        self.set_y(4)
        self.set_font("Body", "B", 9)
        self.set_text_color(255, 255, 255)
        self.set_x(self.l_margin)
        self.cell(0, 6, self.section_title, align="L")
        self.set_x(self.w - self.r_margin - 40)
        self.cell(40, 6, "Fase 1 — IADT", align="R")
        self.ln(12)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_draw_color(*BORDER)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_font("Body", size=8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, f"Página {self.page_no()}/{{nb}}", align="C")

    def set_section(self, title: str) -> None:
        self.section_title = title


def load_fonts(pdf: EntregaPDF) -> None:
    fonts = Path(__file__).resolve().parent
    win = Path(r"C:\Windows\Fonts")
    if (win / "arial.ttf").exists():
        pdf.add_font("Body", "", str(win / "arial.ttf"))
        pdf.add_font("Body", "B", str(win / "arialbd.ttf"))
        pdf.add_font("Body", "I", str(win / "ariali.ttf"))
        pdf.add_font("Mono", "", str(win / "consola.ttf"))
    else:
        import fpdf

        font_dir = Path(fpdf.__file__).parent / "font"
        pdf.add_font("Body", "", str(font_dir / "DejaVuSans.ttf"))
        pdf.add_font("Body", "B", str(font_dir / "DejaVuSans-Bold.ttf"))
        pdf.add_font("Body", "I", str(font_dir / "DejaVuSans-Oblique.ttf"))
        pdf.add_font("Mono", "", str(font_dir / "DejaVuSansMono.ttf"))


def usable_width(pdf: FPDF) -> float:
    return pdf.w - pdf.l_margin - pdf.r_margin


def ensure_space(pdf: FPDF, height: float) -> None:
    if pdf.get_y() + height > pdf.h - pdf.b_margin:
        pdf.add_page()


def draw_heading(pdf: EntregaPDF, text: str, level: int = 1) -> None:
    sizes = {1: 15, 2: 12, 3: 10.5}
    ensure_space(pdf, 14)
    pdf.ln(3)
    pdf.set_font("Body", "B", sizes[level])
    pdf.set_text_color(*PRIMARY)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(usable_width(pdf), 6, text)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.4 if level == 1 else 0.25)
    y = pdf.get_y() + 1
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(4)


def draw_paragraph(pdf: FPDF, text: str, size: float = 10) -> None:
    pdf.set_font("Body", size=size)
    pdf.set_text_color(*TEXT)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(usable_width(pdf), 5.2, text)
    pdf.ln(2)


def draw_bullets(pdf: FPDF, items: list[str], size: float = 10) -> None:
    pdf.set_font("Body", size=size)
    pdf.set_text_color(*TEXT)
    for item in items:
        ensure_space(pdf, 8)
        pdf.set_x(pdf.l_margin + 2)
        pdf.cell(4, 5, "-")
        pdf.multi_cell(usable_width(pdf) - 6, 5.2, item)
    pdf.ln(2)


def draw_box(pdf: FPDF, title: str, content: str, title_color: tuple[int, int, int] = ACCENT) -> None:
    lines = content.strip().splitlines()
    h = 10 + len(lines) * 5.5 + 6
    ensure_space(pdf, h)
    x, y = pdf.l_margin, pdf.get_y()
    w = usable_width(pdf)
    pdf.set_fill_color(*LIGHT_BG)
    pdf.set_draw_color(*BORDER)
    pdf.rect(x, y, w, h, style="FD")
    pdf.set_xy(x + 4, y + 4)
    pdf.set_font("Body", "B", 9.5)
    pdf.set_text_color(*title_color)
    pdf.cell(w - 8, 5, title)
    pdf.ln(6)
    pdf.set_font("Body", size=9.5)
    pdf.set_text_color(*TEXT)
    for line in lines:
        pdf.set_x(x + 6)
        pdf.multi_cell(w - 12, 5, line)
    pdf.set_y(y + h + 4)


def draw_code_block(pdf: FPDF, text: str) -> None:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    h = max(12, len(lines) * 4.8 + 8)
    ensure_space(pdf, h)
    x, y = pdf.l_margin, pdf.get_y()
    w = usable_width(pdf)
    pdf.set_fill_color(248, 249, 250)
    pdf.set_draw_color(*BORDER)
    pdf.rect(x, y, w, h, style="FD")
    pdf.set_font("Mono", size=8.5)
    pdf.set_text_color(50, 50, 50)
    pdf.set_xy(x + 4, y + 4)
    for line in lines:
        pdf.set_x(x + 4)
        pdf.multi_cell(w - 8, 4.6, line)
    pdf.set_y(y + h + 4)


def parse_markdown_table(block: str) -> tuple[list[str], list[list[str]]] | None:
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("|"):
        return None
    rows = []
    for line in lines:
        if re.match(r"^\|[-| :]+\|$", line):
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    if len(rows) < 2:
        return None
    return rows[0], rows[1:]


def draw_table(pdf: FPDF, headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None) -> None:
    w_total = usable_width(pdf)
    n = len(headers)
    if col_widths is None:
        col_widths = [w_total / n] * n

    row_h = 7
    ensure_space(pdf, row_h * (len(rows) + 2))

    def row(cells: list[str], header: bool = False, fill: bool = False) -> None:
        x0 = pdf.l_margin
        y0 = pdf.get_y()
        if header:
            pdf.set_fill_color(*PRIMARY)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Body", "B", 8.5)
        else:
            pdf.set_fill_color(*(LIGHT_BG if fill else (255, 255, 255)))
            pdf.set_text_color(*TEXT)
            pdf.set_font("Body", size=8.5)
        for i, (cell, cw) in enumerate(zip(cells, col_widths)):
            pdf.set_xy(x0 + sum(col_widths[:i]), y0)
            pdf.cell(cw, row_h, cell[:80], border=1, fill=True, align="L")
        pdf.set_y(y0 + row_h)

    row(headers, header=True)
    for i, r in enumerate(rows):
        padded = r + [""] * (n - len(r))
        row(padded[:n], fill=i % 2 == 0)
    pdf.ln(4)


def render_markdown_block(pdf: EntregaPDF, block: str) -> None:
    block = block.strip()
    if not block:
        return

    table = parse_markdown_table(block)
    if table:
        headers, rows = table
        if len(headers) == 2:
            draw_table(pdf, headers, rows, [usable_width(pdf) * 0.38, usable_width(pdf) * 0.62])
        elif len(headers) == 4:
            w = usable_width(pdf)
            draw_table(pdf, headers, rows, [w * 0.28, w * 0.18, w * 0.18, w * 0.18])
        else:
            draw_table(pdf, headers, rows)
        return

    if block.startswith("```"):
        draw_code_block(pdf, re.sub(r"^```\w*\n?", "", block).rstrip("`"))
        return

    text = block
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)

    if re.search(r"^\d+\.\s", text, re.MULTILINE):
        items = [re.sub(r"^\d+\.\s*", "", ln).strip() for ln in text.splitlines() if ln.strip()]
        draw_bullets(pdf, items)
        return

    if text.startswith("- "):
        items = [ln[2:].strip() for ln in text.splitlines() if ln.startswith("- ")]
        draw_bullets(pdf, items)
        return

    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            draw_paragraph(pdf, para)


def extract_notebook_images() -> list[Path]:
    ASSETS.mkdir(exist_ok=True)
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    paths: list[Path] = []
    idx = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for output in cell.get("outputs", []):
            if "image/png" not in output.get("data", {}):
                continue
            img_bytes = base64.b64decode(output["data"]["image/png"])
            path = ASSETS / f"fig_{idx:02d}.png"
            Image.open(BytesIO(img_bytes)).save(path, format="PNG", optimize=True)
            paths.append(path)
            idx += 1
    return paths


def get_api_examples() -> dict:
    pipe = joblib.load(ROOT / "models" / "pipeline_breast_cancer.pkl")
    meta = joblib.load(ROOT / "models" / "pipeline_metadata.pkl")
    benign = json.loads((ROOT / "api" / "example_request.json").read_text(encoding="utf-8"))
    df = pd.DataFrame([benign["patient"]])[meta["feature_columns"]]
    pred = int(pipe.predict(df)[0])
    proba = pipe.predict_proba(df)[0]
    return {
        "model_name": meta["model_name"],
        "health": {"status": "ok", "model_name": meta["model_name"]},
        "predict": {
            "predicao": pred,
            "label": meta["target_labels"][pred],
            "probabilidade_benigno": round(float(proba[0]), 4),
            "probabilidade_maligno": round(float(proba[1]), 4),
        },
        "endpoints": [
            ("GET", "/health", "Status da API"),
            ("GET", "/metadata", "Metadados do modelo"),
            ("POST", "/predict", "Classificação benigno/maligno"),
        ],
    }


def draw_cover(pdf: EntregaPDF) -> None:
    pdf.add_page()
    pdf.set_fill_color(*PRIMARY)
    pdf.rect(0, 0, pdf.w, 72, style="F")
    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, 72, pdf.w, 3, style="F")

    pdf.set_y(22)
    pdf.set_font("Body", "B", 11)
    pdf.set_text_color(200, 220, 240)
    pdf.cell(0, 6, "INSTITUTO DE ADMINISTRAÇÃO E TECNOLOGIA", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Body", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(0, 10, "Tech Challenge — Fase 1", align="C")
    pdf.ln(2)
    pdf.set_font("Body", size=14)
    pdf.multi_cell(0, 8, "Predição de Câncer de Mama\n(Wisconsin Diagnostic)", align="C")

    pdf.ln(28)
    card_x = pdf.l_margin + 12
    card_w = usable_width(pdf) - 24
    card_y = pdf.get_y()
    card_h = 58
    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(*BORDER)
    pdf.rect(card_x, card_y, card_w, card_h, style="FD")
    pdf.set_xy(card_x + 8, card_y + 8)
    pdf.set_font("Body", "B", 10)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(card_w - 16, 6, "Repositório Git")
    pdf.ln(7)
    pdf.set_font("Body", size=9.5)
    pdf.set_text_color(*ACCENT)
    pdf.set_x(card_x + 8)
    pdf.multi_cell(card_w - 16, 5, REPO)
    pdf.ln(4)
    pdf.set_font("Body", size=9)
    pdf.set_text_color(*MUTED)
    pdf.set_x(card_x + 8)
    pdf.multi_cell(
        card_w - 16,
        4.8,
        "Machine Learning | API REST | Docker | Saúde da Mulher",
    )

    pdf.ln(16)
    draw_box(
        pdf,
        "Entregáveis incluídos",
        "Código-fonte completo  •  README.md  •  Dockerfile\n"
        "Dataset  •  Gráficos e métricas  •  Relatório técnico",
    )


def draw_deliverables_page(pdf: EntregaPDF) -> None:
    pdf.add_page()
    pdf.set_section("Mapa dos entregáveis")
    draw_heading(pdf, "Mapa dos entregáveis — Fase 1", 1)

    items = [
        ("Repositório Git", REPO),
        ("Código-fonte", "api/, models/, Tech_Challenge_Breast_Cancer.ipynb"),
        ("README", "Instruções de notebook, Docker e API"),
        ("Docker", "Dockerfile + docker-compose.yml"),
        ("Dataset", "breast_cancer_wisconsin.csv (569 amostras)"),
        ("Resultados", "Gráficos, métricas e exemplos de API neste PDF"),
        ("Relatório", "Seções 1 a 4 — EDA, pré-processamento, modelos e resultados"),
    ]
    draw_table(pdf, ["Item", "Descrição"], [[a, b] for a, b in items], [42, usable_width(pdf) - 42])

    draw_heading(pdf, "Métricas principais (conjunto de teste)", 2)
    draw_table(
        pdf,
        ["Modelo", "Accuracy", "Recall", "F1-Score"],
        [
            ["Logistic Regression", "97,4%", "95,2%", "96,4%"],
            ["Random Forest", "97,4%", "92,9%", "96,3%"],
            ["K-Nearest Neighbors", "95,6%", "90,5%", "93,8%"],
            ["Decision Tree", "90,4%", "83,3%", "86,4%"],
        ],
        [usable_width(pdf) * 0.4, 0.2, 0.2, 0.2],
    )
    draw_paragraph(
        pdf,
        "Modelo selecionado para produção: Logistic Regression (maior F1-Score).",
        size=9.5,
    )


def draw_report_sections(pdf: EntregaPDF, relatorio: str) -> None:
    relatorio = re.sub(r"### Mapa dos entregáveis.*?---\n", "", relatorio, flags=re.DOTALL)
    relatorio = re.sub(r"^# Relatório Técnico.*?\n", "", relatorio)
    relatorio = re.sub(r"^## Predição.*?\n", "", relatorio)

    pdf.add_page()
    pdf.set_section("Relatório técnico")
    draw_heading(pdf, "Relatório Técnico", 1)

    sections = re.split(r"\n(?=## )", relatorio)
    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines = section.split("\n", 1)
        title = lines[0].lstrip("# ").strip()
        body = lines[1] if len(lines) > 1 else ""

        if title.startswith("Referências"):
            draw_heading(pdf, title, 2)
            refs = [ln.lstrip("- ").strip() for ln in body.splitlines() if ln.strip().startswith("-")]
            draw_bullets(pdf, refs, size=9)
            continue

        draw_heading(pdf, title, 2)

        for part in re.split(r"\n(?=### )", body):
            part = part.strip()
            if not part:
                continue
            if part.startswith("### "):
                sub = part.split("\n", 1)
                draw_heading(pdf, sub[0].replace("### ", ""), 3)
                part = sub[1] if len(sub) > 1 else ""
            for chunk in re.split(r"\n(?=\|)", part):
                chunk = chunk.strip()
                if chunk:
                    render_markdown_block(pdf, chunk)


def draw_figure_page(pdf: EntregaPDF, path: Path, caption: str, full_page: bool = False) -> None:
    pdf.add_page()
    pdf.set_section("Resultados visuais")

    with Image.open(path) as img:
        iw, ih = img.size
    ratio = iw / ih
    w_max = usable_width(pdf)
    h_max = pdf.h - pdf.t_margin - pdf.b_margin - 28

    if full_page:
        w = w_max
        h = w / ratio
        if h > h_max:
            h = h_max
            w = h * ratio
    else:
        w = w_max * 0.92
        h = w / ratio
        if h > h_max * 0.85:
            h = h_max * 0.85
            w = h * ratio

    ensure_space(pdf, h + 20)
    x = pdf.l_margin + (w_max - w) / 2
    y = pdf.get_y() + 2

    pdf.set_draw_color(*BORDER)
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(x - 2, y - 2, w + 4, h + 4, style="D")
    pdf.image(str(path), x=x, y=y, w=w, h=h)
    pdf.set_y(y + h + 8)

    pdf.set_font("Body", "B", 9.5)
    pdf.set_text_color(*PRIMARY)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(usable_width(pdf), 5, caption, align="C")
    pdf.ln(2)


def draw_api_section(pdf: EntregaPDF, api: dict) -> None:
    pdf.add_page()
    pdf.set_section("API REST")
    draw_heading(pdf, "API REST — Resultados", 1)
    draw_paragraph(
        pdf,
        "API FastAPI containerizada com Docker. A pipeline treinada no notebook "
        "é carregada automaticamente na inicialização.",
    )

    draw_table(
        pdf,
        ["Método", "Rota", "Descrição"],
        [[m, r, d] for m, r, d in api["endpoints"]],
        [22, 30, usable_width(pdf) - 52],
    )

    draw_heading(pdf, "Exemplo — GET /health", 2)
    draw_code_block(pdf, json.dumps(api["health"], indent=2, ensure_ascii=False))

    draw_heading(pdf, "Exemplo — POST /predict", 2)
    draw_paragraph(pdf, "Request: api/example_request.json (30 medidas de núcleos celulares)", size=9.5)
    draw_code_block(pdf, json.dumps(api["predict"], indent=2, ensure_ascii=False))

    draw_box(
        pdf,
        "Deploy",
        f"Modelo: {api['model_name']}\n"
        "Arquivo: models/pipeline_breast_cancer.pkl\n"
        "Comando: docker compose up --build\n"
        "Docs: http://localhost:8000/docs",
    )


def build_pdf() -> None:
    images = extract_notebook_images()
    api = get_api_examples()
    relatorio = RELATORIO.read_text(encoding="utf-8")

    pdf = EntregaPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 20, 18)
    pdf.alias_nb_pages()
    load_fonts(pdf)

    draw_cover(pdf)
    draw_deliverables_page(pdf)
    draw_report_sections(pdf, relatorio)

    for i, path in enumerate(images):
        caption = IMAGE_CAPTIONS[i] if i < len(IMAGE_CAPTIONS) else f"Figura {i + 1}"
        full = i in {1, 2, 3}  # histograma, pairplot, correlação — página inteira
        draw_figure_page(pdf, path, caption, full_page=full)

    draw_api_section(pdf, api)

    pdf.output(str(OUTPUT))
    print(f"PDF gerado: {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB, {pdf.page_no()} páginas)")


if __name__ == "__main__":
    build_pdf()
