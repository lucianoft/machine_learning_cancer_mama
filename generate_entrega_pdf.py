"""Gera ENTREGA_FASE_1.pdf com relatório, gráficos e resultados da API."""
from __future__ import annotations

import base64
import json
import re
import textwrap
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

IMAGE_CAPTIONS = {
    0: "Distribuição da variável alvo (diagnosis: benigno vs maligno)",
    1: "Histogramas das variáveis numéricas do dataset",
    2: "Pairplot das 10 medidas mean_* (colorido por diagnóstico)",
    3: "Matriz de correlação entre as features",
    4: "Matriz de confusão — Regressão Logística",
    5: "Comparação de modelos por F1-Score",
}


class EntregaPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("DejaVu", size=8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, f"Página {self.page_no()}/{{nb}}", align="C")


def strip_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^---\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def extract_notebook_images() -> list[Path]:
    ASSETS.mkdir(exist_ok=True)
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    paths: list[Path] = []
    idx = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" not in data:
                continue
            img_bytes = base64.b64decode(data["image/png"])
            path = ASSETS / f"fig_{idx:02d}.png"
            Image.open(BytesIO(img_bytes)).save(path, format="PNG")
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
            ("GET", "/health", "Status da API e modelo carregado"),
            ("GET", "/metadata", "Metadados, features e variável alvo"),
            ("POST", "/predict", "Classificação benigno/maligno (30 features)"),
        ],
    }


def write_paragraph(pdf: EntregaPDF, text: str, size: int = 10) -> None:
    pdf.set_font("DejaVu", size=size)
    pdf.set_text_color(30, 30, 30)
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if paragraph.startswith("|"):
            pdf.set_font("DejaVu", size=size - 1)
            for line in paragraph.splitlines():
                if re.match(r"^\|[-| ]+\|$", line.strip()):
                    continue
                clean = " | ".join(cell.strip() for cell in line.strip("|").split("|"))
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5, clean)
            pdf.ln(2)
            pdf.set_font("DejaVu", size=size)
            continue
        if paragraph.startswith("```"):
            pdf.set_font("DejaVu", size=size - 1)
            for line in paragraph.splitlines():
                if line.startswith("```"):
                    continue
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5, line)
            pdf.ln(2)
            pdf.set_font("DejaVu", size=size)
            continue
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, paragraph)
        pdf.ln(2)


def write_heading(pdf: EntregaPDF, text: str, level: int = 1) -> None:
    sizes = {1: 16, 2: 13, 3: 11}
    pdf.ln(4)
    pdf.set_font("DejaVu", "B", sizes.get(level, 11))
    pdf.set_text_color(20, 60, 120)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 7, text)
    pdf.ln(2)


def add_image_fit(pdf: EntregaPDF, path: Path, caption: str) -> None:
    with Image.open(path) as img:
        img_w, img_h = img.size

    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    max_h = pdf.h - pdf.t_margin - pdf.b_margin - 30
    ratio = img_w / img_h

    w = page_w
    h = w / ratio
    if h > max_h:
        h = max_h
        w = h * ratio

    if pdf.get_y() + h + 20 > pdf.h - pdf.b_margin:
        pdf.add_page()

    x = pdf.l_margin + (page_w - w) / 2
    pdf.image(str(path), x=x, y=pdf.get_y(), w=w, h=h)
    pdf.ln(h + 2)
    pdf.set_font("DejaVu", "I", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 5, caption, align="C")
    pdf.ln(4)


def build_pdf() -> None:
    images = extract_notebook_images()
    api = get_api_examples()
    relatorio_raw = RELATORIO.read_text(encoding="utf-8")

    # Remove seção "Mapa dos entregáveis" do corpo (já na capa)
    relatorio_raw = re.sub(
        r"### Mapa dos entregáveis.*?---\n",
        "",
        relatorio_raw,
        flags=re.DOTALL,
    )

    pdf = EntregaPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.alias_nb_pages()
    pdf.add_font("DejaVu", "", r"C:\Windows\Fonts\arial.ttf")
    pdf.add_font("DejaVu", "B", r"C:\Windows\Fonts\arialbd.ttf")
    pdf.add_font("DejaVu", "I", r"C:\Windows\Fonts\ariali.ttf")

    # Capa
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 20)
    pdf.set_text_color(20, 60, 120)
    pdf.ln(35)
    pdf.multi_cell(0, 10, "Tech Challenge IADT — Fase 1", align="C")
    pdf.ln(6)
    pdf.set_font("DejaVu", "B", 16)
    pdf.multi_cell(0, 9, "Predição de Câncer de Mama\n(Wisconsin Diagnostic)", align="C")
    pdf.ln(20)
    pdf.set_font("DejaVu", size=11)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 7, f"Repositório Git:\n{REPO}", align="C")
    pdf.ln(8)
    pdf.set_font("DejaVu", size=10)
    pdf.multi_cell(
        0,
        6,
        "Entregáveis: código-fonte, README, Dockerfile, dataset,\n"
        "resultados (gráficos e métricas) e relatório técnico.",
        align="C",
    )

    # Mapa entregáveis
    pdf.add_page()
    write_heading(pdf, "Mapa dos entregáveis — Fase 1", 1)
    entregaveis = (
        f"Repositório Git: {REPO}\n\n"
        "Código-fonte: api/, models/, Tech_Challenge_Breast_Cancer.ipynb\n"
        "README: README.md (instruções notebook + Docker)\n"
        "Docker: Dockerfile e docker-compose.yml\n"
        "Dataset: breast_cancer_wisconsin.csv (repositório)\n"
        "Links: Kaggle e UCI (ver relatório)\n"
        "Resultados: gráficos e métricas neste PDF + notebook\n"
        "Relatório técnico: seções 1 a 4 abaixo"
    )
    write_paragraph(pdf, entregaveis)

    # Relatório técnico (markdown simplificado)
    pdf.add_page()
    write_heading(pdf, "Relatório Técnico", 1)

    blocks = re.split(r"\n(?=## )", relatorio_raw)
    for block in blocks:
        block = block.strip()
        if not block or block.startswith("# Relatório"):
            continue
        lines = block.split("\n", 1)
        title = lines[0].lstrip("# ").strip()
        body = lines[1] if len(lines) > 1 else ""

        if title.startswith("## "):
            write_heading(pdf, title.replace("## ", ""), 2)
        elif title.startswith("### "):
            write_heading(pdf, title.replace("### ", ""), 3)
        else:
            write_heading(pdf, title, 2)

        for part in re.split(r"\n(?=### )", body):
            part = part.strip()
            if part.startswith("### "):
                sub_lines = part.split("\n", 1)
                write_heading(pdf, sub_lines[0].replace("### ", ""), 3)
                part = sub_lines[1] if len(sub_lines) > 1 else ""
            if part:
                write_paragraph(pdf, strip_markdown(part))

    # Gráficos
    pdf.add_page()
    write_heading(pdf, "Resultados visuais — Notebook", 1)
    write_paragraph(
        pdf,
        "Gráficos gerados pela execução de Tech_Challenge_Breast_Cancer.ipynb "
        "(análise exploratória, matriz de confusão e comparação de modelos).",
    )
    for i, path in enumerate(images):
        caption = IMAGE_CAPTIONS.get(i, f"Figura {i + 1}")
        add_image_fit(pdf, path, caption)

    # API
    pdf.add_page()
    write_heading(pdf, "API REST — Resultados", 1)
    write_paragraph(
        pdf,
        "API FastAPI em Docker. Documentação interativa: http://localhost:8000/docs\n"
        "Collection Postman: postman/Tech_Challenge_Breast_Cancer.postman_collection.json",
    )
    write_heading(pdf, "Endpoints", 2)
    for method, route, desc in api["endpoints"]:
        write_paragraph(pdf, f"{method} {route} — {desc}", size=10)

    write_heading(pdf, "Exemplo GET /health", 2)
    write_paragraph(pdf, json.dumps(api["health"], indent=2, ensure_ascii=False))

    write_heading(pdf, "Exemplo POST /predict (amostra benigna)", 2)
    write_paragraph(
        pdf,
        "Request: api/example_request.json (30 medidas de núcleos celulares)\n\n"
        f"Response:\n{json.dumps(api['predict'], indent=2, ensure_ascii=False)}",
    )
    write_paragraph(
        pdf,
        f"Modelo em produção: {api['model_name']}\n"
        "Pipeline serializada: models/pipeline_breast_cancer.pkl",
    )

    write_heading(pdf, "Execução", 2)
    write_paragraph(
        pdf,
        "docker compose up --build\n"
        "curl -X POST http://localhost:8000/predict "
        "-H \"Content-Type: application/json\" -d @api/example_request.json",
        size=9,
    )

    pdf.output(str(OUTPUT))
    print(f"PDF gerado: {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build_pdf()
