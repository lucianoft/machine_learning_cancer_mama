# Tech Challenge — Câncer de Mama (Fase 1)

Predição **benigno vs maligno** com Machine Learning (Breast Cancer Wisconsin) e API REST em Docker.

**Repositório:** https://github.com/lucianoft/machine_learning_cancer_mama

```bash
git clone https://github.com/lucianoft/machine_learning_cancer_mama.git
cd machine_learning_cancer_mama
```

## Estrutura do projeto

```
desafio/
├── Tech_Challenge_Breast_Cancer.ipynb   # Notebook (EDA, modelos, pipeline)
├── breast_cancer_wisconsin.csv          # Dataset
├── RELATORIO_TECNICO.md                 # Relatório técnico
├── requirements.txt                     # Dependências do notebook
├── requirements-api.txt                 # Dependências da API
├── api/                                 # Código FastAPI
├── models/                              # Pipeline treinada (.pkl)
├── postman/                             # Collection para testes
├── ENTREGA_FASE_1.pdf                   # PDF de entrega (Fase 1)
├── Dockerfile
└── docker-compose.yml
```

## 1. Notebook

```bash
pip install -r requirements.txt
jupyter notebook Tech_Challenge_Breast_Cancer.ipynb
```

Execute todas as células. A seção 8 gera `models/pipeline_breast_cancer.pkl` e `models/pipeline_metadata.pkl`.

## 2. API REST (Docker)

**Pré-requisito:** arquivos em `models/` (gerados pelo notebook).

```bash
docker compose up --build
```

- API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Postman: `postman/Tech_Challenge_Breast_Cancer.postman_collection.json` — inclui **30 variáveis** com descrição de cada campo, exemplos benigno/maligno e template para preencher

### Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Status da API |
| GET | `/metadata` | Metadados do modelo |
| POST | `/predict` | Classificação benigno/maligno |

### Exemplo

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @api/example_request.json
```

### Sem Docker

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload --port 8000
```

## 3. Dataset

O arquivo `breast_cancer_wisconsin.csv` já está no repositório (569 amostras). Fontes oficiais:

- [Kaggle — Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

**Variável alvo:** `diagnosis` — `B` = benigno, `M` = maligno (no notebook/API: 0 = benigno, 1 = maligno).

As features são medidas de **núcleos celulares** extraídas de imagens de PAAF. Cada uma das 10 características abaixo aparece em três versões:

| Sufixo | Significado |
|--------|-------------|
| `mean` | Média dos valores por núcleo |
| `error` | Erro padrão |
| `worst` | Média dos três maiores valores |

### Colunas do dataset

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `id` | int | Identificador da amostra |
| `diagnosis` | texto | Diagnóstico: `B` (benigno) ou `M` (maligno) |
| `mean radius` | float | Raio médio — média das distâncias do centro ao perímetro do núcleo |
| `mean texture` | float | Textura média — desvio padrão dos tons de cinza |
| `mean perimeter` | float | Perímetro médio do núcleo |
| `mean area` | float | Área média do núcleo |
| `mean smoothness` | float | Suavidade média — variação local do raio |
| `mean compactness` | float | Compacidade média — (perímetro² / área) − 1 |
| `mean concavity` | float | Concavidade média — severidade das porções côncavas do contorno |
| `mean concave points` | float | Pontos côncavos médios — número de porções côncavas |
| `mean symmetry` | float | Simetria média do núcleo |
| `mean fractal dimension` | float | Dimensão fractal média — aproximação da “costa” do contorno |
| `radius error` | float | Erro padrão do raio |
| `texture error` | float | Erro padrão da textura |
| `perimeter error` | float | Erro padrão do perímetro |
| `area error` | float | Erro padrão da área |
| `smoothness error` | float | Erro padrão da suavidade |
| `compactness error` | float | Erro padrão da compacidade |
| `concavity error` | float | Erro padrão da concavidade |
| `concave points error` | float | Erro padrão dos pontos côncavos |
| `symmetry error` | float | Erro padrão da simetria |
| `fractal dimension error` | float | Erro padrão da dimensão fractal |
| `worst radius` | float | Pior raio — média dos 3 maiores valores |
| `worst texture` | float | Pior textura — média dos 3 maiores valores |
| `worst perimeter` | float | Pior perímetro — média dos 3 maiores valores |
| `worst area` | float | Pior área — média dos 3 maiores valores |
| `worst smoothness` | float | Pior suavidade — média dos 3 maiores valores |
| `worst compactness` | float | Pior compacidade — média dos 3 maiores valores |
| `worst concavity` | float | Pior concavidade — média dos 3 maiores valores |
| `worst concave points` | float | Pior pontos côncavos — média dos 3 maiores valores |
| `worst symmetry` | float | Pior simetria — média dos 3 maiores valores |
| `worst fractal dimension` | float | Pior dimensão fractal — média dos 3 maiores valores |

> Na API (`POST /predict`), envie as **30 features** (sem `id` nem `diagnosis`). Exemplo em `api/example_request.json`.

## 4. Relatório técnico

Documentação completa da EDA, modelos e resultados: `RELATORIO_TECNICO.md`.

Gráficos e métricas detalhadas: execute o notebook (`Tech_Challenge_Breast_Cancer.ipynb`).

## 5. Entrega Fase 1 (PDF)

Monte um **único PDF** para a plataforma com:

1. Link do repositório: https://github.com/lucianoft/machine_learning_cancer_mama  
2. Referência ao código, README, Dockerfile e dataset (itens acima)  
3. **Prints** do notebook: distribuição do alvo, histogramas, pairplot, matriz de correlação, matriz de confusão e comparação de modelos  
4. **Prints** da API: Swagger (`/docs`) e resposta do `POST /predict` (Postman ou curl)  
5. Resumo do relatório técnico (ou anexe/exporte o `RELATORIO_TECNICO.md` em PDF)

Arquivo de entrega: **`ENTREGA_FASE_1.pdf`** (na raiz do projeto).

> Ferramenta de **apoio à decisão**. Não substitui avaliação médica nem biópsia.
