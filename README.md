# Tech Challenge — Câncer de Mama (Fase 1)

Predição **benigno vs maligno** com Machine Learning (Breast Cancer Wisconsin) e API REST em Docker.

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
- Postman: `postman/Tech_Challenge_Breast_Cancer.postman_collection.json`

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

## 3. Relatório técnico

Documentação completa da EDA, modelos e resultados: `RELATORIO_TECNICO.md`.

> Ferramenta de **apoio à decisão**. Não substitui avaliação médica nem biópsia.
