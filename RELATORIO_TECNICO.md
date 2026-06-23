# Relatório Técnico — Tech Challenge Fase 1
## Predição de Câncer de Mama (Wisconsin Diagnostic)

**Projeto:** Sistema de apoio à decisão clínica com Machine Learning  
**Dataset:** `breast_cancer_wisconsin.csv`  
**Variável alvo:** `diagnosis` (0 = benigno, 1 = maligno)  
**Referência:** [Kaggle - Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)  
**Notebook:** `Tech_Challenge_Breast_Cancer.ipynb`

### Contexto clínico

O dataset **Breast Cancer Wisconsin (Diagnostic)** contém **569 amostras** de massas mamárias analisadas por **PAAF** (Punção Aspirativa por Agulha Fina). Para cada núcleo celular foram calculadas **30 medidas numéricas** (média, erro padrão e pior valor de 10 características morfológicas), permitindo classificar o tumor como **benigno** ou **maligno**.

---

## 1. Discussões da Análise Exploratória

### 1.1 Contexto do problema

O conjunto foi curado por Wolberg, Street e Mangasarian (University of Wisconsin) e está disponível no repositório UCI. É um benchmark clássico em classificação médica e saúde da mulher — tema central do Tech Challenge.

A variável alvo é **`diagnosis`**: `B` (benigno) ou `M` (maligno), codificada como 0 e 1 no pipeline.

### 1.2 Dimensão e balanceamento

| Aspecto | Valor |
|---------|-------|
| Registros | 569 |
| Atributos | 32 colunas (id + diagnosis + 30 features) |
| Benigno (0) | 357 (62,7%) |
| Maligno (1) | 212 (37,3%) |

O dataset é **moderadamente balanceado** em comparação com problemas de triagem por questionário, o que permite métricas mais estáveis (recall, F1 e accuracy).

### 1.3 Valores ausentes

**Nenhum valor ausente** no dataset diagnostic. Não foi necessária imputação agressiva; o `SimpleImputer` no pipeline permanece por consistência e robustez em produção.

### 1.4 Variáveis relevantes

As 30 features derivam de 10 medidas de núcleos celulares:

- **Geométricas:** raio, perímetro, área
- **Textura:** desvio padrão de tons de cinza
- **Forma:** suavidade, compacidade, concavidade, pontos côncavos, simetria, dimensão fractal

Cada medida aparece em três versões: **mean** (média), **error** (erro padrão) e **worst** (média dos três maiores valores).

### 1.5 Correlação

A matriz de correlação mostra forte associação entre raio, perímetro e área — esperado por definição geométrica. Features `worst_*` tendem a correlacionar fortemente com o diagnóstico maligno.

### 1.6 Conclusões da EDA

1. Dataset adequado ao desafio de classificação em saúde feminina.
2. Balanceamento razoável permite usar accuracy, recall e F1 de forma complementar.
3. Features derivadas diretamente da amostra citológica — contexto de **apoio ao diagnóstico pós-PAAF**, não triagem populacional por questionário.

---

## 2. Estratégias de Pré-processamento

### 2.1 Seleção de features (30 colunas)

| Excluída | Motivo |
|----------|--------|
| `id` | Identificador da amostra |
| `diagnosis` | Variável alvo |

### 2.2 Pipeline sklearn

```
SimpleImputer(median) → StandardScaler → Classificador
```

Encapsulado em `ColumnTransformer` + `Pipeline` para uso idêntico no treino e na API REST.

### 2.3 Divisão dos dados

- **80% treino** (455) / **20% teste** (114)
- Estratificação por `diagnosis`
- `random_state=42`

### 2.4 Balanceamento

`class_weight='balanced'` em Regressão Logística, Árvore e Random Forest.

---

## 3. Modelos Usados e Porquê

| Modelo | Justificativa |
|--------|---------------|
| **Regressão Logística** | Baseline interpretável; excelente em dados linearmente separáveis |
| **Árvore de Decisão** | Captura regras não lineares entre medidas morfológicas |
| **Random Forest** | Ensemble robusto; referência em benchmarks tabulares |
| **KNN** | Compara amostras similares; referência do notebook de ML avançado |

Critério de seleção para deploy: **maior F1-Score** no teste.

---

## 4. Resultados e Interpretação dos Dados

### 4.1 Métricas comparativas (teste — 114 amostras)

| Modelo | Accuracy | Recall | F1-Score |
|--------|----------|--------|----------|
| **Logistic Regression** | **97,4%** | **95,2%** | **96,4%** |
| Random Forest | 97,4% | 92,9% | 96,3% |
| K-Nearest Neighbors | 95,6% | 90,5% | 93,8% |
| Decision Tree | 90,4% | 83,3% | 86,4% |

### 4.2 Regressão Logística — detalhamento

```
              precision    recall  f1-score   support

     Benigno       0.97      0.99      0.98        72
     Maligno       0.98      0.95      0.96        42

    accuracy                           0.97       114
```

### 4.3 Interpretação

- **Regressão Logística** apresenta melhor **F1-Score (96,4%)** e foi salva como pipeline final para REST.
- **Recall de 95,2%** para malignos — poucos falsos negativos, crítico em contexto oncológico.
- Métricas altas refletem que as features são extraídas do **mesmo material citológico** usado no diagnóstico — problema bem definido e com forte sinal preditivo.
- **Falsos negativos** (maligno classificado como benigno) permanecem o erro mais grave clinicamente.

### 4.4 Uso prático

1. **Apoio à decisão** em análise de PAAF — segunda opinião automatizada.
2. **API REST** — pipeline em `models/pipeline_breast_cancer.pkl`.
3. **Integração** com sistemas de patologia digital (mediante extração das 30 medidas).

### 4.5 Limitações

- Amostra de **569 casos** de uma única origem (Wisconsin, anos 1990).
- Features exigem **imagem digitalizada de PAAF** — não aplicável a triagem só com dados demográficos.
- Não substitui avaliação médica, mamografia nem biópsia definitiva.
- **O profissional de saúde tem a palavra final.**

### 4.6 Conclusão

O projeto atende ao Tech Challenge com classificação em saúde feminina, EDA documentada, pipeline reprodutível e modelo serializado para REST. O dataset Wisconsin oferece base mais estável que problemas fortemente desbalanceados por questionário, com métricas adequadas para demonstração e deploy.

---

## Referências

- UCI ML Repository: [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- Kaggle: [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- Tech Challenge IADT — Fase 1 (`IADT - Fase 1 - Tech challenge A.pdf`)
