# Setup

## Prerequisites

- Python 3.10+
- VS Code veya Jupyter destekli IDE
- Paket yöneticisi: pip

## Installation

### Create Virtual Environment
'''
python -m venv .venv'''

### Activate Environment
'''
# Windows
.venv\Scripts\activate
'''
'''
# Mac/Linux
source .venv/bin/activate
'''

### Install Dependencies
'''
pip install -r requirements.txt
'''

### Dependencies

**Core:**
- pandas, numpy, scikit-learn

**Visualization / Dev:**
- matplotlib, seaborn, plotly
- jupyter

## Project Structure

```
FinRisk-AI/
├── data/
│   ├── raw/           # Original datasets
│   ├── processed/     # Cleaned and transformed data
│   └── samples/       # Sample datasets for experiment
├── models/            # Saved models
├── notebooks/
│   ├── analysis/      # EDA, data exploration, visualizations
│   └── modeling/      # Baseline, feature engineering, model training
├── src/               # Source code for pipeline, inference, API
├── docs/              # Documentation
└── tests/             # Unit tests, validation scripts
```