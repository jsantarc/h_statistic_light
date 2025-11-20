# h_statistic_light

`h_statistic_light` is a lightweight Python implementation of the **H-statistic**, a metric for quantifying **interaction strength between features** in machine learning models. It is especially useful for interpreting complex models (e.g. tree ensembles, gradient boosting) by identifying and measuring how pairs of features interact.

> **Note**: As of now, this implementation has **not been tested with classification models**.

---

## Why the H-Statistic?

For a model $f(x) $ the H-statistic for a pair of features $x_j, x_k$ measures how much of the variance in the model’s predictions is explained by their **joint effect** beyond the sum of their individual effects.

Roughly speaking:

- **H ≈ 0**: little to no interaction between the two features  
- **H ≈ 1**: strong interaction; their joint effect explains most of the variance

This is particularly helpful for:

- Understanding **black-box models** (e.g. random forests, gradient boosting)  
- Feature engineering and **interaction discovery**  
- Model debugging and **interpretability reports**

---

## Features

- ⚡ Lightweight implementation focused on interaction strength via the H-statistic  
- 🤝 Designed to work with common ML models (e.g. scikit-learn estimators that implement `fit` / `predict`)  
- 📓 Example Jupyter notebook (`Feature Interaction_example.ipynb`) demonstrating usage  
- 🧩 Minimal dependencies and simple API

---

## Installation

This project is not (yet) published to PyPI. To use it in your own work:

```bash
# Clone the repository
git clone https://github.com/jsantarc/h_statistic_light.git
cd h_statistic_light

# (Optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .
```

---

## Quickstart

Below is a **high-level usage sketch**. For a complete working example, see  
`Feature Interaction_example.ipynb`.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# 1. Fit any regression model that implements fit / predict
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# 2. Choose the two features whose interaction you want to measure
feature_j = 0
feature_k = 1

# 3. Compute the H-statistic using this repo's implementation
from h_statistic_light import compute_h_statistic

h_value = compute_h_statistic(
    model=model,
    X=X_train,
    feature_indices=(feature_j, feature_k),
    grid_resolution=50,
)

print(f"H-statistic for features ({feature_j}, {feature_k}) = {h_value:.3f}")
```

---

## Interpreting the Results

- **H ≈ 0** — little to no interaction  
- **0 < H < 1** — some interaction  
- **H → 1** — strong interaction; their joint effect explains most of the prediction variance

You can compute the H-statistic across many feature pairs and rank them to find the strongest interactions.

---

## Working with Different Models

- ✅ **Regression models** (tree ensembles, boosting, etc.) fully supported  
- ⚠️ **Classification models** not yet tested — use caution

---

## Repository Structure

- `h_statistic_light/` – core implementation  
- `Feature Interaction_example.ipynb` – usage example notebook  
- `README.md` – this file  

---

## Roadmap

- More examples (RandomForestRegressor, XGBoost, LightGBM)  
- Systematic testing on classification models  
- Tools for visualizing interaction heatmaps  
- Automatic ranking of all feature interactions  

---

## License

. MIT, Apache-2.0
