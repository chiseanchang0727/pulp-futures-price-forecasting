# Pulp Futures Price Forecasting

This repository contains a project aimed at building a model to predict the futures price of Bleached Softwood Kraft Pulp. The project is designed for iterative development, with the initial focus on setting up training and evaluation pipelines. Feature engineering and advanced modeling techniques will be implemented in subsequent phases.

---

## Features
- Predicts futures prices for Bleached Softwood Kraft Pulp.
- Initial model: Multi-Layer Perceptron (MLP).
- Upcoming enhancements:
  - Long Short-Term Memory (LSTM) models.
  - Ensemble learning techniques.
- Modular design for feature engineering and experimentation.

---

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/pulp-futures-price-forecasting.git
cd pulp-futures-price-forecasting
pip install -r requirements.txt


## Usage

### Training
```bash
python main.py --mode train

### Evaluation
```bash
python main.py --mode eval

### Hyperparameter configuration
Hyperparameters are stored in the ./configs.yaml file.

