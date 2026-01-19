# XAI Alternative Assessment

This project demonstrates Explainable AI (XAI) techniques for diabetes prediction using machine learning. Using the Pima Indians Diabetes Database, it focuses on building interpretable models to predict diabetes outcomes and explaining the predictions through various XAI methods.

## Project Structure

- **app.py** - Interactive Streamlit web application for model exploration and XAI visualization
- **code.ipynb** - Jupyter notebook with detailed demonstrations and analysis
- **archive/diabetes.csv** - Pima Indians Diabetes Database used for training and evaluation
- **requirements.txt** - Python dependencies

## Files

### app.py
A Streamlit application that provides an interactive interface for:
- Diabetes risk predictions
- Feature importance visualization
- Explainable AI insights
- Interactive data exploration

To run the Streamlit app:
```bash
streamlit run app.py
```

### code.ipynb
A comprehensive Jupyter notebook containing:
- Pima Indians Diabetes Database exploration and preprocessing
- Model training and evaluation for diabetes prediction
- XAI technique demonstrations
- Visualizations and analysis

## Key Features

### User Role Personalization
The application tailors the experience based on the selected stakeholder role:
- **Domain Specialists**: Focuses on clinical plausibility and feature importance (SHAP).
- **Regulators & Governance Bodies**: Emphasizes global model behavior, transparency, and fairness (SHAP Global Summary).
- **End Users (Patients)**: Provides simplified, empathetic explanations and actionable counterfactual examples (DiCE).
- **Data Scientists & AI Developers**: Offers technical depth with full access to model metrics, SHAP plots, and detailed counterfactuals.

### LLM-Powered Explanations (AI Specialist)
Integrated with **Groq's API**, the app generates context-aware natural language explanations for each prediction. The AI acts as a specialized assistant, dynamically adjusting its tone and content focus according to the selected user role:
- **For Domain Specialists**: Analyzes risk factors like Glucose and BMI in a clinical context.
- **For Regulators**: Discusses alignment with clinical guidelines and model safety.
- **For Patients**: Offers non-technical, reassuring advice and lifestyle suggestions.
- **For Data Scientists**: Breaks down prediction probabilities and feature contributions.

*Note: Requires a `GROQ_API_KEY` in the `.env` file to function.*

## Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Interactive App**: Run `streamlit run app.py` to launch the web interface
2. **Notebook Demo**: Open `code.ipynb` in Jupyter Lab or VS Code to explore the full analysis

## Requirements

See [requirements.txt](requirements.txt) for a complete list of dependencies.
