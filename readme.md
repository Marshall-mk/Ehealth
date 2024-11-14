# Student Lab

## Introduction
This repository is designed to help you learn how to convert a typical experimental data science notebook into a well-structured, scalable, and maintainable Python codebase. By following the steps provided, you will gain skills in software engineering practices within data science, including proper folder structuring, code modularization, environment setup, and experiment tracking.

## Repository Structure
```
project/
├── data/                # Directory for storing raw, interim, and processed data.
├── notebooks/           # Jupyter notebooks for experimentation and exploration.
├── src/                 # Source folder for all Python scripts.
│   ├── data/            # Scripts for data loading and processing.
│   ├── features/        # Scripts for feature engineering.
│   ├── models/          # Scripts for defining, training, and evaluating models.
│   └── utils/           # Utility scripts and helper functions.
├── requirements.txt     # Project dependencies.
├── config/              # Configurations for model parameters, data paths, and experiments.
└── README.md            # Project documentation (this file).
```

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Git
- `pip` or `conda` for managing Python packages

### Setting Up the Environment
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd project
   ```
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate  # For Windows
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Initial Exploration
1. **Navigate to the notebooks directory**:
   The `notebooks/` folder contains Jupyter notebooks where you can start your exploration and experimentation.
   ```bash
   jupyter notebook
   ```
2. Open the `sample_notebook.ipynb` to see how a typical experiment is structured.

## Converting a Notebook to a Structured Codebase
The key learning objective of this lab is to transform the unstructured code in the notebook into a production-ready Python project. Follow these steps:

1. **Refactor Code into Modules**
   - Move data loading code to `src/data/`.
   - Put feature engineering code in `src/features/`.
   - Place model training and evaluation code in `src/models/`.
   - Create utility functions (e.g., logging, metrics) in `src/utils/`.

2. **Configuration Management**
   - Use configuration files stored in the `config/` directory to manage hyperparameters, file paths, and other settings.
   - Explore `OmegaConf` or `hydra` for flexible configuration management.

3. **Environment and Dependencies**
   - Use `pipreqs` to generate a `requirements.txt` file that includes the dependencies for your codebase.
   ```bash
   pipreqs .
   ```

## Experiment Tracking
To track experiments and results, you can use either MLFlow or Weights and Biases (wandb).

1. **Setting Up MLFlow**
   - Install MLFlow:
     ```bash
     pip install mlflow
     ```
   - Run an experiment:
     ```bash
     mlflow run .
     ```
   - MLFlow will track metrics, parameters, and artifacts for each experiment.

2. **Using Weights and Biases (wandb)**
   - Create an account at [wandb.ai](https://wandb.ai) and install the library:
     ```bash
     pip install wandb
     ```
   - Log in and initialize wandb to log experiment results.

## Version Control for Data and Models
**Data Version Control (DVC)** helps keep track of changes to datasets and models. Follow these steps:
1. **Install DVC**:
   ```bash
   pip install dvc
   ```
2. **Initialize DVC in the repository**:
   ```bash
   dvc init
   ```
3. **Track data files**:
   ```bash
   dvc add data/raw/
   ```
   This will create `.dvc` files that can be versioned using Git.

## Best Practices
- **Write Modular Code**: Split your code into reusable functions and classes.
- **Document Your Code**: Use comments and docstrings to explain your code.
- **Use Version Control**: Use Git for version control, committing changes regularly.
- **Use Virtual Environments**: Keep dependencies isolated to avoid conflicts.

## Additional Resources
- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Hydra Documentation](https://hydra.cc/docs/intro/)

## Summary
This lab is designed to help you gain experience in structuring data science projects properly. By completing these tasks, you'll learn how to write clean, reusable, and scalable code for real-world data science applications.

Feel free to ask questions or consult the instructor if you get stuck at any point.

---
**Happy coding!**
