# Student Lab

## Introduction
This repository is designed to help you learn how to convert a typical experimental data science notebook into a well-structured, scalable, and maintainable Python codebase. By following the steps provided, you will gain skills in software engineering practices within data science, including proper folder structuring, code modularization, environment setup, and experiment tracking.

## Repository Structure
```
project/
├── data/                # Directory for storing datasets.
├── notebooks/           # Jupyter notebooks for experimentation and exploration.
├── src/                 # Source folder for all Python scripts.
│   ├── data.py          # Script for data loading and processing.
│   ├── train.py         # Script for training models (could be called main)
|   ├── trainer.py       # Script containing the custom training functions.
│   ├── models.py        # Script containing the model architectures.
│   └── utils.py         # Utility script and helper functions.
├── requirements.txt     # Project dependencies.
├── configs/             # Configurations for model parameters, data paths, and experiments.
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
   git clone https://github.com/Marshall-mk/Ehealth-Tutorial
   cd Ehealth-Tutorial
   ```
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv myEhealth
   source myEhealth/bin/activate  # For Linux/Mac
   myEhealth\Scripts\activate  # For Windows
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Initial Exploration
1. **Navigate to the notebooks directory**:
   The `notebooks/` folder contains Jupyter notebooks where you can start your exploration and experimentation.
2. Open the `intructions.ipynb` to continue the experiments.

## Additional Resources
- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [Hydra Documentation](https://hydra.cc/docs/intro/)

## Summary
This lab is designed to help you gain experience in structuring data science projects properly. By completing these tasks, you'll learn how to write clean, reusable, and scalable code for real-world data science applications.

Feel free to ask questions or consult the instructor if you get stuck at any point.

---
**Happy coding!**
