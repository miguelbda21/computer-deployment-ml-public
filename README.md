# Computer Deployment Data Engineering Pipeline

## Overview

This project demonstrates a structured data engineering pipeline for predicting computer deployment timelines. The repository emphasizes modular architecture, reproducibility, validation, and maintainability using Python and SQL.

The primary focus of this project is on pipeline design and engineering best practices rather than model complexity.

---

## Architecture Overview

Data Source  
→ SQL Extraction  
→ Data Validation  
→ Preprocessing & Feature Engineering  
→ Model Training  
→ Output Reports  

This repository is structured to support:

- Modular pipeline development
- Clear separation of raw, processed, and model artifacts
- Reproducible workflows
- Environment-based configuration
- Version control discipline
- Testable components

---

## Project Structure
├── notebooks/ # Exploratory analysis and experimentation
├── reports/ # Generated reports and outputs
├── src/ # Core pipeline logic (data ingestion, preprocessing, modeling)
├── tests/ # Unit and pipeline validation tests
├── requirements.txt # Project dependencies
├── .gitignore # Files excluded from version control
└── README.md # Project documentation


---

## Engineering Concepts Demonstrated

- Modular data pipeline architecture
- Structured project organization
- Data validation prior to model execution
- Separation of concerns
- Dependency management via requirements.txt
- Git-based version control workflow
- Reproducible execution design

---

## Technologies Used

- Python
- SQL
- Pandas
- Scikit-learn
- Git / GitHub

---

## How to Run

Clone the repository:

git clone https://github.com/miguelbda21/computer-deployment-ml-public.git

Install dependencies:

pip install -r requirements.txt

Run the pipeline:

python src/main.py

---

## Purpose

This repository serves as a template for structured data engineering and machine learning workflows, emphasizing scalability, maintainability, and clean architecture principles.
