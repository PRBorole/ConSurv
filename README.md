<div align="center">
  <img height="200" alt="ConSurv" src="./figures/Logo.png/">
</div>

<div align="center">
  <em>Concept Bottleneck Model for Survival Analysis in Breast Cancer.</em>
</div>

 <p align="center">
  <a href="#Introduction">Introduction</a> •
  <a href="#Requirements">Requirements</a> •
  <a href="#Directory-Structure">Directory Structure</a> •
  <a href="#Usage">Usage</a> •
  <a href="#license">License</a> 
</p>

---
## Introduction
Survival analysis refers to statistical procedures used to analyze data that focuses on the time until an event occurs, such as death in cancer patients. Traditionally, the linear Cox Proportional Hazards (CPH) model is widely used due to its inherent interpretability. CPH model help identify key disease-associated factors (through feature weights), providing insights into patient risk of death. However, their reliance on linear assumptions limits their ability to capture the complex, non-linear relationships present in real-world data. To overcome this, more advanced models, such as neural networks, have been introduced, offering significantly improved predictive accuracy. However, these gains come at the expense of interpretability, which is essential for clinical trust and practical application. To address the trade-off between predictive accuracy and interpretability in survival analysis, we propose ConSurv, a concept bottleneck model that maintains state-of-the-art performance while providing transparent and interpretable insights. Using gene expression and clinical data from breast cancer patients, ConSurv captures complex feature interactions and predicts patient risk. By offering clear, biologically meaningful explanations for each prediction, ConSurv attempts to build trust among clinicians and researchers in using the model for informed decision-making.


---
## Requirements

Python version - 3.9

See <a href="./requirements.txt"> requirements.txt</a> for the list of python packages.


---
## Directory Structure

| File/Folder | Description |
| ----------- | ----------- |
| Data | Folder containing data needed for project |
| Figures | Folder containing result figures for paper |
| Hyperopts | Folder containing configs, code and results for hyperparameter tuning |
| Notebooks | Folder containing experiments ipynbs for the paper|
| src | Folder containing  code needed for COnSurv|


```bash
tree -L 3

```
---
## Usage
