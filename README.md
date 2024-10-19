# Knowledge Distillation for Cost-Effective Fault Prediction in Manufacturing Process

## Table of Contents
1. [Background](#background)
2. [Problems](#problems)
3. [Proposed Method](#proposed-method)
4. [Goals](#goals)
5. [Data Description](#data-description)
6. [Experiment](#experiment)
7. [Results](#results)

## Background
In manufacturing, predicting product conditions early is essential to reduce repair and replacement costs. This process involves:
- **Basic Inspections**: Performed on all products to check general quality and functionality.
- **Advanced Inspections**: Performed selectively for in-depth quality verification but incurs higher costs.

Balancing these two inspection methods is crucial, as basic inspections are cost-effective but may miss some defects, while advanced inspections, though more accurate, are costlier.

![Basic vs Advanced Inspection](https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/8ec0f1ee-8902-45b8-8e3e-dbf302bc5b60)

## Problems
- **Cost and time inefficiencies**: Performing advanced inspections on every product is costly and time-consuming.
- **Defect prediction accuracy**: Basic inspections may not capture all defects.
- **Balancing inspection types**: Determining which products require advanced inspection is challenging without predictive data.

## Proposed Method
We propose using **Knowledge Distillation**, where a more advanced model transfers knowledge to a basic model. The base model is trained to predict faults using both basic and advanced inspection data, improving defect prediction accuracy with lower inspection costs.

![Knowledge Distillation](https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/46818e9f-d5bf-4f6f-9c9e-51091dd778ea)

- **Distilled Basic Model (Neural Network)**: Shows how the basic model learns from the advanced model.
  ![Neural Network](https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/1d149d99-8b0d-407d-8cb3-f4c0f576f3bc)

- **Distilled Basic Model (Non-Neural Network)**: Illustrates non-neural network approach.
  ![Non-Neural Network](https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/1143bd10-5102-4585-b975-3136e51bcd41)

## Goals
- **Developing a better-performing base model**: Leverage knowledge distillation to transfer knowledge from advanced models to the base model, thereby improving the performance of the base model.
- **Cost-Effective Defect Prediction**: Minimize costly advanced inspections while maintaining high prediction accuracy.
- **Optimize Advanced Inspection Sampling**: Perform advanced inspections only on selected products using efficient sampling methods.

## Data Description
- In semiconductor manufacturing, wafers are inspected throughout the process. The dataset can predict whether a wafer will be defective before the end of production.
  
  ![Wafer Inspection](https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/78f9db95-a31f-46fe-a0bb-6c7bff16b1af)

## Experiment
- **Data Split**: Train/test data split 50:50.
- **Models**: Artificial Neural Networks (ANN) and Random Forest.
- **Sampling**:
  - **Random Sampling**: Selecting products randomly for advanced inspection.
  - **Uncertainty Sampling**: Prioritizing products with uncertain defect predictions.
- **Metrics**: AUROC was used to evaluate model performance.
- **Iterations**: 30 iterations per experiment.

## Results
- **Improved Basic Model Performance**: Knowledge distillation significantly enhanced the basic model's prediction accuracy.
- **Cost-Effective Inspection**: A combination of basic and advanced inspections outperformed applying advanced inspection to all products.
- **Sampling Strategies**: Uncertainty sampling showed better results compared to random sampling.

![Results](https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/777d26fb-97c5-4629-9f55-f142a07ec89f)
