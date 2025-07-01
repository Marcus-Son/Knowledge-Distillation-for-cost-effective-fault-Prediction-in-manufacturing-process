# Knowledge Distillation for Cost-Effective Fault Prediction in Semiconductor Manufacturing

A real-world machine learning project applying knowledge distillation and active sampling to maximize defect detection accuracy while minimizing inspection costs in highly imbalanced semiconductor process data.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background & Motivation](#2-background--motivation)
3. [Problem Statement](#3-problem-statement)
4. [Proposed Method](#4-proposed-method)
5. [Data Description](#5-data-description)
6. [Project Structure](#6-project-structure)
7. [Experimental Design & Methods](#7-experimental-design--methods)
8. [Key Results & Insights](#8-key-results--insights)
9. [My Contribution](#9-my-contribution)
10. [Publications / Presentation History](#10-publications--presentation-history)
11. [Notes / Limitations](#11-notes--limitations)
12. [Contact](#12-contact)

## 1. Project Overview

This project tackles the problem of fault (defect) prediction in semiconductor manufacturing, where rare but critical faults can result in substantial economic loss. To address the cost-accuracy trade-off in industrial inspection, we propose a machine learning pipeline using knowledge distillation and active sampling strategies. Our approach enables highly accurate defect detection using only inexpensive basic inspection data, reducing the need for costly advanced inspections.

Key features:
- End-to-end pipeline for binary classification of rare faults in real-world process data.
- Knowledge distillation transfers information from a high-cost, high-accuracy “teacher” model to a low-cost “student” model.
- Active sampling (uncertainty-based and random) to optimize inspection resource allocation.
- Extensive experiments on two real manufacturing datasets (Recipe1 & Recipe2) with severe class imbalance.

---

## 2. Background & Motivation

In semiconductor fabrication, process faults are extremely rare but have outsized impact on yield and cost. Traditional defect detection systems rely on:
- **Basic inspection:** Low-cost, fast, but low accuracy and prone to missing faults.
- **Advanced inspection:** High accuracy but expensive and resource-intensive, making it impractical to apply to all products.

![Basic vs Advanced Inspection](https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/8ec0f1ee-8902-45b8-8e3e-dbf302bc5b60)


Given the scale of manufacturing, blindly using advanced inspections on every product is economically unviable. At the same time, relying solely on basic inspection can allow costly faults to go undetected.

**Motivation:**  
To resolve this dilemma, this project leverages advanced machine learning techniques—specifically, knowledge distillation and active sampling—to maximize the predictive value of low-cost inspection data. By distilling the “knowledge” of a high-performing teacher model (trained with expensive advanced features) into a lightweight student model (trained with basic features), we can boost detection accuracy without the high operational costs.  
Furthermore, we evaluate how active sampling strategies (random and uncertainty-based) can further improve cost-effectiveness by focusing advanced inspection on the most informative or ambiguous samples.

This work directly addresses key challenges in real-world manufacturing AI: severe class imbalance, high-dimensional noisy sensor data, and the practical need to balance quality with operational cost.

## 3. Problem Statement

In semiconductor manufacturing, the majority of products are non-defective, but failing to catch rare faults can result in substantial costs and quality issues. The industry typically uses:

- **Basic Inspection** for all products: Low-cost, fast, but inaccurate for rare faults.
- **Advanced Inspection** for a small subset: High accuracy, but expensive and slow.

The current industrial standard is to select products for advanced inspection using **random sampling**—that is, products are chosen at random for expensive testing.  
However, this approach is inefficient because:
- Many advanced inspections are performed on normal (non-faulty) products, wasting resources.
- There is **no intelligent prioritization**, so valuable inspection capacity is not focused where it can have the most impact.

<p align="center">
  <img src="https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/8ec0f1ee-8902-45b8-8e3e-dbf302bc5b60" alt="Basic vs Advanced Inspection" width="500"/>
</p>
<p align="center" style="font-size:0.95em">
<b>Figure:</b> Basic inspections are applied to all, but random advanced inspection is inefficient and costly.
</p>

**Key challenge:**  
How can we maximize fault detection performance across the entire production line, while keeping inspection costs as low as possible?  
How can we make advanced inspections smarter—not just more frequent?

---

## 4. Proposed Method

Our approach is two-fold:

1. **Boost Base Model Accuracy with Knowledge Distillation (KD):**  
   - We train a high-capacity “teacher” model using the rich, advanced inspection data.
   - Through knowledge distillation, we transfer the teacher’s expertise to a “student” model that uses only basic inspection data.
   - This framework is implemented for both:
     - **Neural Network (NN) Models:** The student neural network is trained with a combination of true labels and the teacher’s soft predictions (using temperature-scaled softmax and KL-divergence loss).
     - **Non-Neural Network Models (Random Forest):**  
       The teacher Random Forest is trained on all features, while the student Random Forest uses only basic features and is trained to regress the teacher’s output probabilities (using mean squared error loss).
   - This dual-model approach demonstrates that KD is effective for both high-capacity deep learning and interpretable, industry-friendly models like Random Forests.
   - As a result, the student model—whether NN or RF—achieves higher accuracy and better fault detection using only cheap, universally-available data, effectively closing the performance gap between basic and advanced inspections.

   <p align="center">
     <img src="https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/46818e9f-d5bf-4f6f-9c9e-51091dd778ea" width="620"/><br>
     <b>Figure 1:</b> Application on active inspection framework.
   </p>

   <p align="center">
     <img src="https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/1d149d99-8b0d-407d-8cb3-f4c0f576f3bc" width="500"/><br>
     <b>Figure 2:</b> Neural Network-based student model distillation.
   </p>

   <p align="center">
     <img src="https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/1143bd10-5102-4585-b975-3136e51bcd41" width="500"/><br>
     <b>Figure 3:</b> Random Forest (Non-Neural Network)-based distillation for interpretable settings.
   </p>

2. **Optimize Advanced Inspection with Smarter Sampling:**  
   - Instead of random sampling, we experiment with multiple advanced sampling strategies—such as **uncertainty sampling**—to target advanced inspections where the model is least confident.
   - We systematically evaluate which sampling strategies, when combined with KD, yield the highest system-level efficiency across both NN and RF models.
   - Our goal is not only to reduce wasted inspections on non-faulty products, but also to enhance the *total* defect detection rate for the entire process.

**Ultimate Goal:**  
Build the most cost-effective fault detection system by  
- raising the baseline prediction power (via KD, in both neural and non-neural models), and  
- finding the sampling policy that best complements KD in a real production setting.

---

## 5. Data Description

**Datasets:**  
- Two real-world semiconductor manufacturing datasets: **Recipe1** and **Recipe2**.

**Data Structure:**  
- **Features:**  
  - *Basic inspection features*: Available for all products; cheap but limited sensitivity.
  - *Advanced inspection features*: Costly, measured only on a small subset.
- **Labels:**  
  - Binary classification (0: Normal, 1: Faulty).
  - Severe class imbalance (faults <2%).

**Key Statistics:**
- Recipe1: ~18,000 samples, 8 basic + 91 advanced features.
- Recipe2: ~245,000 samples, 8 basic + 80 advanced features.

**Industrial Realism:**
- Over 98% of products are non-faulty.
- In practice, advanced inspections are performed on a randomly chosen subset—meaning many expensive tests are wasted on normal units.
- This setup accurately reflects the real economic and operational challenges of manufacturing AI.

*Note: Raw data cannot be publicly released, but we provide a detailed EDA notebook and structure overview for transparency.*

---
## 6. Project Structure

This repository is organized as follows:
```
├── ANN/
│   ├── modules/                       # Core model and utility scripts (training, prediction, KD logic)
│   ├── seed_everything.py             # For reproducibility (sets random seeds)
│   ├── train_teacher.py               # Train NN teacher model (advanced + basic features)
│   ├── train_student.py               # Train NN student model (basic features only, no KD)
│   ├── train_kdstudent.py             # Train NN KD-student (basic features, with teacher guidance)
│   ├── split_and_predict_randomsampling.py   # Random sampling experiments (NN)
│   ├── split_and_predict_uncertainty.py      # Uncertainty-based sampling experiments (NN)
│   ├── test_predict.py                # Prediction/testing scripts (NN)
│   ├── test_predict_vm.py             # Virtual features / ablation testing
│   ├── virtual_features.py            # Synthetic/virtual feature handling
│   └── …
├── RandomForest/
│   ├── modules/                       # RF model scripts (training, prediction, KD logic)
│   ├── train_teacher_rf.py            # Train RF teacher model (advanced + basic features)
│   ├── train_student_rf.py            # Train RF student (basic only, no KD)
│   ├── train_kdstudent_rf.py          # Train RF KD-student (basic, with teacher regression supervision)
│   ├── split_and_predict_randomsampling_rf.py   # Random sampling (RF)
│   ├── split_and_predict_uncertainty_rf.py      # Uncertainty-based sampling (RF)
│   └── …
│   ├── recipe1/
│   │   └── random_forest_recipe1_experiment.ipynb   # RF experiments on Recipe1
│   ├── recipe2/
│   │   └── random_forest_recipe2_experiment.ipynb   # RF experiments on Recipe2
│   └── …
├── Dataset/
│   └── activeinspection_recipe1.mat    # Sample (or reference) dataset
├── recipe1/
│   ├── total_pytorch_recipe1_margin_copy.ipynb
│   ├── total_pytorch_recipe1_randomsampling copy.ipynb
│   ├── total_pytorch_recipe1_biased_margin copy.ipynb
│   └── …                      # NN experiments (Recipe1)
├── recipe2/
│   ├── total_pytorch_recipe2_margin.ipynb
│   ├── total_pytorch_recipe2_randomsampling.ipynb
│   ├── total_pytorch_recipe2_biased_margin copy 2.ipynb
│   └── …                      # NN experiments (Recipe2)
├── EDA_activeinspection_recipe1.ipynb   # In-depth EDA notebook
├── graph.ipynb                          # Aggregates and visualizes experimental results
├── requirements.txt                     # Python dependencies for reproduction
└── README.md                            # Project summary, usage, and documentation
```

**Notes:**
- Both `ANN/` and `RandomForest/` directories contain full pipelines for neural network and non-neural network (RF) models, including KD and sampling experiments.
- Each `recipe1/` and `recipe2/` folder contains Jupyter notebooks for detailed experiment results with various sampling strategies (margin, random, biased), for both ANN and RF models.
- All code modules include detailed comments for clarity and reproducibility.
- EDA and all key results can be found in the corresponding `.ipynb` files.

---

## 7. Experimental Design & Methods

This project uses a step-by-step experimental pipeline for both Neural Network (ANN) and Random Forest (Non-Neural Network) models:

### 1. **Training a Teacher Model**
- **Neural Network (ANN):**  
  Train a high-capacity neural network (teacher) using both basic and advanced inspection features.
- **Random Forest:**  
  Train a Random Forest teacher model on the same combined feature set.
- Both models serve as the "teacher" for knowledge distillation in their respective architectures.

### 2. **Training Student & KD-Student Models**
- **Student Model:**  
  Trained only with basic features.
    - **ANN:** Uses standard cross-entropy loss with ground truth labels.
    - **Random Forest:** Trained as a classifier (or regressor) with ground truth labels.
- **KD-Student Model:**  
  Trained with basic features, but supervised using both the true labels and the teacher’s soft predictions.
    - **ANN:** Loss combines classification (cross-entropy) loss and distillation loss (e.g., KL divergence with teacher soft labels, temperature scaling).
    - **Random Forest:** Student RF is trained to regress the teacher’s predicted probabilities (soft outputs) using mean squared error (MSE), in addition to standard classification loss.

### 3. **Advanced Inspection Sampling Strategies**
- **Random Sampling:**  
  Randomly select a subset of products for advanced inspection and use this for teacher training and model evaluation.
- **Uncertainty Sampling:**  
  Select samples where the student model (ANN or RF) is most uncertain (e.g., output probability close to 0.5), focusing expensive inspections on ambiguous cases.
- **Biased Sampling:**  
  Optionally, sample based on class balance or other heuristics to further optimize advanced inspection budget.

### 4. **Evaluation**
- All experiments are repeated for various inspection rates (cost levels) and sampling strategies, across both model types (ANN & RF).
- The main metric is **AUROC** (Area Under the Receiver Operating Characteristic Curve), robust to severe class imbalance.
- **Cost-benefit analysis:** For each strategy, compare the number of advanced inspections required vs. the achieved fault detection rate.

### 5. **Visualization & Interpretation**
- Results for both ANN and Random Forest models are aggregated and visualized (`graph.ipynb`), showing performance curves, cost-accuracy trade-offs, and sampling efficiency.
- Key insights are highlighted to support practical adoption in real manufacturing lines.

**Reproducibility:**  
- Random seeds are set for all experiments.
- Each experiment is run multiple times (e.g., 30 trials) for statistical reliability.

---

## 8. Key Results & Insights

This section summarizes the main findings from our experiments and provides an in-depth analysis of system performance under different scenarios, for both neural network (ANN) and non-neural network (Random Forest) models.

---

### 1. **Improved Basic Model Performance with Knowledge Distillation**

- Applying knowledge distillation (KD) allowed the basic inspection model—using only cheap, universally-available features—to approach the accuracy of the advanced inspection model.
- KD-student models consistently outperformed basic-only models, especially in the context of extreme class imbalance, boosting the detection rate of rare faults without requiring more expensive data for all samples.
- This demonstrates that advanced inspection “knowledge” can be effectively compressed and leveraged to make cheaper inspections smarter and more reliable, whether with deep learning or Random Forest models.

---

### 2. **Cost-Effective Inspection: Combining Basic and Targeted Advanced Inspections**

- The system achieved the best trade-off between cost and performance when combining basic inspection for all products with **targeted advanced inspection** only on selected samples, rather than using advanced inspection indiscriminately.
- Applying advanced inspection to every product is economically infeasible; our results show that selective use—driven by model feedback—delivers comparable or even superior defect detection at a fraction of the cost.
- This finding is critical for large-scale manufacturing, where reducing unnecessary advanced inspections leads directly to substantial resource savings.

---

### 3. **Sampling Strategies: Uncertainty Sampling vs. Random Sampling**

- **Uncertainty sampling**—where advanced inspections are allocated to products with the least confident predictions—consistently outperformed random sampling.
- This method prioritizes high-value inspections on ambiguous or borderline cases, dramatically improving system efficiency and boosting the overall AUROC score, especially when the inspection budget is limited.
- Random sampling, by contrast, often wastes expensive inspections on products that are almost certainly non-defective.

---

#### **Neural Network (ANN) Results**

<p align="center">
  <img src="https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/777d26fb-97c5-4629-9f55-f142a07ec89f" alt="ANN Results: AUROC vs Inspection Rate" width="700"/>
</p>
<p align="center" style="font-size:0.95em">
<b>Figure 1 (ANN):</b> AUROC as a function of advanced inspection rate and sampling strategy for the neural network-based (ANN) model.<br>
The orange line (KD + uncertainty sampling) consistently achieves higher AUROC with lower inspection cost compared to random sampling or the “all advanced inspection” baseline.
</p>

#### **Random Forest (Non-Neural Network) Results**

<p align="center">
  <img width="863" alt="image" src="https://github.com/user-attachments/assets/cd9c1177-8b1a-43da-8ffd-a62e4fee7040" />
</p>
<p align="center" style="font-size:0.95em">
<b>Figure 2 (Random Forest, Recipe1):</b> AUROC across different inspection rates and sampling strategies for the Random Forest-based (non-neural network) model on Recipe1 data.<br>
KD-student consistently outperforms the standard RF, especially when combined with uncertainty-based and biased sampling.
</p>

<p align="center">
  <img width="863" alt="image" src="https://github.com/user-attachments/assets/a696c887-f359-4d75-9b7f-e65a0490f2e6" />
</p>
<p align="center" style="font-size:0.95em">
<b>Figure 3 (Random Forest, Recipe2):</b> AUROC across different inspection rates and sampling strategies for the Random Forest-based (non-neural network) model on Recipe2 data.<br>
Cost-effective sampling and KD together yield significant gains, even under extreme class imbalance.
</p>

---

**Step-by-Step Analysis of the Figures:**

1. **X-axis:** Proportion of products receiving advanced inspection (inspection cost).
2. **Y-axis:** Average AUROC (fault prediction performance).
3. **Curves:** Each line corresponds to a strategy—random sampling, margin/uncertainty, biased margin, with and without KD.
4. **Key Insights:**
   - **Knowledge Distillation + Smarter Sampling** (KD + margin or KD + biased margin) achieves the highest AUROC at lower cost, for both ANN and RF models.
   - The benefit is especially pronounced at lower inspection rates, which is critical in high-volume manufacturing.
   - Random Forest (non-neural network) experiments confirm the framework’s generality and potential for interpretable industrial deployment.
   - Increasing inspection rate beyond a certain threshold yields diminishing returns, emphasizing the value of strategic inspection allocation.

---

**Summary:**  
By combining knowledge distillation with active, model-driven sampling strategies, our system achieves robust, scalable, and highly cost-effective fault prediction.  
This is directly applicable to real-world manufacturing, enabling factories to maximize product quality while minimizing unnecessary inspection costs—regardless of whether a deep learning or interpretable model (RF) is used.

---

## 9. My Contribution

- Led data preprocessing and exploratory data analysis (EDA).
- Designed and conducted all experiments involving advanced sampling strategies (random, uncertainty-based, biased) and knowledge distillation.
- Systematically tuned KD hyperparameters, including temperature values, and analyzed the impact on model performance.
- Compiled, visualized, and interpreted all experimental results.

---

## 10. Publications / Presentation History

- **Conference Poster Presentation:**  
- Presented a summary of this work as a poster at the Korean Institute of Industrial Engineers (대한산업공학회).

- **Journal Submission:**  
- Manuscript titled  
  _"Incorporating Knowledge Distillation into an Active Inspection Framework for Cost-Effective Fault Prediction in Manufacturing"_  
  was submitted to **The Journal of Supercomputing** (Springer) on May 25, 2025.  
  (See below for submission confirmation.)

---

## 11. Notes / Limitations

- **Data Privacy:** Due to strict confidentiality agreements, actual raw data cannot be made public. All code and experiment logic, however, are fully documented and reproducible with dummy or sample data.
- **Reproducibility:** Experiments were run in a specific computational environment. Users may need to adjust for package or hardware differences.
- **Academic Status:** The submitted manuscript is currently under review and has not yet been accepted for publication.
- **Applicability:** The methods are validated on real semiconductor process data; generalization to other industries may require further adaptation.

---

## 12. Contact

- **Email:** shawn22587@gmail.com

Feel free to contact me for questions regarding this project or potential collaborations.
