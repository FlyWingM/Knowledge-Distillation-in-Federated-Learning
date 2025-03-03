# Carbon-Conscious Federated Reinforcement Learning (CCFRL)

## Overview

The **Carbon-Conscious Federated Reinforcement Learning (CCFRL) Approach** is an innovative reinforcement learning-based framework designed to optimize **carbon efficiency** and **model performance** in **Federated Learning (FL)**. Unlike conventional static or greedy client selection methods, **CCFRL dynamically balances** immediate resource needs with **long-term sustainability goals**, ensuring **energy-efficient and high-performance FL deployments**.

## Key Features

- **Adaptive Client Selection**: Uses **entropy-based allocation** and **t-tests for stagnation detection** to select optimal clients.
- **Carbon-Efficient Resource Management**: Integrates **real-time carbon intensity data** to optimize **computational efficiency**.
- **Enhanced Model Performance**: Utilizes **Principal Component Analysis (PCA)** for model similarity evaluation and adaptive exploration-exploitation transitions.
- **Dynamic Optimization**: Continuously learns from changing conditions, ensuring robust decision-making for real-world **non-IID data distributions**.

## Motivation

With increasing computational demand and energy-intensive machine learning applications, **CCFRL aims to reduce the carbon footprint of FL while maintaining high model accuracy**. Unlike existing FL frameworks that either **compromise performance for carbon efficiency** or **ignore environmental impact**, CCFRL provides a balanced and scalable approach to **sustainable AI**.

## Implementation Details

### 1. System Architecture

CCFRL consists of:
1. **State Representation**: Captures key metrics like **accuracy, model similarity, and carbon footprint**.
2. **Reward Mechanism**: Guides the RL agent towards an optimal balance of **carbon efficiency** and **model performance**.
3. **Exploration-Exploitation Strategy**: Uses **stagnation detection (t-test)** to transition between exploration and exploitation phases.
4. **Client Allocation Strategies**:
   - **Model Similarity PCA-based Client Allocation (MSPCA)**
   - **Accuracy-Prioritized Allocation (APA)**
   - **Low-Carbon Footprint Allocation (LCA)**
   - **Randomized Double Greedy Allocation (RDGA)** (baseline for comparison)

### 2. Experimental Results

#### **Performance vs. Carbon Efficiency**

| Method  | Accuracy (IID, κ=∞) | Accuracy (non-IID, κ=0.06) | CO2 Reduction | Energy Savings |
|---------|--------------------|--------------------|--------------|---------------|
| **CCFRL** | **+1.18%** | **+2.71%** | **Up to 64.23%** | **Up to 61.78%** |
| MSPCA  | +1.03%  | +2.03%  | ~42.99% | ~42.70% |
| APA    | +1.24%  | +2.29%  | ~33.64% | ~25.69% |
| LCA    | +0.15%  | +1.84%  | ~37.14% | ~11.39% |
| RDGA   | -0.37%  | -2.30%  | N/A | N/A |

### 3. How CCFRL Works

1. **Client Selection**: Identifies clients using PCA-based model similarity and **carbon-aware incentives**.
2. **Carbon-Aware Reinforcement Learning**:
   - Uses an **entropy-based reward mechanism**.
   - Employs **adaptive transitions** to avoid stagnation in training.
3. **Energy & Carbon Metrics**:
   - Computes **real-time energy consumption and emissions** based on client locations.
   - Prioritizes clients operating in **low-carbon intensity regions**.


## 4. Access
You can access the **Accepted Manuscript** directly here: ([link-to-file.pdf](https://github.com/FlyWingM/my-publications/blob/1868d0e587c6ef502794b0c3843ba1bb319a6d7c/Pioneering_Eco_Efficiency_in_Cloud_Computing__The_Carbon_Conscious_Federated_Reinforcement_Learning__CCFRL__Approach_revision_v_final_github.pdf).  
*This version has been peer-reviewed and accepted but may differ from the final published version.*
