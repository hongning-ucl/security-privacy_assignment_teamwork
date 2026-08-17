# ML-Powered Click Fraud Detection & Risk-Control System

[中文](#中文版) | [English](#english-version)

---

# 中文版

## 项目简介

本项目实现了一个面向数字广告平台的 **机器学习驱动点击欺诈检测与风险控制系统**。

在 CPC（Cost-Per-Click，按点击付费）广告模式下，恶意用户或自动化 Bot 可以通过持续产生虚假点击来消耗广告主预算，同时污染后续用于分析和模型训练的行为数据。

与单纯进行欺诈分类不同，本项目关注完整的 AI 系统流程，将：

**流量模拟 → 实时过滤 → 行为特征提取 → 机器学习异常检测 → 风险评分 → 业务决策 → 结果监控**

连接为一个端到端系统。

系统采用三层防御架构，在检测异常行为的同时，将模型输出进一步转化为实际的计费和预算控制动作。

---

## 系统架构

```text
Incoming Click Traffic
        │
        ▼
┌─────────────────────────────┐
│ Layer 1                     │
│ 实时流量过滤                 │
│                             │
│ Token Bucket                │
│ Rate Limiting               │
│ Adaptive State Machine      │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Layer 2                     │
│ 行为异常检测                 │
│                             │
│ Feature Engineering         │
│ Rule-Based Score            │
│ Isolation Forest            │
│ Fraud Risk Score            │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Layer 3                     │
│ 预算风险控制                 │
│                             │
│ Charge                      │
│ Withhold                    │
│ Campaign Cap                │
│ Block                       │
└─────────────┬───────────────┘
              │
              ▼
       Logging & Monitoring
```

这种分层设计避免对所有流量执行相同复杂度的分析。

明显异常的高频流量首先通过轻量级规则进行过滤，只有通过第一层的流量才进一步进入行为分析和机器学习检测模块。

---

## 攻击模拟

项目通过 Python 构建合成点击数据，模拟三种不同类型的广告流量。

### 正常流量

用于模拟正常用户行为，包括：

* 不同设备和 IP 地址
* 相对随机的点击间隔
* 正常的页面停留时间
* 较低的单一广告点击集中度

### Competitive Click Fraud

模拟恶意用户主动点击竞争对手广告，从而消耗其广告预算。

典型特征包括：

* 对目标广告进行重复点击
* 较高的点击频率
* 较短的停留时间
* 更明显的重复行为模式

### Botnet Click Fraud

模拟脚本或 Bot 网络产生的大规模自动化点击攻击。

主要特征包括：

* 极高点击频率
* 极短点击间隔
* 接近即时离开的停留行为
* 大量重复点击目标广告
* 部分复用真实用户 IP 以模拟正常流量

---

## Layer 1：实时流量过滤

第一层负责在流量进入机器学习模块之前过滤明显异常行为。

### Token Bucket

系统针对 IP 地址实现 Token Bucket 限流机制，用于限制短时间内允许产生的点击数量。

当请求频率超过允许范围时，系统可以提前阻止异常流量进入后续处理流程。

### 自适应状态机

系统维护三种 IP 状态：

```text
Trusted → Challenged → Blocked
```

当某个 IP 表现出异常行为时，可以从正常状态进入 Challenge 状态。

如果持续产生高风险行为，则进一步进入 Blocked 状态。

这一层主要承担低成本、实时的初步风险控制。

---

## Layer 2：机器学习行为异常检测

第二层是系统的核心机器学习模块。

系统首先将原始点击记录聚合为 IP 级行为特征。

主要特征包括：

* Click Count
* Click Rate
* Inter-click Interval
* Average Dwell Time
* Same-ad Concentration
* Device-level Activity
* Time Since Last Click
* Challenge History

随后使用两种方式计算风险。

### Rule-Based Risk Score

基于可解释行为规则生成风险评分，使最终结果能够被人工检查和审计。

### Isolation Forest

使用 **Isolation Forest** 对用户行为进行无监督异常检测，从而识别与正常行为模式存在明显差异的 IP。

最终风险分数融合规则评分和机器学习异常评分：

```text
Fraud Score =
0.55 × Rule-Based Score
+
0.45 × Isolation Forest Score
```

规则模块拥有稍高权重，从而在异常检测能力与可解释性之间取得平衡。

最终 Fraud Score 会进一步被映射为不同风险等级，并传递至下一层决策系统。

---

## Layer 3：预算风险控制

模型检测到异常行为之后，并不会仅输出一个 Fraud / Normal 标签。

系统进一步将风险评分转化为真实的业务控制动作。

包括：

* **Charge**：正常计费
* **Withhold**：暂缓或取消计费
* **Campaign Cap**：触发广告活动级预算限制
* **Block**：直接阻止高风险流量

因此系统形成了：

```text
Model Prediction
        ↓
Risk Assessment
        ↓
Decision Policy
        ↓
Business Action
```

的完整闭环。

---

## 实验结果

系统分别在 Normal Traffic、Competitive Fraud 和 Bot Fraud 条件下进行了模拟测试。

| Traffic Type      | Mitigation Rate |
| ----------------- | --------------: |
| Bot Fraud         |       **83.3%** |
| Competitive Fraud |       **77.8%** |
| Normal Traffic    |       **21.2%** |

Mitigation Rate 定义为：

```text
(Blocked + Cap Blocked + Withheld) / Total Clicks
```

实验结果表明，系统能够显著降低 Bot Fraud 和 Competitive Fraud 对广告预算造成的影响。

同时，Normal Traffic 中仍存在 **21.2%** 的流量受到风险控制影响，这也暴露出系统设计中的重要权衡：

> 更激进的风险控制能够提高欺诈拦截能力，但同时可能增加正常用户被误处理的概率。

---

## Precision–Recall Trade-off

项目进一步分析了不同风险阈值对模型行为的影响。

仅将 High-Risk IP 视为欺诈时：

```text
Precision ≈ 0.83
Recall    ≈ 0.50
```

同时将 Medium-Risk 和 High-Risk IP 视为欺诈时：

```text
Precision ≈ 0.44
Recall    ≈ 0.70
```

可以看到，降低决策阈值虽然能够发现更多欺诈行为，但同时会带来更高的 False Positive。

因此在真实 AI 系统中，阈值选择不仅是模型性能问题，也需要综合考虑不同错误所对应的实际业务成本。

---

## 可视化与监控

项目提供了用于观察攻击过程和防御结果的可视化结果，包括：

* Click Frequency
* Click Interval Distribution
* Fraud Risk Scores
* Defense Action Distribution
* Budget Consumption
* Attack vs Defense Comparison

同时实现了 Web Dashboard，用于展示：

* 实时点击流量
* Fraud Clicks
* Remaining Budget
* Fraud Risk Scores
* Layer-level Defense Results
* Billing Actions

通过可视化界面，可以直接观察攻击行为如何经过不同防御层，并最终影响预算和计费结果。

---

## 项目结构

```text
.
├── attack.py
├── defense.py
├── main.py
├── environment.yml
│
├── fore_end/
│   └── adsarmor_integrated.html
│
└── outputs_full_pipeline/
    ├── attack_raw_events.csv
    ├── attack_processed_features.csv
    ├── layer1_full_audit.csv
    ├── layer2_ip_results.csv
    ├── defense_final_results.csv
    ├── summary_defense_actions.csv
    ├── summary_by_traffic_type.csv
    └── visualisation outputs
```

### `attack.py`

负责：

* 正常流量生成
* Competitive Fraud 模拟
* Botnet Fraud 模拟
* 点击事件生成
* 行为特征计算
* 攻击统计与预算分析

### `defense.py`

负责：

* Token Bucket
* IP 状态管理
* 行为特征聚合
* Rule-Based Fraud Score
* Isolation Forest
* Fraud Risk Score
* Budget-aware Decision Policy
* 实验结果统计

### `main.py`

负责串联完整流程：

```text
Attack Simulation
        ↓
Traffic Processing
        ↓
Layer 1
        ↓
Layer 2
        ↓
Layer 3
        ↓
Evaluation
        ↓
Visualisation
```

---

## 技术栈

### Machine Learning

* Python
* Scikit-learn
* Isolation Forest

### Data Processing

* Pandas
* NumPy

### Visualisation

* Matplotlib

### Frontend

* HTML
* CSS
* JavaScript

### System Design

* Token Bucket Rate Limiting
* Adaptive State Machine
* Behavioural Feature Engineering
* Anomaly Detection
* Risk Scoring
* Budget-Aware Decision Control
* Audit Logging

---

## 运行项目

创建环境：

```bash
conda env create -f environment.yml
```

激活环境：

```bash
conda activate daps-hackathon-1
```

运行完整系统：

```bash
python main.py
```

实验输出将保存在：

```text
outputs_full_pipeline/
```

目录中。

---

## 项目核心思路

这个项目不仅关注单独的机器学习模型，而更加关注模型如何成为完整智能系统的一部分：

```text
Data
 ↓
Feature Engineering
 ↓
Model Inference
 ↓
Risk Score
 ↓
Decision Policy
 ↓
System Action
 ↓
Evaluation & Monitoring
```

项目体现了几个重要的 AI 系统设计问题：

* 如何组合规则系统与机器学习模型
* 如何将模型输出映射为真实系统行为
* 如何权衡 Precision、Recall 与业务风险
* 如何减少模型误判对正常用户的影响
* 如何设计可解释、可审计的 AI 决策流程
* 如何拆分实时推理与离线模型训练

这些设计思想同样可以应用于更广泛的 AI Application、AI System 和智能决策系统。

---

# English Version

## Overview

This project implements an **ML-powered click fraud detection and risk-control system** for digital advertising platforms.

Under the Cost-Per-Click (CPC) advertising model, advertisers are charged whenever users interact with their advertisements. This creates an inherent vulnerability: malicious users or automated bot networks can generate fraudulent clicks to exhaust advertiser budgets and contaminate behavioural data used by downstream analytical and machine-learning systems.

Rather than treating fraud detection as an isolated classification task, this project builds an end-to-end pipeline connecting:

**traffic simulation → real-time filtering → behavioural feature engineering → machine-learning detection → risk scoring → operational decision making → monitoring**

The system adopts a three-layer architecture that converts model predictions into concrete billing and budget-control actions.

---

## System Architecture

```text
Incoming Click Traffic
        │
        ▼
┌─────────────────────────────┐
│ Layer 1                     │
│ Real-Time Filtering         │
│                             │
│ Token Bucket                │
│ Rate Limiting               │
│ Adaptive State Machine      │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Layer 2                     │
│ Behavioural ML Detection    │
│                             │
│ Feature Engineering         │
│ Rule-Based Score            │
│ Isolation Forest            │
│ Fraud Risk Score            │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Layer 3                     │
│ Budget-Aware Risk Control   │
│                             │
│ Charge                      │
│ Withhold                    │
│ Campaign Cap                │
│ Block                       │
└─────────────┬───────────────┘
              │
              ▼
       Logging & Monitoring
```

The layered architecture prevents expensive analysis from being applied uniformly to every incoming request.

Clearly abnormal high-frequency traffic is handled by lightweight filtering first, while more detailed behavioural analysis is performed only when required.

---

## Attack Simulation

The project includes a Python-based synthetic traffic generator covering three major traffic types.

### Normal Traffic

Legitimate behaviour is simulated using:

* diverse IP addresses and devices
* irregular click intervals
* realistic dwell times
* relatively low advertisement concentration

### Competitive Click Fraud

Competitive fraud represents malicious users deliberately clicking competitors' advertisements to consume their advertising budgets.

Typical characteristics include:

* repeated target-ad interactions
* elevated click frequency
* shorter dwell times
* increasingly repetitive behavioural patterns

### Botnet Click Fraud

Botnet traffic represents highly automated and scalable click attacks.

Characteristics include:

* extremely high click frequency
* very short inter-click intervals
* near-instant departure
* repeated target-ad interactions
* partial reuse of legitimate IP addresses to imitate normal users

---

## Layer 1 — Real-Time Traffic Filtering

Layer 1 performs lightweight traffic filtering before requests reach the machine-learning stage.

### Token Bucket

A Token Bucket mechanism controls the permitted click rate for individual IP addresses.

Requests exceeding the configured rate limits can therefore be restricted before entering more computationally expensive stages.

### Adaptive State Machine

Traffic sources can transition between:

```text
Trusted → Challenged → Blocked
```

Suspicious behaviour escalates an IP into the challenge state, while repeated violations can result in hard blocking.

This layer provides a low-cost first line of defence against obvious high-frequency abuse.

---

## Layer 2 — Behavioural Machine Learning Detection

Layer 2 forms the core machine-learning component of the system.

Raw click events are aggregated into IP-level behavioural features, including:

* click count
* click rate
* inter-click interval statistics
* average dwell time
* same-ad concentration
* device-level activity
* time since the previous click
* challenge history

Two complementary detection mechanisms are then used.

### Rule-Based Risk Score

An interpretable rule-based component evaluates suspicious behavioural patterns.

This makes the resulting decisions easier to review and audit.

### Isolation Forest

An **Isolation Forest** model performs unsupervised behavioural anomaly detection.

The anomaly score is combined with the rule-based component:

```text
Fraud Score =
0.55 × Rule-Based Score
+
0.45 × Isolation Forest Score
```

The slightly higher weight assigned to the rule-based component maintains interpretability while still benefiting from machine-learning-based anomaly detection.

The resulting fraud score is mapped to operational risk levels and passed to the final control layer.

---

## Layer 3 — Budget-Aware Risk Control

Fraud detection alone does not directly protect advertisers.

Layer 3 therefore converts risk predictions into concrete operational actions.

Possible actions include:

* **Charge**
* **Withhold Billing**
* **Campaign Cap**
* **Block**

The full decision flow becomes:

```text
Model Prediction
        ↓
Risk Assessment
        ↓
Decision Policy
        ↓
Business Action
```

This allows the system to move beyond passive fraud classification and directly influence billing outcomes.

---

## Experimental Results

The system was evaluated under simulated normal, competitive-fraud, and bot-fraud traffic.

| Traffic Type      | Mitigation Rate |
| ----------------- | --------------: |
| Bot Fraud         |       **83.3%** |
| Competitive Fraud |       **77.8%** |
| Normal Traffic    |       **21.2%** |

Mitigation Rate is defined as:

```text
(Blocked + Cap Blocked + Withheld) / Total Clicks
```

The results show that the system substantially reduces the financial impact of malicious traffic.

At the same time, the **21.2% mitigation rate for normal traffic** demonstrates an important system-level trade-off: aggressive protection can reduce fraud losses but may also affect legitimate users.

---

## Precision–Recall Trade-off

When only high-risk IPs are classified as suspicious:

```text
Precision ≈ 0.83
Recall    ≈ 0.50
```

When both medium- and high-risk IPs are considered suspicious:

```text
Precision ≈ 0.44
Recall    ≈ 0.70
```

This demonstrates that lower decision thresholds improve fraud coverage but increase false positives.

In practical AI systems, threshold selection is therefore not purely a model-optimisation problem. It must also consider the operational costs associated with false positives and false negatives.

---

## Visualisation and Monitoring

The project generates visual analytics for both attack behaviour and defence outcomes, including:

* click frequency
* click interval distribution
* fraud risk scores
* defence action distribution
* budget consumption
* attack-versus-defence comparison

An interactive Web dashboard is also provided to visualise:

* traffic activity
* fraud clicks
* remaining advertiser budget
* fraud risk scores
* layer-level defence outputs
* final billing actions

This provides a system-level view of how incoming attacks propagate through the defence pipeline.

---

## Project Structure

```text
.
├── attack.py
├── defense.py
├── main.py
├── environment.yml
│
├── fore_end/
│   └── adsarmor_integrated.html
│
└── outputs_full_pipeline/
    ├── attack_raw_events.csv
    ├── attack_processed_features.csv
    ├── layer1_full_audit.csv
    ├── layer2_ip_results.csv
    ├── defense_final_results.csv
    ├── summary_defense_actions.csv
    ├── summary_by_traffic_type.csv
    └── visualisation outputs
```

### `attack.py`

Implements:

* normal traffic generation
* competitive-fraud simulation
* botnet-fraud simulation
* raw click-event generation
* behavioural feature construction
* attack statistics
* budget-consumption analysis

### `defense.py`

Implements:

* Token Bucket filtering
* adaptive traffic states
* behavioural feature aggregation
* rule-based fraud scoring
* Isolation Forest anomaly detection
* unified fraud-risk scoring
* budget-aware control
* experimental evaluation

### `main.py`

Orchestrates the complete pipeline:

```text
Attack Simulation
        ↓
Traffic Processing
        ↓
Layer 1
        ↓
Layer 2
        ↓
Layer 3
        ↓
Evaluation
        ↓
Visualisation
```

---

## Tech Stack

### Machine Learning

* Python
* Scikit-learn
* Isolation Forest

### Data Processing

* Pandas
* NumPy

### Visualisation

* Matplotlib

### Frontend

* HTML
* CSS
* JavaScript

### System Design

* Token Bucket Rate Limiting
* Adaptive State Machine
* Behavioural Feature Engineering
* Anomaly Detection
* Risk Scoring
* Budget-Aware Decision Control
* Audit Logging

---

## Getting Started

Create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate daps-hackathon-1
```

Run the complete pipeline:

```bash
python main.py
```

Generated experiment results will be saved under:

```text
outputs_full_pipeline/
```

---

## Key Takeaways

This project focuses not only on an individual machine-learning model, but on how ML inference operates as part of a complete intelligent system:

```text
Data
 ↓
Feature Engineering
 ↓
Model Inference
 ↓
Risk Score
 ↓
Decision Policy
 ↓
System Action
 ↓
Evaluation & Monitoring
```

The project explores several broader AI-system engineering challenges:

* combining deterministic rules with machine learning
* converting model outputs into operational decisions
* balancing precision, recall, and business risk
* controlling false-positive impact on legitimate users
* maintaining interpretable and auditable decisions
* separating real-time inference from offline learning

These design principles are transferable to broader **AI systems, AI applications, risk-control systems, and model-driven decision pipelines**.
