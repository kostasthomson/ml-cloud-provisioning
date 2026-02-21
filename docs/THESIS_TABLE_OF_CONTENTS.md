# Πίνακας Περιεχομένων Διπλωματικής Εργασίας
# Deep Reinforcement Learning for Energy-Efficient Resource Provisioning in Heterogeneous Cloud Environments

---

## **1. Εισαγωγή (Introduction)**

### 1.1 Πρόβλημα – Σημαντικότητα του θέματος (Problem – Significance)
- Ενεργειακή κατανάλωση κέντρων δεδομένων (Data center energy consumption - 1-2% of global electricity ~200 TWh/year)
- Αύξηση φόρτων εργασίας HPC (Growth of HPC workloads: AI/ML training, scientific computing)
- Πολυπλοκότητα ετερογενούς υλικού (Complexity of heterogeneous hardware: CPU, GPU, FPGA, MIC)
- Περιορισμοί στατικών ευρετικών μεθόδων (Limitations of static heuristics)

### 1.2 Σκοπός – Στόχοι (Purpose – Goals)
- Ανάπτυξη συστήματος ενισχυτικής μάθησης για ενεργειακά-αποδοτική κατανομή πόρων
- Ελαχιστοποίηση κατανάλωσης ενέργειας με τήρηση SLA
- Δημιουργία infrastructure-agnostic μοντέλου
- Σύγκριση με παραδοσιακές μεθόδους (heuristics, scoring allocator)

### 1.3 Ερωτήματα – Υποθέσεις (Research Questions – Hypotheses)
- **RQ1**: Μπορεί η ενισχυτική μάθηση να βελτιστοποιήσει αποφάσεις κατανομής πόρων;
- **RQ2**: Πώς επηρεάζουν τα χαρακτηριστικά κατάστασης (state features) την απόδοση;
- **RQ3**: Μπορεί το μοντέλο να γενικευθεί σε διαφορετικά περιβάλλοντα χωρητικότητας;
- **H1**: PPO θα επιτύχει υψηλότερο ποσοστό αποδοχής από heuristics
- **H2**: Domain randomization βελτιώνει τη γενίκευση

### 1.4 Συνεισφορά (Contribution)
- Νέα αναπαράσταση κατάστασης ειδικά για ετερογενή cloud (v3 encoder με capacity features)
- Πολυ-στοχική συνάρτηση ανταμοιβής (energy + SLA + acceptance)
- Action masking για εγγυημένα έγκυρες αποφάσεις
- Πειραματική αξιολόγηση σε 5 προρυθμίσεις χωρητικότητας
- Ολοκληρωμένο σύστημα: Simulator ↔ RL Agent ↔ REST API

### 1.5 Βασική Ορολογία (Basic Terminology)
- Reinforcement Learning (RL), Markov Decision Process (MDP)
- Proximal Policy Optimization (PPO)
- Hardware Types (CPU, GPU, DFE/FPGA, MIC)
- Service Level Agreement (SLA)
- Domain Randomization, Curriculum Learning
- State Encoder, Action Masking

### 1.6 Διάρθρωση της μελέτης (Thesis Structure)

---

## **2. Βιβλιογραφική Επισκόπηση – Θεωρητικό Υπόβαθρο (Literature Review – Theoretical Background)**

### 2.1 Υπολογιστικό Νέφος και Διαχείριση Πόρων (Cloud Computing & Resource Management)
- Ετερογενή περιβάλλοντα cloud (CPU, GPU, FPGA, accelerators)
- Παραδοσιακές μέθοδοι δρομολόγησης (First-Fit, Best-Fit, Round-Robin)
- CloudLightning project και αρχιτεκτονική SOSM Broker

### 2.2 Ενεργειακή Αποδοτικότητα σε Κέντρα Δεδομένων (Energy Efficiency in Data Centers)
- Μοντέλα ενεργειακής κατανάλωσης: E = (P_idle + util × (P_max - P_idle)) × duration
- Green computing και περιβαλλοντικός αντίκτυπος
- Τεχνικές βελτιστοποίησης ενέργειας

### 2.3 Ενισχυτική Μάθηση (Reinforcement Learning)
- Markov Decision Process (MDP): (S, A, P, R, γ)
- Αλγόριθμοι Policy Gradient
- Proximal Policy Optimization (PPO) - Schulman et al., 2017
- Actor-Critic αρχιτεκτονικές
- Generalized Advantage Estimation (GAE)

### 2.4 Μηχανική Μάθηση για Διαχείριση Πόρων Cloud (ML for Cloud Resource Management)
- Εποπτευόμενη μάθηση για πρόβλεψη φόρτου
- RL για δυναμική κατανομή πόρων
- Σύγκριση RL με βελτιστοποίηση (ILP, MILP)

### 2.5 Ανάλυση Κενού και Τοποθέτηση Έρευνας (Gap Analysis)
- Υφιστάμενες προσεγγίσεις: περιορισμοί
- Καινοτομία παρούσας εργασίας

---

## **3. Μεθοδολογία (Methodology)**

### 3.1 Διατύπωση Προβλήματος ως MDP (MDP Formulation)
- Ορισμός χώρου καταστάσεων (State Space)
- Ορισμός χώρου ενεργειών (Action Space)
- Συνάρτηση μετάβασης (Transition Function)
- Συνάρτηση ανταμοιβής (Reward Function)

### 3.2 Σχεδιασμός Χώρου Καταστάσεων (State Space Design)

#### 3.2.1 Task Features (12 dimensions)
- num_vms, vcpus_per_vm, memory_per_vm, instructions
- Task compatibility flags, deadline info

#### 3.2.2 Hardware Type Features (N × 16 dimensions)
- Utilization metrics (CPU, memory, storage, network, accelerator)
- Capacity ratios, power model parameters
- Compute capabilities

#### 3.2.3 Global Features (5 dimensions)
- Total power consumption, queue length, acceptance rate

#### 3.2.4 Scarcity Features - v2 (5 dimensions)
- Average utilization, min capacity ratios, scarcity indicator

#### 3.2.5 Capacity Features - v3 (6 dimensions) - *Novel Contribution*
- System scale normalization, task fit ratios, scale bucket

### 3.3 Σχεδιασμός Χώρου Ενεργειών (Action Space Design)
- N+1 διακριτές ενέργειες (N hardware types + reject)
- Action masking για εγκυρότητα αποφάσεων

### 3.4 Συνάρτηση Ανταμοιβής (Reward Function Design)
- R_energy: Ενεργειακή απόδοση
- R_sla: Συμμόρφωση με SLA
- R_acceptance: Μπόνους/ποινή αποδοχής/απόρριψης
- Weighted combination: R = w_e × R_energy + w_s × R_sla + R_acceptance

### 3.5 Αλγόριθμος PPO (PPO Algorithm)
- Clipped objective function
- Actor-Critic αρχιτεκτονική νευρωνικού δικτύου
- Hyperparameters (learning rate, gamma, clip range, epochs)

### 3.6 Τεχνικές Εκπαίδευσης (Training Techniques)

#### 3.6.1 Domain Randomization
- Presets: small, medium, large, high_load, stress_test, enterprise
- Mixed capacity training

#### 3.6.2 Curriculum Learning
- Εξέλιξη δυσκολίας κατά την εκπαίδευση

#### 3.6.3 Distributed Training (Multi-GPU)
- PyTorch DDP με torchrun
- Vectorized environments

---

## **4. Υλοποίηση Συστήματος (System Implementation)**

### 4.1 Αρχιτεκτονική Συστήματος (System Architecture)
```
C++ Simulator → CSV logs → Python merger → Training → Model → REST API
     ↑                                                              ↓
     └────────────────── rlBroker HTTP calls ───────────────────────┘
```

### 4.2 CloudLightning Simulator (C++)
- Discrete-event simulation
- SOSM Broker integration
- CSV decision logging (25 columns)

### 4.3 RL Module (Python)

#### 4.3.1 State Encoder
- v1 (17-dim), v2 (22-dim), v3 (28-dim) εξέλιξη

#### 4.3.2 Policy Network
- Input → Linear(128) → ReLU → Actor/Critic heads
- BatchNorm, Dropout regularization

#### 4.3.3 Environment Wrapper
- CloudProvisioningEnv implementation
- Simulated infrastructure generation

#### 4.3.4 Trainer
- PPO training loop
- Distributed training support

### 4.4 REST API (FastAPI)
- /rl/predict - Inference endpoint
- /rl/training - Training management
- /rl/model - Model management

### 4.5 Baseline Allocators
- Scoring Allocator (multi-objective weighted)
- Random Allocator
- Heuristic methods

---

## **5. Αποτελέσματα (Results)**

### 5.1 Πειραματική Διάταξη (Experimental Setup)
- Environment presets: small, medium, large, high_load, stress_test
- Training configuration: timesteps, GPUs, hyperparameters
- Evaluation metrics definitions

### 5.2 Μετρικές Αξιολόγησης (Evaluation Metrics)

| Μετρική | Ορισμός |
|---------|---------|
| Acceptance Rate | accepted / total_tasks |
| Policy Rejection % | policy_rejections / total_rejections |
| Capacity Rejection Ratio | capacity_rejections / total_rejections |
| Energy per Task | total_energy_kwh / accepted_tasks |
| SLA Compliance | tasks meeting deadline / completed tasks |

### 5.3 Εξέλιξη Εκδόσεων Μοντέλου (Model Version Evolution)

| Version | Key Change | Avg Acceptance | vs Baseline |
|---------|------------|----------------|-------------|
| V4 | Domain randomization | 37.42% | Baseline |
| V5 | Scarcity-aware rewards | 35.25% | -5.8% |
| V6 | Gentler scaling | 37.44% | +0.0% |
| V7 | Capacity features (v3) | 37.82% | **+1.4%** |

### 5.4 Σύγκριση με Baselines (Comparison with Baselines)
- PPO vs Scoring Allocator
- PPO vs Random
- PPO vs Heuristics (First-Fit, Round-Robin)

### 5.5 Ανάλυση Χρησιμοποίησης Πόρων (Resource Utilization Analysis)
- CPU/GPU/Memory utilization per preset
- Rejection patterns analysis
- Energy efficiency per hardware type

### 5.6 Στατιστική Ανάλυση (Statistical Analysis)
- Welch's t-test for significance
- Cohen's d effect size
- Confidence intervals

### 5.7 Διαγνωστικά State Vector (State Vector Diagnostics)
- Task fit ratios ανά preset
- Scale blindness problem και λύση

---

## **6. Επίλογος (Epilogue)**

### 6.1 Σύνοψη και Συμπεράσματα (Summary and Conclusions)
- Επιτυχής εφαρμογή PPO για cloud provisioning
- V3 capacity features βελτίωσαν γενίκευση (+1.4%)
- Domain randomization κρίσιμο για robustness
- Action masking εξασφαλίζει έγκυρες αποφάσεις

### 6.2 Όρια και Περιορισμοί της Έρευνας (Limitations)
- Simulated environment (όχι real-world deployment)
- Fixed task distributions
- Single-cell evaluation (όχι multi-cell coordination)
- Limited hardware type variety (4 types)

### 6.3 Μελλοντικές Επεκτάσεις (Future Work)
- Multi-agent RL για πολλαπλά cells
- Offline RL από ιστορικά logs
- Model ensemble για robustness
- Real-world deployment και validation
- Continuous action space για finer control
- Curriculum learning refinement

---

## **Παράρτημα A - Τεχνικά Στοιχεία (Technical Appendix)**

### A.1 Hyperparameters Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 - 3e-4 |
| Gamma | 0.99 |
| GAE Lambda | 0.95 |
| Clip Range | 0.2 |
| PPO Epochs | 10 |
| Batch Size | 64 |

### A.2 CSV Format (25 columns)
```
num_vms,cpu_req,mem_req,util_cpu_before,util_mem_before,
avail_cpu_before,avail_mem_before,avail_storage_before,
avail_accelerators_before,total_cpu,total_mem,total_storage,
total_accelerators,avail_network,total_network,util_network,
cpu_idle_power,cpu_max_power,acc_idle_power,acc_max_power,
compute_cap_per_cpu,compute_cap_acc,energy_kwh,chosen_hw_type,accepted
```

### A.3 API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| /rl/health | GET | Service health check |
| /rl/predict | POST | Predict action for state |
| /rl/model | GET | Model info |
| /rl/model/save | POST | Save model |
| /rl/model/load | POST | Load model |
| /rl/experience | POST | Submit experience |
| /rl/training/status | GET | Training status |
| /rl/training/start | POST | Start training |

### A.4 Neural Network Architecture Details

```
Policy Network (Actor-Critic):

Input (state_dim)
    → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(64) → ReLU
    ├── Actor Head: Linear(num_actions) → Softmax → π(a|s)
    └── Critic Head: Linear(1) → V(s)
```

### A.5 Training Commands

```bash
# Single GPU
python scripts/train_rl_distributed.py --timesteps 100000 --num-envs 8

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 scripts/train_rl_distributed.py --timesteps 2000000

# With domain randomization and curriculum
torchrun --nproc_per_node=4 scripts/run_academic_evaluation_v5.py \
    --timesteps 500000 --output-dir results/academic_v8 \
    --use-capacity-features --domain-preset mixed_capacity --curriculum --lr 1e-4
```

---

## **Παράρτημα B - Βιβλιογραφία (Bibliography)**

### B.1 Αναφορές στο Κείμενο (In-text Citations)

### B.2 Βιβλιογραφία (Bibliography)

#### B.2.1 Βιβλία (Books)
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

#### B.2.2 Επιστημονικά Άρθρα (Journal/Conference Papers)
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
- Mao, H., Alizadeh, M., Menache, I., & Kandula, S. (2016). Resource Management with Deep Reinforcement Learning. *HotNets*.
- Beloglazov, A., Abawajy, J., & Buyya, R. (2012). Energy-aware resource allocation heuristics for efficient management of data centers for cloud computing. *Future Generation Computer Systems*, 28(5), 755-768.

#### B.2.3 Τεχνικές Αναφορές (Technical Reports)
- CloudLightning Project Deliverables
- PyTorch Distributed Data Parallel Documentation
- OpenAI Spinning Up Documentation

### B.3 Ηλεκτρονικές Πηγές (Online Sources)
- PyTorch Documentation: https://pytorch.org/docs/
- OpenAI Spinning Up: https://spinningup.openai.com/
- Gymnasium Documentation: https://gymnasium.farama.org/

---

## Ευρετήριο Σχημάτων (List of Figures)

- Figure 1: System Architecture Overview
- Figure 2: MDP Formulation Diagram
- Figure 3: State Encoder Evolution (v1 → v2 → v3)
- Figure 4: PPO Actor-Critic Network Architecture
- Figure 5: Training Reward Curves
- Figure 6: Acceptance Rate Comparison Across Presets
- Figure 7: Resource Utilization Heatmaps
- Figure 8: Rejection Analysis by Category
- Figure 9: Energy Efficiency per Hardware Type
- Figure 10: Generalization Performance Across Environments

---

## Ευρετήριο Πινάκων (List of Tables)

- Table 1: Hardware Type Specifications
- Table 2: State Space Dimensions Summary
- Table 3: Reward Function Components
- Table 4: PPO Hyperparameters
- Table 5: Environment Preset Configurations
- Table 6: Model Version Evolution Results
- Table 7: Baseline Comparison Results
- Table 8: Statistical Significance Tests
- Table 9: Ablation Study Results
