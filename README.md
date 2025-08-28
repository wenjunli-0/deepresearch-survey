# Reinforcement Learning Foundations for Deep Research Systems: A Survey

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

Deep research systems are agentic AI models that solve complex, multi-step information tasks by coordinating reasoning, search, and tool use—often under partial observability and non-stationary conditions. This survey centers on **reinforcement learning (RL) as the primary driver** of recent progress, arguing that trajectory-level optimization, exploration, and credit assignment are essential for end-to-end decision quality in tool-rich environments.

## 📋 Survey Structure

This survey is organized around **one primary focus** and **two secondary areas**:

### **Primary Focus: RL Foundations for Deep Research Systems**
We organize RL-based advancements along three primary axes that capture the key technical challenges and innovations in developing capable and generalizable research agents:

1. **Data Synthesis & Curation** - Methods to generate, select, and validate trajectory data that support multi-hop reasoning and robust tool interaction.
2. **RL Methods for Agentic Research** - Training-time choices that determine decision quality over trajectories: training regimes and optimization structure; reward design and credit assignment for long horizons and multi-objective trade-offs; and integration of multimodal perception with multi-tool action interfaces.
   - **2.1 Training Regimes and Optimization Structure** - New training procedures beyond standard reinforcement learning
   - **2.2 Reward Design and Credit Assignment** - Design and analysis of reward signals and distribution for long horizons and multi-objective trade-offs
   - **2.3 Multimodal and Multi-Tool Integration** - Integration of multimodal perception with multi-tool action interfaces
3. **Systems & Infrastructure** - Scaffolding that makes agentic RL feasible at scale: reproducible environments and tool/API sandboxes; asynchronous actors and rollout collection; caching and rate-limit handling; compute/memory management; distributed training; and logging/evaluation harnesses.

### **Secondary Focus 1: Agent Architecture & Coordination**
Explores hierarchical, modular, and multi-agent designs for structural composition of research agents.

### **Secondary Focus 2: Evaluations & Benchmarks**
Comprehensive evaluation frameworks for holistic, tool-interactive assessment of research agents.

_Note: Some key papers may appear in more than one category if they contribute equally to multiple aspects._

_Ordering: Within each list, key papers are ordered by descending GitHub repository stars as of Aug 25, 2025._

## 📚 Related Surveys

This section provides an overview of existing surveys in related areas to deep research systems.

**Key Related Surveys:**
- [Deep Research Agents: A Systematic Examination And Roadmap](https://arxiv.org/pdf/2506.18096)
- [A Survey on AI Search with Large Language Models](https://www.preprints.org/frontend/manuscript/79453d62cbbfce9ac42239071098a3d9/download_pub)
- [A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges](https://arxiv.org/pdf/2508.05668)
- [A Comprehensive Survey of Deep Research: Systems, Methodologies, and Applications](https://arxiv.org/pdf/2506.12594)
- [Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs](https://arxiv.org/pdf/2507.09477)

---


## 📑 Table of Contents

### Introduction
- [Related Surveys](#-related-surveys)

### Primary Focus: RL Foundations
- [Data Synthesis & Curation](#1-data-synthesis--curation)
- [RL Methods for Agentsuric Research](#2-rl-methods-for-agentic-research)
  - [Training Regimes and Optimization Structure](#21-training-regimes-and-optimization-structure)
  - [Reward Design and Credit Assignment](#22-reward-design-and-credit-assignment)
  - [Multimodal and Multi-Tool Integration](#23-multimodal-and-multi-tool-integration)
- [Systems & Infrastructure](#3-systems--infrastructure)

### Secondary Focuses
- [Agent Architecture & Coordination](#secondary-focus-1-agent-architecture--coordination)
- [Evaluations & Benchmarks](#secondary-focus-2-evaluations--benchmarks)

### Additional Resources
- [Contributing](#contributing)
- [Citation](#citation)

---

## Primary Focus: RL Foundations for Deep Research Systems

### 1. Data Synthesis & Curation

Methods to generate, select, and validate trajectory data that support multi-hop reasoning and robust tool interaction. Introduces novel methods for generating or curating synthetic training data. Examples include graph-based question generation, automated answer verification, or multi-step trace synthesis. Merely proposing a new evaluation dataset does not qualify as data synthesis.

**Key Papers:**
- [WebDancer: Towards Autonomous Information Seeking Agency](https://arxiv.org/pdf/2505.22648). <a href="https://github.com/Alibaba-NLP/WebAgent" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [WebSailor: Navigating Super-human Reasoning for Web Agent](https://arxiv.org/pdf/2507.02592). <a href="https://github.com/Alibaba-NLP/WebAgent" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization](https://arxiv.org/pdf/2507.15061). <a href="https://github.com/Alibaba-NLP/WebAgent" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [WebWatcher: Breaking New Frontier of Vision-Language Deep Research Agent](https://arxiv.org/pdf/2508.05748). <a href="https://github.com/Alibaba-NLP/WebAgent" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/pdf/2503.09516). <a href="https://github.com/PeterGriffinJin/Search-R1" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/pdf/2504.03160). <a href="https://github.com/GAIR-NLP/DeepResearcher" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](https://arxiv.org/pdf/2506.15841). <a href="https://github.com/MIT-MI/MEM1" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization](https://arxiv.org/pdf/2505.15107). <a href="https://github.com/Zillwang/StepSearch" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Go-Browse: Training Web Agents with Structured Exploration](https://arxiv.org/pdf/2506.03533). <a href="https://github.com/ApGa/Go-Browse" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Reinforcement Fine-Tuning for Reasoning towards Multi-Step Multi-Source Search in Large Language Models](https://arxiv.org/pdf/2506.08352). <a href="https://github.com/wentao0429/ReasoningSearch" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Pangu DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](https://arxiv.org/pdf/2505.24332).
- [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/pdf/2504.04736).
- [Enhancing LLMs' Reasoning-Intensive Multimedia Search Capabilities through Fine-Tuning and Reinforcement Learning](https://arxiv.org/pdf/2505.18831). 

---

### 2. RL Methods for Agentic Research

Training-time choices that determine decision quality over trajectories: training regimes and optimization structure; reward design and credit assignment for long horizons and multi-objective trade-offs; and integration of multimodal perception with multi-tool action interfaces.

#### 2.1 Training Regimes and Optimization Structure

Determines when learning happens and what data is used. Papers in this category propose new training procedures beyond standard reinforcement learning. This includes step-wise optimization, curriculum learning, multi-stage pipelines, or novel long-horizon credit handling (e.g., return decomposition, trajectory truncation). Use the standard `R1-like-training` as the default reference point. Note that vanilla preference optimization alone (e.g., basic DPO) is not considered sufficiently novel for this category.

**Key Papers:**
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/pdf/2503.09516). <a href="https://github.com/PeterGriffinJin/Search-R1" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/pdf/2503.19470). <a href="https://github.com/Agent-RL/ReCall" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2503.05592). <a href="https://github.com/RUCAIBox/R1-Searcher" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/pdf/2505.04588). <a href="https://github.com/Alibaba-NLP/ZeroSearch" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [s3: You Don't Need That Much Data to Train a Search Agent via RL](https://arxiv.org/pdf/2505.14146). <a href="https://github.com/pat-jj/s3" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Reasoning-Table: Exploring Reinforcement Learning for Table Reasoning](https://arxiv.org/pdf/2506.01710). <a href="https://github.com/MJinXiang/Reasoning-Table" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs](https://arxiv.org/pdf/2505.11277). <a href="https://github.com/syr-cn/AutoRefine" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Agentic Reinforced Policy Optimization](https://arxiv.org/pdf/2507.19849). <a href="https://github.com/SiliangZeng/Multi-Turn-RL-Agent" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](https://arxiv.org/pdf/2506.15841). <a href="https://github.com/MIT-MI/MEM1" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism](https://arxiv.org/pdf/2507.02962). <a href="https://github.com/inclusionAI/AgenticLearning/tree/main/RAG-R1" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [FrugalRAG: Learning to retrieve and reason for multi-hop QA](https://arxiv.org/pdf/2507.07634). 
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/pdf/2505.22501).
- [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://arxiv.org/pdf/2505.01441).
- [Reinforcement Fine-Tuning for Reasoning towards Multi-Step Multi-Source Search in Large Language Models](https://arxiv.org/pdf/2506.08352). <a href="https://github.com/wentao0429/ReasoningSearch" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Pangu DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](https://arxiv.org/pdf/2505.24332).


---

#### 2.2 Reward Design and Credit Assignment

Designs or analyzes how reward signals are defined or distributed for long horizons and multi-objective trade-offs. This includes final outcome rewards, intermediate/step-level rewards, and shaped or multi-component objective functions. Long-horizon return handling is not categorized here (see Category 1), or modality-specific reward coupling (see Category 2.3).

**Key Papers:**
- [s3: You Don't Need That Much Data to Train a Search Agent via RL](https://arxiv.org/pdf/2505.14146). <a href="https://github.com/pat-jj/s3" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs](https://arxiv.org/pdf/2505.11277). <a href="https://github.com/syr-cn/AutoRefine" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment](https://arxiv.org/pdf/2505.11821). <a href="https://github.com/SiliangZeng/Multi-Turn-RL-Agent" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent](https://arxiv.org/pdf/2505.07596). <a href="https://github.com/hzy312/knowledge-r1" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/pdf/2505.17005). <a href="https://github.com/RUCAIBox/R1-Searcher-plus" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization](https://arxiv.org/pdf/2505.15107). <a href="https://github.com/Zillwang/StepSearch" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering](https://arxiv.org/pdf/2505.16582). <a href="https://github.com/KnowledgeXLab/O2-Searcher" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning](https://arxiv.org/pdf/2506.04185). <a href="https://github.com/QingFei1/R-Search" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>


<!-- **Irrelevant Papers** -->
<!-- - [KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality](https://www.arxiv.org/pdf/2506.19807). <a href="https://github.com/zjunlp/KnowRL" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a> -->
<!-- - [Lessons from Training Grounded LLMs with Verifiable Rewards](https://arxiv.org/pdf/2506.15522). -->
<!-- - [SAGE: Strategy-Adaptive Generation Engine for Query Rewriting](https://www.arxiv.org/pdf/2506.19783). -->


---

#### 2.3 Multimodal and Multi-Tool Integration

Integration of multimodal perception with multi-tool action interfaces. Interacts with `multiple modalities` (e.g., vision, text, audio) or makes use of `multiple tool types` beyond simple textual search. Examples include code interpreters, calculators, visual reasoning modules, data extraction tools, or APIs. Work that couples reward across modalities should also be categorized here (not under Category 2.2).

**Key Papers:**

**(1) End-to-End Multimodal Models**
- [Visual Agentic Reinforcement Fine-Tuning](https://arxiv.org/pdf/2505.14246). <a href="https://github.com/Liuziyu77/Visual-RFT/tree/main/Visual-ARFT" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning](https://arxiv.org/pdf/2505.22019). <a href="https://github.com/Alibaba-NLP/VRAG" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [MMSearch-R1: Incentivizing LMMs to Search](https://arxiv.org/pdf/2506.20670). <a href="https://github.com/EvolvingLMMs-Lab/multimodal-search-r1" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](https://arxiv.org/pdf/2505.19255). <a href="https://github.com/VTool-R1/VTool-R1" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [WebWatcher: Breaking New Frontier of Vision-Language Deep Research Agent](https://arxiv.org/pdf/2508.05748). <a href="https://github.com/Alibaba-NLP/WebAgent" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>

**(2) Tool-Augmented Language Models**
- [Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning](https://arxiv.org/pdf/2506.13654). <a href="https://github.com/egolife-ai/Ego-R1" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Enhancing LLMs' Reasoning-Intensive Multimedia Search Capabilities through Fine-Tuning and Reinforcement Learning](https://arxiv.org/pdf/2505.18831)

---

### 3. Systems & Infrastructure

Scaffolding that makes agentic RL feasible at scale: reproducible environments and tool/API sandboxes; asynchronous actors and rollout collection; caching and rate-limit handling; compute/memory management; distributed training; and logging/evaluation harnesses.

**Key Papers and Repos:**
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/pdf/2505.24298v2). <a href="https://github.com/inclusionAI/AReaL/" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [rLLM – RL for Post-Training Language Agents](https://agentica-project.com/). <a href="https://github.com/agentica-project/rllm" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [ROLL – Reinforcement Learning Optimization for LLMs](https://arxiv.org/pdf/2506.06122). <a href="https://github.com/alibaba/ROLL" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- Qwen-Agent – Agent Framework for Qwen LLMs. <a href="https://github.com/QwenLM/Qwen-Agent" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [OpenR – Open-Source Reasoning & RL Platform](https://arxiv.org/pdf/2410.09671) <a href="https://openreasoner.github.io/" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Agent Lightning – General RL Training for AI Agents](https://arxiv.org/pdf/2508.03680v1) <a href="https://github.com/microsoft/agent-lightning" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [verl – Volcano Engine RL Training Library](https://arxiv.org/pdf/2409.19256v2) <a href="https://github.com/volcengine/verl" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>

---

## Secondary Focus 1: Agent Architecture & Coordination

Explores hierarchical, modular, and multi-agent designs for structural composition of research agents. This includes multi-agent systems, self-reflective loops, hierarchical planners, expert routing, or modular sub-agent coordination.

**Key Papers:**
- [OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation](https://arxiv.org/pdf/2505.23885). <a href="https://github.com/camel-ai/owl" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [PaSa: An LLM Agent for Comprehensive Academic Paper Search](https://arxiv.org/pdf/2501.10120). <a href="https://github.com/bytedance/pasa" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [WebThinker: Empowering Large Reasoning Models with Deep Research Capability](https://arxiv.org/pdf/2504.21776). <a href="https://github.com/RUC-NLPIR/WebThinker" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/pdf/2504.03160). <a href="https://github.com/GAIR-NLP/DeepResearcher" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2501.15228). <a href="https://github.com/chenyiqun/MMOA-RAG" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Decoupled Planning and Execution: A Hierarchical Reasoning Framework for Deep Search](https://arxiv.org/pdf/2507.02652). <a href="https://github.com/RUC-NLPIR/HiRA" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards](https://arxiv.org/pdf/2507.03041). <a href="https://github.com/snap-stanford/optimas" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Heterogeneous Group-Based Reinforcement Learning for LLM-based Multi-Agent Systems](https://arxiv.org/pdf/2506.02718).

---

## Secondary Focus 2: Evaluations & Benchmarks

Comprehensive evaluation frameworks for holistic, tool-interactive assessment of research agents. This section discusses:

- **Objective Queries:** Standardized tasks with clear, measurable answers.
- **Subjective Queries:** Open-ended or interpretive tasks requiring nuanced evaluation.
- **General Agent Tasks:** Multi-step or integrated research challenges that test the full pipeline.

**Key Papers:**
- [Papers will be added here]

---

## 🤝 Contributing

We welcome contributions to this survey! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:

- Add new papers or repositories
- Suggest new categories or reorganization
- Report errors or outdated information
- Propose improvements to the survey structure

### How to Contribute

1. Fork this repository
2. Create a new branch for your contribution
3. Add your content following the established format
4. Submit a pull request with a clear description of your changes

## 📚 Citation

If you find this survey useful for your research, please consider citing:

```bibtex
@misc{deepresearch-survey-2025,
  title={Reinforcement Learning Foundations for Deep Research Systems: A Survey},
  author={[Author names will be added]},
  year={2025},
  url={https://github.com/[username]/deepresearch-survey}
}
```
```

---

**Last Updated:** [Date will be updated]  
**Maintainers:** [Maintainer information will be added]

## 📄 License

This work is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
