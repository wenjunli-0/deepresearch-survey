# Reinforcement Learning Foundations for Deep Research Systems: A Survey

[![arXiv](https://img.shields.io/badge/arXiv-xxx-red.svg)](xxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

## Abstract

Deep research systems, agentic AI that solve complex, multi-step tasks by coordinating reasoning, search across the open web and user files, and tool use, are moving toward hierarchical deployments with a Planner, Coordinator, and Executors. In practice, training entire stacks end-to-end remains impractical, so most work trains a single planner connected to core tools such as search, browsing, and code. While SFT imparts protocol fidelity, it suffers from imitation and exposure biases and underuses environment feedback. Preference alignment methods such as DPO are schema- and proxy-dependent, off-policy, and weak for long-horizon credit assignment and multi-objective trade-offs. A further limitation of SFT and DPO is their reliance on human-defined decision points and subskills through schema design and labeled comparisons. Reinforcement learning aligns with closed-loop, tool-interaction research by optimizing trajectory-level policies, enabling exploration, recovery behaviors, and principled credit assignment, and it reduces dependence on such human priors and rater biases.

This survey is, to our knowledge, the first dedicated to the RL foundations of deep research systems. It systematizes work after DeepSeek-R1 along three axes: (i) data synthesis and curation; (ii) RL methods for agentic research covering stability, sample efficiency, long context handling, reward and credit design, multi-objective optimization, and multimodal integration; and (iii) agentic RL training systems and frameworks. We also cover agent architecture and coordination, as well as evaluation and benchmarks, including recent QA, VQA, long-form synthesis, and domain-grounded, tool-interaction tasks. We distill recurring patterns, surface infrastructure bottlenecks, and offer practical guidance for training robust, transparent deep research agents with RL.

## 📋 Survey Structure

This survey is organized around **one primary focus** and **two secondary areas**:

### **Primary Focus: RL Foundations for Deep Research Systems**
We organize RL-based advancements along three primary axes that capture the key technical challenges and innovations in developing capable and generalizable research agents:

1. **Data Synthesis & Curation** - Methods to generate, select, and validate trajectory data that support multi-hop reasoning and robust tool interaction.
2. **RL Methods for Agentic Research** - Training-time choices that determine decision quality over trajectories: training regimes and optimization structure; reward design and credit assignment for long horizons and multi-objective trade-offs; and integration of multimodal perception with multi-tool action interfaces.
   - **2.1 Training Regimes and Optimization Structure** - New training procedures beyond standard reinforcement learning
   - **2.2 Reward Design and Credit Assignment** - Design and analysis of reward signals and distribution for long horizons and multi-objective trade-offs
   - **2.3 Multimodal Research Agents** - Integration of multimodal perception with multi-tool action interfaces
3. **Agentic RL Training Frameworks** - Scaffolding that makes agentic RL feasible at scale: reproducible environments and tool/API sandboxes; asynchronous actors and rollout collection; caching and rate-limit handling; compute/memory management; distributed training; and logging/evaluation harnesses.

### **Secondary Focus 1: Agent Architecture & Coordination**
Explores hierarchical, modular, and multi-agent designs for structural composition of research agents.

### **Secondary Focus 2: Evaluations & Benchmarks**
Comprehensive evaluation frameworks for holistic, tool-interactive assessment of research agents.

_Note: Some key papers may appear in more than one category if they contribute equally to multiple aspects._

## 📚 Related Surveys

This section provides an overview of existing surveys in related areas to deep research systems.

**Key Related Surveys:**
- [Deep Research Agents: A Systematic Examination And Roadmap](https://arxiv.org/pdf/2506.18096)
- [A Survey on AI Search with Large Language Models](https://www.preprints.org/frontend/manuscript/79453d62cbbfce9ac42239071098a3d9/download_pub)
- [A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges](https://arxiv.org/pdf/2508.05668)
- [A Comprehensive Survey of Deep Research: Systems, Methodologies, and Applications](https://arxiv.org/pdf/2506.12594)
- [Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs](https://arxiv.org/pdf/2507.09477)
- [Deep Research: A Survey of Autonomous Research Agents](https://arxiv.org/pdf/2508.12752)
- [The Landscape of Agentic Reinforcement Learning for LLMs: A Survey](https://arxiv.org/pdf/2509.02547)


---


## 📑 Table of Contents

### Overview
- [Abstract](#abstract)
- [Survey Structure](#survey-structure)
- [Related Surveys](#related-surveys)

### Primary Focus: RL Foundations
- [Data Synthesis & Curation](#1-data-synthesis--curation)
- [RL Methods for Agentic Research](#2-rl-methods-for-agentic-research)
  - [Training Regimes and Optimization Structure](#21-training-regimes-and-optimization-structure)
  - [Reward Design and Credit Assignment](#22-reward-design-and-credit-assignment)
  - [Multimodal Research Agents](#23-multimodal-research-agents)
- [Agentic RL Training Frameworks](#3-agentic-rl-training-frameworks)

### Secondary Focus Areas
- [Agent Architecture & Coordination](#secondary-focus-1-agent-architecture--coordination)
- [Evaluations & Benchmarks](#secondary-focus-2-evaluations--benchmarks)

### Additional Resources
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

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
- [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://arxiv.org/pdf/2505.01441).
- [Pangu DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](https://arxiv.org/pdf/2505.24332).
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/pdf/2505.22501).
- [FrugalRAG: Learning to retrieve and reason for multi-hop QA](https://arxiv.org/pdf/2507.07634). 
- [Writing-RL: Advancing Long-form Writing via Adaptive Curriculum Reinforcement Learning](https://arxiv.org/pdf/2506.05760).

---

#### 2.2 Reward Design and Credit Assignment

Designs or analyzes how reward signals are defined or distributed for long horizons and multi-objective trade-offs. This includes final outcome rewards, intermediate/step-level rewards, and shaped or multi-component objective functions. Long-horizon return handling is not categorized here (see Section 2.1), nor modality-specific reward coupling (see Section 2.3).

**Key Papers:**
- [s3: You Don't Need That Much Data to Train a Search Agent via RL](https://arxiv.org/pdf/2505.14146). <a href="https://github.com/pat-jj/s3" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs](https://arxiv.org/pdf/2505.11277). <a href="https://github.com/syr-cn/AutoRefine" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment](https://arxiv.org/pdf/2505.11821). <a href="https://github.com/SiliangZeng/Multi-Turn-RL-Agent" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent](https://arxiv.org/pdf/2505.07596). <a href="https://github.com/hzy312/knowledge-r1" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/pdf/2505.17005). <a href="https://github.com/RUCAIBox/R1-Searcher-plus" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization](https://arxiv.org/pdf/2505.15107). <a href="https://github.com/Zillwang/StepSearch" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering](https://arxiv.org/pdf/2505.16582). <a href="https://github.com/KnowledgeXLab/O2-Searcher" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning](https://arxiv.org/pdf/2506.04185). <a href="https://github.com/QingFei1/R-Search" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>


---

#### 2.3 Multimodal Research Agents

End-to-end multimodal models that natively perceive and reason over multiple modalities (e.g., vision and language) without delegating core competence to external tool executors. This section focuses on vision–language models (VLMs) that perform iterative perception–reasoning cycles, directly ingesting visual evidence (images, charts, documents, UI screenshots, etc.) and producing grounded reasoning within a unified multimodal token space. We exclude systems whose multimodality is realized through auxiliary tools (e.g., separate OCR/vision pipelines, code interpreters, or retrieval modules).

**Key Papers:**
**(1) End-to-End Multimodal Models**
- [Visual Agentic Reinforcement Fine-Tuning](https://arxiv.org/pdf/2505.14246). <a href="https://github.com/Liuziyu77/Visual-RFT/tree/main/Visual-ARFT" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning](https://arxiv.org/pdf/2505.22019). <a href="https://github.com/Alibaba-NLP/VRAG" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [MMSearch-R1: Incentivizing LMMs to Search](https://arxiv.org/pdf/2506.20670). <a href="https://github.com/EvolvingLMMs-Lab/multimodal-search-r1" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](https://arxiv.org/pdf/2505.19255). <a href="https://github.com/VTool-R1/VTool-R1" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning](https://arxiv.org/pdf/2505.08617). <a href="https://github.com/zhaochen0110/OpenThinkIMG" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [WebWatcher: Breaking New Frontier of Vision-Language Deep Research Agent](https://arxiv.org/pdf/2508.05748). <a href="https://github.com/Alibaba-NLP/WebAgent" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>

**(2) Tool-Augmented Language Models**
- [Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning](https://arxiv.org/pdf/2506.13654). <a href="https://github.com/egolife-ai/Ego-R1" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation](https://arxiv.org/pdf/2505.23885). <a href="https://github.com/camel-ai/owl" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory](https://arxiv.org/pdf/2508.09736). <a href="https://github.com/bytedance-seed/m3-agent" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Enhancing LLMs' Reasoning-Intensive Multimedia Search Capabilities through Fine-Tuning and Reinforcement Learning](https://arxiv.org/pdf/2505.18831)

---

### 3. Agentic RL Training Frameworks

Scaffolding that makes agentic RL feasible at scale: reproducible environments and tool/API sandboxes; asynchronous actors and rollout collection; caching and rate-limit handling; compute/memory management; distributed training; and logging/evaluation harnesses.

**Key Papers and Repos:**
- [Agent Lightning: Train ANY AI Agents with Reinforcement Learning](https://arxiv.org/pdf/2508.03680). <a href="https://github.com/microsoft/agent-lightning" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [AWorld: Orchestrating the Training Recipe for Agentic AI](https://arxiv.org/pdf/2508.20404). <a href="https://github.com/inclusionAI/AWorld/" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/pdf/2505.24298v2). <a href="https://github.com/inclusionAI/AReaL/" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models](https://arxiv.org/pdf/2410.09671). <a href="https://github.com/openreasoner/openr" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [rLLM – RL for Post-Training Language Agents](https://agentica-project.com/). <a href="https://github.com/agentica-project/rllm" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [ROLL – Reinforcement Learning Optimization for LLMs](https://arxiv.org/pdf/2506.06122). <a href="https://github.com/alibaba/ROLL" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [slime: An SGLang-Native Post-Training Framework for RL Scaling](https://lmsys.org/blog/2025-07-09-slime/). <a href="https://github.com/THUDM/slime" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Verifiers](https://verifiers.readthedocs.io/en/latest/overview.html#). <a href="https://github.com/willccbb/verifiers" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Verifiers](https://verifiers.readthedocs.io/en/latest/overview.html). <a href="https://github.com/willccbb/verifiers" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [verl – Volcano Engine RL Training Library.](https://arxiv.org/pdf/2409.19256v2) <a href="https://github.com/volcengine/verl" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>

---

## Secondary Focus 1: Agent Architecture & Coordination

Explores hierarchical, modular, and multi-agent designs for structural composition of research agents. This includes multi-agent systems, self-reflective loops, hierarchical planners, expert routing, or modular sub-agent coordination.

**Key Papers:**
- [OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation](https://arxiv.org/pdf/2505.23885). <a href="https://github.com/camel-ai/owl" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL](https://arxiv.org/pdf/2508.13167). <a href="https://github.com/OPPO-PersonalAI/Agent_Foundation_Models" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
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

- **QA/VQA Benchmarks:** Stress final answer accuracy under retrieval and browsing.
- **Long-Form Text Benchmarks:** Long form synthesis, which assesses the quality of extended reports.
- **Domain-Grounded Benchmarks:** End-to-end task execution with tools.

**Key Papers:**
- [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/pdf/1809.09600). <a href="https://hotpotqa.github.io/" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps](https://arxiv.org/pdf/2011.01060). <a href="https://rajpurkar.github.io/SQuAD-explorer/" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Natural Questions: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026.pdf). <a href="https://github.com/openai/simple-evals" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [MuSiQue: Multihop Questions via Single-hop Question Composition](https://arxiv.org/pdf/2108.00573). <a href="https://github.com/stonybrooknlp/musique" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [FEVER: a Large-scale Dataset for Fact Extraction and VERification](https://aclanthology.org/N18-1074.pdf). <a href="https://github.com/awslabs/fever" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/pdf/2210.03350). <a href="https://github.com/ofirpress/self-ask" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems](https://arxiv.org/pdf/1704.00057). <a href="http://datasets.maluuba.com/Frames" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents](https://arxiv.org/pdf/2504.12516). <a href="https://github.com/openai/simple-evals" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese](https://arxiv.org/pdf/2504.19314). <a href="https://github.com/PALIN2018/BrowseComp-ZH" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation](https://arxiv.org/pdf/2505.15872). <a href="https://infodeepseek.github.io/" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [WebWalker: Benchmarking LLMs in Web Traversal](https://arxiv.org/pdf/2501.07572). <a href="https://github.com/Alibaba-NLP/WebAgent" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [WideSearch: Benchmarking Agentic Broad Info-Seeking](https://arxiv.org/pdf/2508.07999). <a href="https://github.com/ByteDance-Seed/WideSearch" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [MMSearch: Benchmarking the Potential of Large Models as Multi-modal Search Engines](https://arxiv.org/pdf/2409.12959). <a href="https://mmsearch.github.io/" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [MRAMG-Bench: A Comprehensive Benchmark for Advancing Multimodal Retrieval-Augmented Multimodal Generation](https://arxiv.org/pdf/2502.04176). <a href="https://github.com/MRAMG-Bench/MRAMG" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Benchmarking Retrieval-Augmented Generation in Multi-Modal Contexts](https://arxiv.org/pdf/2502.17297). <a href="https://github.com/NEUIR/M2RAG" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [MM-BrowseComp: A Comprehensive Benchmark for Multimodal Browsing Agents](https://arxiv.org/pdf/2508.13186). <a href="https://github.com/MMBrowseComp/MM-BrowseComp" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [OmniBench: Towards The Future of Universal Omni-Language Models](https://arxiv.org/pdf/2409.15272). <a href="https://github.com/multimodal-art-projection/OmniBench" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [PROXYQA: An Alternative Framework for Evaluating Long-Form Text Generation with Large Language Models](https://arxiv.org/pdf/2401.15042). <a href="https://proxy-qa.com/" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [WritingBench: A Comprehensive Benchmark for Generative Writing](https://arxiv.org/pdf/2503.05244). <a href="https://github.com/X-PLUG/WritingBench" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [LongEval: A Comprehensive Analysis of Long-Text Generation Through a Plan-based Paradigm](https://arxiv.org/pdf/2502.19103). <a href="https://github.com/Wusiwei0410/LongEval" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents](https://arxiv.org/pdf/2409.15272). <a href="https://github.com/Ayanami0730/deep_research_bench" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [xbench: Tracking Agents Productivity Scaling with Profession-Aligned Real-World Evaluations](https://www.arxiv.org/pdf/2506.13651). <a href="https://xbench.org/" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [τ2-Bench: Evaluating Conversational Agents in a Dual-Control Environment](https://arxiv.org/pdf/2506.07982). <a href="https://github.com/sierra-research/tau2-bench" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [Finance Agent Benchmark: Benchmarking LLMs on Real-world Financial Research Tasks](https://arxiv.org/pdf/2508.00828). <a href="https://github.com/vals-ai/finance-agent" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [FinGAIA: A Chinese Benchmark for AI Agents in Real-World Financial Domain](https://arxiv.org/pdf/2507.17186). <a href="https://github.com/SUFEAIFLM-Lab/FinGAIA" target="_blank"><img src="assets/github-mark.svg" alt="GitHub" width="16" height="16"></a>
- [OdysseyBench: Evaluating LLM Agents on Long-Horizon Complex Office Application Workflows](https://arxiv.org/pdf/2508.09124).

---

## 📚 Citation

If you find this survey useful for your research, please consider citing:

```bibtex
@misc{deepresearch-survey-2025,
  title={Reinforcement Learning Foundations for Deep Research Systems: A Survey},
  author={Wenjun Li, Zhi Chen, Hannan Cao, Jingru Lin, Wei Han, Sheng Liang, Zhi Zhang, Kuicai Dong, Yong Liu},
  year={2025},
  url={https://github.com/wenjunli-0/deepresearch-survey}
}
```
 

## 📄 License

This work is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
