# Reinforcement Learning Foundations for Deep Research Systems: A Survey

[![arXiv](https://img.shields.io/badge/arXiv-xxx-red.svg)](xxx)

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

_Note: Some papers may appear in more than one category if they contribute equally to multiple aspects._

## 📚 Related Surveys

This section list concurrent surveys in related areas to deep research systems.

| Paper Title | Paper Link |
|-------------|------------|
| Deep Research Agents: A Systematic Examination And Roadmap | [arXiv](https://arxiv.org/pdf/2506.18096) |
| A Survey on AI Search with Large Language Models | [Preprints](https://www.preprints.org/frontend/manuscript/79453d62cbbfce9ac42239071098a3d9/download_pub) |
| A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges | [arXiv](https://arxiv.org/pdf/2508.05668) |
| A Comprehensive Survey of Deep Research: Systems, Methodologies, and Applications | [arXiv](https://arxiv.org/pdf/2506.12594) |
| Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs | [arXiv](https://arxiv.org/pdf/2507.09477) |
| Deep Research: A Survey of Autonomous Research Agents | [arXiv](https://arxiv.org/pdf/2508.12752) |
| The Landscape of Agentic Reinforcement Learning for LLMs: A Survey | [arXiv](https://arxiv.org/pdf/2509.02547) |


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

| Paper Title | Paper Link | Code Link |
|-------------|------------|------------------|
| WebDancer: Towards Autonomous Information Seeking Agency | [arXiv](https://arxiv.org/pdf/2505.22648) | [GitHub](https://github.com/Alibaba-NLP/WebAgent) |
| WebSailor: Navigating Super-human Reasoning for Web Agent | [arXiv](https://arxiv.org/pdf/2507.02592) | [GitHub](https://github.com/Alibaba-NLP/WebAgent) |
| WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization | [arXiv](https://arxiv.org/pdf/2507.15061) | [GitHub](https://github.com/Alibaba-NLP/WebAgent) |
| WebWatcher: Breaking New Frontier of Vision-Language Deep Research Agent | [arXiv](https://arxiv.org/pdf/2508.05748) | [GitHub](https://github.com/Alibaba-NLP/WebAgent) |
| Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2503.09516) | [GitHub](https://github.com/PeterGriffinJin/Search-R1) |
| DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments | [arXiv](https://arxiv.org/pdf/2504.03160) | [GitHub](https://github.com/GAIR-NLP/DeepResearcher) |
| MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents | [arXiv](https://arxiv.org/pdf/2506.15841) | [GitHub](https://github.com/MIT-MI/MEM1) |
| StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization | [arXiv](https://arxiv.org/pdf/2505.15107) | [GitHub](https://github.com/Zillwang/StepSearch) |
| Go-Browse: Training Web Agents with Structured Exploration | [arXiv](https://arxiv.org/pdf/2506.03533) | [GitHub](https://github.com/ApGa/Go-Browse) |
| Reinforcement Fine-Tuning for Reasoning towards Multi-Step Multi-Source Search in Large Language Models | [arXiv](https://arxiv.org/pdf/2506.08352) | [GitHub](https://github.com/wentao0429/ReasoningSearch) |
| Pangu DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2505.24332) |  |
| Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use | [arXiv](https://arxiv.org/pdf/2504.04736) |  |
| Enhancing LLMs' Reasoning-Intensive Multimedia Search Capabilities through Fine-Tuning and Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2505.18831) |  |

---

### 2. RL Methods for Agentic Research

Training-time choices that determine decision quality over trajectories: training regimes and optimization structure; reward design and credit assignment for long horizons and multi-objective trade-offs; and integration of multimodal perception.

#### 2.1 Training Regimes and Optimization Structure

Determines when learning happens and what data is used. Papers in this category propose new training procedures beyond standard reinforcement learning. This includes step-wise optimization, curriculum learning, multi-stage pipelines, or novel long-horizon credit handling (e.g., return decomposition, trajectory truncation). Use the standard `DeepSeek-R1-style-training` as the default reference point. Note that we focus on end-to-end RL training rather than SFT/DPO approaches.

| Paper Title | Paper Link | Code Link |
|-------------|------------|------------------|
| Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2503.09516) | [GitHub](https://github.com/PeterGriffinJin/Search-R1) |
| ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2503.19470) | [GitHub](https://github.com/Agent-RL/ReCall) |
| R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2503.05592) | [GitHub](https://github.com/RUCAIBox/R1-Searcher) |
| ZeroSearch: Incentivize the Search Capability of LLMs without Searching | [arXiv](https://arxiv.org/pdf/2505.04588) | [GitHub](https://github.com/Alibaba-NLP/ZeroSearch) |
| s3: You Don't Need That Much Data to Train a Search Agent via RL | [arXiv](https://arxiv.org/pdf/2505.14146) | [GitHub](https://github.com/pat-jj/s3) |
| Reasoning-Table: Exploring Reinforcement Learning for Table Reasoning | [arXiv](https://arxiv.org/pdf/2506.01710) | [GitHub](https://github.com/MJinXiang/Reasoning-Table) |
| Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs | [arXiv](https://arxiv.org/pdf/2505.11277) | [GitHub](https://github.com/syr-cn/AutoRefine) |
| Agentic Reinforced Policy Optimization | [arXiv](https://arxiv.org/pdf/2507.19849) | [GitHub](https://github.com/SiliangZeng/Multi-Turn-RL-Agent) |
| MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents | [arXiv](https://arxiv.org/pdf/2506.15841) | [GitHub](https://github.com/MIT-MI/MEM1) |
| RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism | [arXiv](https://arxiv.org/pdf/2507.02962) | [GitHub](https://github.com/inclusionAI/AgenticLearning/tree/main/RAG-R1) |
| Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2505.01441) |  |
| Pangu DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2505.24332) |  |
| EvolveSearch: An Iterative Self-Evolving Search Agent | [arXiv](https://arxiv.org/pdf/2505.22501) |  |
| FrugalRAG: Learning to retrieve and reason for multi-hop QA | [arXiv](https://arxiv.org/pdf/2507.07634) |  |
| Writing-RL: Advancing Long-form Writing via Adaptive Curriculum Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2506.05760) |  |

---

#### 2.2 Reward Design and Credit Assignment

Designs or analyzes how reward signals are defined or distributed for long horizons and multi-objective trade-offs. This includes final outcome rewards, intermediate/step-level rewards, and shaped or multi-component objective functions. Long-horizon return handling is not categorized here (see Section 2.1), nor modality-specific reward coupling (see Section 2.3).

| Paper Title | Paper Link | Code Link |
|-------------|------------|------------------|
| s3: You Don't Need That Much Data to Train a Search Agent via RL | [arXiv](https://arxiv.org/pdf/2505.14146) | [GitHub](https://github.com/pat-jj/s3) |
| Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs | [arXiv](https://arxiv.org/pdf/2505.11277) | [GitHub](https://github.com/syr-cn/AutoRefine) |
| Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment | [arXiv](https://arxiv.org/pdf/2505.11821) | [GitHub](https://github.com/SiliangZeng/Multi-Turn-RL-Agent) |
| Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent | [arXiv](https://arxiv.org/pdf/2505.07596) | [GitHub](https://github.com/hzy312/knowledge-r1) |
| R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2505.17005) | [GitHub](https://github.com/RUCAIBox/R1-Searcher-plus) |
| StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization | [arXiv](https://arxiv.org/pdf/2505.15107) | [GitHub](https://github.com/Zillwang/StepSearch) |
| O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering | [arXiv](https://arxiv.org/pdf/2505.16582) | [GitHub](https://github.com/KnowledgeXLab/O2-Searcher) |
| R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2506.04185) | [GitHub](https://github.com/QingFei1/R-Search) |

---

#### 2.3 Multimodal Research Agents

End-to-end multimodal models that natively perceive and reason over multiple modalities (e.g., vision and language) without delegating core competence to external tool executors. This section focuses on vision–language models (VLMs) that perform iterative perception–reasoning cycles, directly ingesting visual evidence (images, charts, documents, UI screenshots, etc.) and producing grounded reasoning within a unified multimodal token space. We exclude systems whose multimodality is realized through auxiliary tools (e.g., separate OCR/vision pipelines, code interpreters, or retrieval modules).

#### End-to-End Multimodal Models
| Paper Title | Paper Link | Code Link |
|-------------|------------|------------------|
| Visual Agentic Reinforcement Fine-Tuning | [arXiv](https://arxiv.org/pdf/2505.14246) | [GitHub](https://github.com/Liuziyu77/Visual-RFT/tree/main/Visual-ARFT) |
| VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2505.22019) | [GitHub](https://github.com/Alibaba-NLP/VRAG) |
| MMSearch-R1: Incentivizing LMMs to Search | [arXiv](https://arxiv.org/pdf/2506.20670) | [GitHub](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1) |
| VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use | [arXiv](https://arxiv.org/pdf/2505.19255) | [GitHub](https://github.com/VTool-R1/VTool-R1) |
| OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2505.08617) | [GitHub](https://github.com/zhaochen0110/OpenThinkIMG) |
| WebWatcher: Breaking New Frontier of Vision-Language Deep Research Agent | [arXiv](https://arxiv.org/pdf/2508.05748) | [GitHub](https://github.com/Alibaba-NLP/WebAgent) |

#### Tool-Augmented Language Models

| Paper Title | Paper Link | Code Link |
|-------------|------------|------------------|
| Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning | [arXiv](https://arxiv.org/pdf/2506.13654) | [GitHub](https://github.com/egolife-ai/Ego-R1) |
| OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation | [arXiv](https://arxiv.org/pdf/2505.23885) | [GitHub](https://github.com/camel-ai/owl) |
| Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory | [arXiv](https://arxiv.org/pdf/2508.09736) | [GitHub](https://github.com/bytedance-seed/m3-agent) |
| Enhancing LLMs' Reasoning-Intensive Multimedia Search Capabilities through Fine-Tuning and Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2505.18831) |  |

---

### 3. Agentic RL Training Frameworks

Scaffolding that makes agentic RL feasible at scale: reproducible environments and tool/API sandboxes; asynchronous actors and rollout collection; caching and rate-limit handling; compute/memory management; distributed training; and logging/evaluation harnesses.

| Paper Title | Paper Link | Code Link |
|-------------|------------|------------------|
| Agent Lightning: Train ANY AI Agents with Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2508.03680) | [GitHub](https://github.com/microsoft/agent-lightning) |
| AWorld: Orchestrating the Training Recipe for Agentic AI | [arXiv](https://arxiv.org/pdf/2508.20404) | [GitHub](https://github.com/inclusionAI/AWorld/) |
| AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning | [arXiv](https://arxiv.org/pdf/2505.24298v2) | [GitHub](https://github.com/inclusionAI/AReaL/) |
| OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models | [arXiv](https://arxiv.org/pdf/2410.09671) | [GitHub](https://github.com/openreasoner/openr) |
| rLLM – RL for Post-Training Language Agents | [Website](https://agentica-project.com/) | [GitHub](https://github.com/agentica-project/rllm) |
| ROLL – Reinforcement Learning Optimization for LLMs | [arXiv](https://arxiv.org/pdf/2506.06122) | [GitHub](https://github.com/alibaba/ROLL) |
| slime: An SGLang-Native Post-Training Framework for RL Scaling | [Blog](https://lmsys.org/blog/2025-07-09-slime/) | [GitHub](https://github.com/THUDM/slime) |
| Verifiers | [Docs](https://verifiers.readthedocs.io/en/latest/overview.html#) | [GitHub](https://github.com/willccbb/verifiers) |
| verl – Volcano Engine RL Training Library | [arXiv](https://arxiv.org/pdf/2409.19256v2) | [GitHub](https://github.com/volcengine/verl) |

---

## Secondary Focus 1: Agent Architecture & Coordination

Explores hierarchical, modular, and multi-agent designs for structural composition of research agents. This includes multi-agent systems, self-reflective loops, hierarchical planners, expert routing, or modular sub-agent coordination.

| Paper Title | Paper Link | Code Link |
|-------------|------------|------------------|
| OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation | [arXiv](https://arxiv.org/pdf/2505.23885) | [GitHub](https://github.com/camel-ai/owl) |
| Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL | [arXiv](https://arxiv.org/pdf/2508.13167) | [GitHub](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models) |
| PaSa: An LLM Agent for Comprehensive Academic Paper Search | [arXiv](https://arxiv.org/pdf/2501.10120) | [GitHub](https://github.com/bytedance/pasa) |
| WebThinker: Empowering Large Reasoning Models with Deep Research Capability | [arXiv](https://arxiv.org/pdf/2504.21776) | [GitHub](https://github.com/RUC-NLPIR/WebThinker) |
| DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments | [arXiv](https://arxiv.org/pdf/2504.03160) | [GitHub](https://github.com/GAIR-NLP/DeepResearcher) |
| Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning | [arXiv](https://arxiv.org/pdf/2501.15228) | [GitHub](https://github.com/chenyiqun/MMOA-RAG) |
| Decoupled Planning and Execution: A Hierarchical Reasoning Framework for Deep Search | [arXiv](https://arxiv.org/pdf/2507.02652) | [GitHub](https://github.com/RUC-NLPIR/HiRA) |
| Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards | [arXiv](https://arxiv.org/pdf/2507.03041) | [GitHub](https://github.com/snap-stanford/optimas) |
| Heterogeneous Group-Based Reinforcement Learning for LLM-based Multi-Agent Systems | [arXiv](https://arxiv.org/pdf/2506.02718) |  |

---

## Secondary Focus 2: Evaluations & Benchmarks

Comprehensive evaluation frameworks for holistic, tool-interactive assessment of research agents. This section discusses:

- **QA/VQA Benchmarks:** Stress final answer accuracy under retrieval and browsing.
- **Long-Form Text Benchmarks:** Long form synthesis, which assesses the quality of extended reports.
- **Domain-Grounded Benchmarks:** End-to-end task execution with tools.

### QA/VQA Benchmarks

| Paper Title | Paper Link | Code Link |
|-------------|------------|------------------|
| HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering | [arXiv](https://arxiv.org/pdf/1809.09600) | [Website](https://hotpotqa.github.io/) |
| Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps | [arXiv](https://arxiv.org/pdf/2011.01060) | [Website](https://rajpurkar.github.io/SQuAD-explorer/) |
| Natural Questions: A Benchmark for Question Answering Research | [ACL](https://aclanthology.org/Q19-1026.pdf) | [GitHub](https://github.com/openai/simple-evals) |
| MuSiQue: Multihop Questions via Single-hop Question Composition | [arXiv](https://arxiv.org/pdf/2108.00573) | [GitHub](https://github.com/stonybrooknlp/musique) |
| FEVER: a Large-scale Dataset for Fact Extraction and VERification | [ACL](https://aclanthology.org/N18-1074.pdf) | [GitHub](https://github.com/awslabs/fever) |
| Measuring and Narrowing the Compositionality Gap in Language Models | [arXiv](https://arxiv.org/pdf/2210.03350) | [GitHub](https://github.com/ofirpress/self-ask) |
| Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems | [arXiv](https://arxiv.org/pdf/1704.00057) | [Website](http://datasets.maluuba.com/Frames) |
| BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents | [arXiv](https://arxiv.org/pdf/2504.12516) | [GitHub](https://github.com/openai/simple-evals) |
| BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese | [arXiv](https://arxiv.org/pdf/2504.19314) | [GitHub](https://github.com/PALIN2018/BrowseComp-ZH) |
| InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation | [arXiv](https://arxiv.org/pdf/2505.15872) | [Website](https://infodeepseek.github.io/) |
| WebWalker: Benchmarking LLMs in Web Traversal | [arXiv](https://arxiv.org/pdf/2501.07572) | [GitHub](https://github.com/Alibaba-NLP/WebAgent) |
| WideSearch: Benchmarking Agentic Broad Info-Seeking | [arXiv](https://arxiv.org/pdf/2508.07999) | [GitHub](https://github.com/ByteDance-Seed/WideSearch) |
| MMSearch: Benchmarking the Potential of Large Models as Multi-modal Search Engines | [arXiv](https://arxiv.org/pdf/2409.12959) | [Website](https://mmsearch.github.io/) |
| MRAMG-Bench: A Comprehensive Benchmark for Advancing Multimodal Retrieval-Augmented Multimodal Generation | [arXiv](https://arxiv.org/pdf/2502.04176) | [GitHub](https://github.com/MRAMG-Bench/MRAMG) |
| Benchmarking Retrieval-Augmented Generation in Multi-Modal Contexts | [arXiv](https://arxiv.org/pdf/2502.17297) | [GitHub](https://github.com/NEUIR/M2RAG) |
| MM-BrowseComp: A Comprehensive Benchmark for Multimodal Browsing Agents | [arXiv](https://arxiv.org/pdf/2508.13186) | [GitHub](https://github.com/MMBrowseComp/MM-BrowseComp) |
| OmniBench: Towards The Future of Universal Omni-Language Models | [arXiv](https://arxiv.org/pdf/2409.15272) | [GitHub](https://github.com/multimodal-art-projection/OmniBench) |

### Long-Form Text Benchmarks

| Paper Title | Paper Link | Code Link |
|-------------|------------|------------------|
| PROXYQA: An Alternative Framework for Evaluating Long-Form Text Generation with Large Language Models | [arXiv](https://arxiv.org/pdf/2401.15042) | [Website](https://proxy-qa.com/) |
| WritingBench: A Comprehensive Benchmark for Generative Writing | [arXiv](https://arxiv.org/pdf/2503.05244) | [GitHub](https://github.com/X-PLUG/WritingBench) |
| LongEval: A Comprehensive Analysis of Long-Text Generation Through a Plan-based Paradigm | [arXiv](https://arxiv.org/pdf/2502.19103) | [GitHub](https://github.com/Wusiwei0410/LongEval) |

### Domain-Grounded Benchmarks

| Paper Title | Paper Link | Code Link |
|-------------|------------|------------------|
| DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents | [arXiv](https://arxiv.org/pdf/2409.15272) | [GitHub](https://github.com/Ayanami0730/deep_research_bench) |
| xbench: Tracking Agents Productivity Scaling with Profession-Aligned Real-World Evaluations | [arXiv](https://www.arxiv.org/pdf/2506.13651) | [Website](https://xbench.org/) |
| τ2-Bench: Evaluating Conversational Agents in a Dual-Control Environment | [arXiv](https://arxiv.org/pdf/2506.07982) | [GitHub](https://github.com/sierra-research/tau2-bench) |
| Finance Agent Benchmark: Benchmarking LLMs on Real-world Financial Research Tasks | [arXiv](https://arxiv.org/pdf/2508.00828) | [GitHub](https://github.com/vals-ai/finance-agent) |
| FinGAIA: A Chinese Benchmark for AI Agents in Real-World Financial Domain | [arXiv](https://arxiv.org/pdf/2507.17186) | [GitHub](https://github.com/SUFEAIFLM-Lab/FinGAIA) |
| OdysseyBench: Evaluating LLM Agents on Long-Horizon Complex Office Application Workflows | [arXiv](https://arxiv.org/pdf/2508.09124) |  |

---

## 📚 Citation

If you find this survey useful for your research, please consider citing:

```bibtex
@misc{deepresearch-survey-2025,
  title={Reinforcement Learning Foundations for Deep Research Systems: A Survey},
  author={Wenjun Li, Zhi Chen, Hannan Cao, Jingru Lin, Wei Han, Sheng Liang, Zhi Zhang, Kuicai Dong, Chen Zhang, Dexun Li, Yong Liu},
  year={2025},
  url={https://github.com/wenjunli-0/deepresearch-survey}
}
```
