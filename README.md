# Reinforcement Learning Enhancements for Deep Research Systems: A Survey

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

Modern deep research systems—advanced AI assistants designed for complex, multi-step information tasks—can be viewed as pipelines composed of distinct yet interdependent capabilities. These capabilities correspond to sequential stages in processing a complex user query, from initial understanding to final answer generation. In this survey, we decompose these key stages into seven core capabilities: (1) Query Understanding and Reasoning, (2) Search Strategy Planning and Query Decomposition, (3) Tool Invocation, (4) Information Retrieval and Processing, (5) Iterative Reasoning and Refinement, (5) Knowledge Fusion and Integration, and (6) Answer Synthesis and Generation.

This survey primarily focuses on `reinforcement learning (RL) approaches` for enhancing deep research systems in an end-to-end manner. We intentionally limit our coverage of other post-training techniques (e.g., supervised fine-tuning, preference optimization, instruction tuning), as RL-based methods represent a rapidly evolving and increasingly prominent paradigm in both academia and industry for training agentic, tool-augmented systems.

Beyond the discussion of RL methods, we also examine evaluation protocols for deep research systems. Due to the complexity and integration of capabilities involved, these systems require comprehensive and multi-faceted benchmarks to accurately assess performance across both objective and open-ended tasks.

## Table of Contents

- [Overview](#overview)
- [Reinforcement Learning Enhancements]()
  - [Training Regime & Data Source]()
  - [Reward & Credit-Assignment Strategy]()
  - [Agent Topology & Coordination]()
  - [Environment & Modality Interface]()
- [Evaluations & Benchmarks](#evaluations--benchmarks)
  - Objective Queries
  - Subjective Queries
  - Complex Tasks
- [Contributing](#contributing)
- [Citation](#citation)

![Overview of Survey](assets/survey%20overview.jpg)

*Figure: High-level overview of the core capabilities in deep research systems.*


[Decoupled Planning and Execution: A Hierarchical Reasoning Framework for Deep Search](https://arxiv.org/pdf/2507.02652). [Codes](https://github.com/RUC-NLPIR/HiRA)


## To be categorized...
- [DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning](https://arxiv.org/pdf/2503.00223). [Codes](https://github.com/pat-jj/DeepRetrieval)
- [WebWalker: Benchmarking LLMs in Web Traversal](https://moonshotai.github.io/Kimi-Researcher/)
- [Enhancing LLMs' Reasoning-Intensive Multimedia Search Capabilities through Fine-Tuning and Reinforcement Learning](https://arxiv.org/pdf/2505.18831).
- [Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents](https://arxiv.org/pdf/2505.12065). [Codes](https://github.com/tiannuo-yang/SearchAgent-X)
- [Reinforcement Fine-Tuning for Reasoning towards Multi-Step Multi-Source Search in Large Language Models](https://arxiv.org/pdf/2506.08352). [Codes]()
- [RAG-R1: Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism](https://arxiv.org/pdf/2507.02962). [Codes](https://github.com/inclusionAI/AgenticLearning/tree/main/RAG-R1)


## technical reports
- [Kimi-Researcher]()


## Categorization

### 1. Training Regime and Algorithm

Determines when learning happens and what data is used. Papers in this category propose new training procedures beyond standard reinforcement learning. This includes step-wise optimization, curriculum learning, multi-stage pipelines, or novel long-horizon credit handling (e.g., return decomposition, trajectory truncation). Use the standard `R1-like-training` as the default reference point. Note that vanilla preference optimization alone (e.g., basic DPO) is not considered sufficiently novel for this category.

**Key Papers:**
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/pdf/2503.09516). [Codes](https://github.com/PeterGriffinJin/Search-R1)
- [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/pdf/2503.19470). [Codes](https://github.com/Agent-RL/ReCall)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2503.05592). [Codes](https://github.com/RUCAIBox/R1-Searcher)
- [s3: You Don’t Need That Much Data to Train a Search Agent via RL](https://arxiv.org/pdf/2505.14146). [Codes](https://github.com/pat-jj/s3)
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/pdf/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)
- [Writing-RL: Advancing Long-form Writing via Adaptive Curriculum Reinforcement Learning](https://arxiv.org/pdf/2506.05760). [Codes](https://github.com/Tongyi-Zhiwen/Writing-RL)
- [LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning](https://arxiv.org/pdf/2506.18841). [Data and Model](https://huggingface.co/THU-KEG/)
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/pdf/2505.22501).
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/pdf/2505.17391).
- [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/pdf/2504.04736).

---

### 2. Data Synthesis

Introduces novel methods for generating or curating synthetic training data. Examples include graph-based question generation, automated answer verification, or multi-step trace synthesis. Merely proposing a new evaluation dataset does not qualify as data synthesis.

**Key Papers:**
- [StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization](https://arxiv.org/pdf/2505.15107). [Codes](https://github.com/Zillwang/StepSearch)
- [Go-Browse: Training Web Agents with Structured Exploration](https://arxiv.org/pdf/2506.03533). [Codes](https://github.com/ApGa/Go-Browse)
- [WebDancer: Towards Autonomous Information Seeking Agency](https://arxiv.org/pdf/2505.22648). [Codes](https://github.com/Alibaba-NLP/WebAgent)
- [WebSailor: Navigating Super-human Reasoning for Web Agent](https://arxiv.org/pdf/2507.02592). [Codes](https://github.com/Alibaba-NLP/WebAgent)
- [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/pdf/2504.04736).

---

### 3. Reward & Credit-Assignment Strategy

Designs or analyzes how reward signals are defined or distributed. This includes final outcome rewards, intermediate/step-level rewards, and shaped or multi-component objective functions. Long-horizon return handling is not categorized here (see Category 1), or modality-specific reward coupling (see Category 5).

**Key Papers:**
- [Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment](https://arxiv.org/pdf/2505.11821). [Codes](https://github.com/SiliangZeng/Multi-Turn-RL-Agent)
- [s3: You Don’t Need That Much Data to Train a Search Agent via RL](https://arxiv.org/pdf/2505.14146). [Codes](https://github.com/pat-jj/s3)
- [PaSa: An LLM Agent for Comprehensive Academic Paper Search](https://arxiv.org/pdf/2501.10120). [Codes](https://github.com/bytedance/pasahttps://github.com/bytedance/pasa)
- [O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering](https://arxiv.org/pdf/2505.16582). [Codes](https://github.com/KnowledgeXLab/O2-Searcher)
- [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/pdf/2505.17005). [Codes](https://github.com/RUCAIBox/R1-Searcher-plus)
- [Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent](https://arxiv.org/pdf/2505.07596). [Codes](https://github.com/hzy312/knowledge-r1)
- [Writing-RL: Advancing Long-form Writing via Adaptive Curriculum Reinforcement Learning](https://arxiv.org/pdf/2506.05760). [Codes](https://github.com/Tongyi-Zhiwen/Writing-RL)
- [KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality](https://www.arxiv.org/pdf/2506.19807). [Codes](https://github.com/zjunlp/KnowRL)
- [StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization](https://arxiv.org/pdf/2505.15107). [Codes](https://github.com/Zillwang/StepSearch)
- [LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning](https://arxiv.org/pdf/2506.18841). [Data and Model](https://huggingface.co/THU-KEG/)
- [R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning](https://arxiv.org/pdf/2506.04185). [Codes](https://github.com/QingFei1/R-Search)
- [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://arxiv.org/pdf/2505.01441).
- [SAGE: Strategy-Adaptive Generation Engine for Query Rewriting](https://www.arxiv.org/pdf/2506.19783).
- [Lessons from Training Grounded LLMs with Verifiable Rewards](https://arxiv.org/pdf/2506.15522).

---

### 4. Agent Topology & Coordination

Explores the structural composition of the agent(s). This includes multi-agent systems, self-reflective loops, hierarchical planners, expert routing, or modular sub-agent coordination.

**Key Papers:**
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/pdf/2504.03160). [Codes](https://github.com/GAIR-NLP/DeepResearcher)
- [PaSa: An LLM Agent for Comprehensive Academic Paper Search](https://arxiv.org/pdf/2501.10120). [Codes](https://github.com/bytedance/pasahttps://github.com/bytedance/pasa)
- [Knowledge-Aware Iterative Retrieval for Multi-Agent Systems](https://arxiv.org/pdf/2503.13275).
- [Decoupled Planning and Execution: A Hierarchical Reasoning Framework for Deep Search](https://arxiv.org/pdf/2507.02652). [Codes](https://github.com/RUC-NLPIR/HiRA)

---

### 5. Multimodal and Multi-tool Interface

Interacts with `multiple modalities` (e.g., vision, text, audio) or makes use of `multiple tool types` beyond simple textual search. Examples include code interpreters, calculators, visual reasoning modules, data extraction tools, or APIs. Work that couples reward across modalities should also be categorized here (not under Category 3).

**Key Papers:**
- [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/pdf/2504.11536). [Codes](https://retool-rl.github.io/)
- [MMSearch-R1: Incentivizing LMMs to Search](https://arxiv.org/pdf/2506.20670). [Codes](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
- [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://arxiv.org/pdf/2505.01441).
- [Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards](https://arxiv.org/pdf/2506.11425).


---

### 6. Others

Interacts with `multiple modalities` (e.g., vision, text, audio) or makes use of `multiple tool types` beyond simple textual search. Examples include code interpreters, calculators, visual reasoning modules, data extraction tools, or APIs. Work that couples reward across modalities should also be categorized here (not under Category 3).

**Key Papers:**

---

## Evaluations & Benchmarks

Comprehensive evaluation and benchmarking are essential for assessing the effectiveness of post-training enhancements. This section discusses:

- **Objective Queries:** Standardized tasks with clear, measurable answers.
- **Subjective Queries:** Open-ended or interpretive tasks requiring nuanced evaluation.
- **Complex Tasks:** Multi-step or integrated research challenges that test the full pipeline.

**Key Papers:**
- [Papers will be added here]
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()

---

## Papers Outside the Scope of Enhancing Core Capabilities But Covers Other Aspects of Deep Research Systems
- [Heterogeneous Group-Based Reinforcement Learning for LLM-based Multi-Agent Systems](https://arxiv.org/pdf/2506.02718).
  - Reinforcement Learning — specifically multi-agent RL via MHGPO.
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()


---

## Contributing

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

## Citation

If you find this survey useful for your research, please consider citing:

```bibtex
@misc{deepresearch-survey-2025,
  title={Post-Training Enhancements to Core Capabilities in Deep Research Systems: A Survey},
  author={[Author names will be added]},
  year={2025},
  url={https://github.com/[username]/deepresearch-survey}
}
```

---

**Last Updated:** [Date will be updated]  
**Maintainers:** [Maintainer information will be added]

## License

This work is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
