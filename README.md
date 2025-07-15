# Post-Training Enhancements to Core Capabilities in Deep Research Systems: A Survey

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

Modern deep research systems (advanced AI research assistants) can be understood as pipelines of distinct yet interrelated capabilities. These capabilities correspond to sequential steps in processing a complex user query, from initial understanding to final answer generation. In this survey, we decompose these key steps into seven core capabilities, which are presented in order and categorized into the main sections outlined in the table of contents below.

This survey focuses on `post-training enhancements` that improve these core capabilities in deep research systems. In addition to the seven primary capabilities, we also include discussions on data curation and evaluation. Effective post-training methods critically rely on well-curated data, and the complexity of deep research systems demands comprehensive evaluation protocols and benchmarks to assess system performance across diverse tasks.

## Table of Contents

- [Overview](#overview)
- [Core Capability Areas](#core-capability-areas)
  - [Query Understanding and Reasoning](#query-understanding)
  - [Search Strategy Planning & Query Decomposition](#search-strategy-and-query-decomposition)
  - [Tool Invocation](#tool-invocation)
  - [Information Retrieval and Processing](#information-retrieval-and-processing)
  - [Iterative Reasoning and Refinement (Feedback Loop)](#iterative-reasoning-and-refinement)
  - [Knowledge Fusion and Integration](#knowledge-fusion-and-integration)
  - [Answer Synthesis and Generation](#answer-synthesis-and-generation)
- [Data Curation]()
  - Data Sources
  - Data Quality Control
  - Synthetic Data Generation
  - Data Augmentation
- [Evaluations & Benchmarks](#evaluations--benchmarks)
  - Objective Queries
  - Subjective Queries
  - Complex tasks
- [Contributing](#contributing)
- [Citation](#citation)

![Overview of Survey](assets/survey%20overview.jpg)

*Figure: High-level overview of the core capabilities in deep research systems.*


- [RQ-RAG: Learning to Refine Queries for Retrieval Augmented Generation](). [Codes]()
- [MaFeRw: Query Rewriting with Multi-Aspect Feedbacks for Retrieval-Augmented Large Language Models](). [Codes]()


# RL-based Methods
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160). [Codes](https://github.com/GAIR-NLP/DeepResearcher)
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). [Codes](https://github.com/PeterGriffinJin/Search-R1)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592). [Codes](https://github.com/RUCAIBox/R1-Searcher)
- [Heterogeneous Group-Based Reinforcement Learning for LLM-based Multi-Agent Systems](https://arxiv.org/abs/2506.02718). [Codes]()
- [Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment](https://arxiv.org/abs/2505.11821). [Codes]()
- [Iterative Self-Incentivization Empowers Large Language Models as Agentic Searchers](https://arxiv.org/abs/2505.20128). [Codes]()
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501). [Codes]()
- [s3: You Don’t Need That Much Data to Train a Search Agent via RL](https://arxiv.org/abs/2505.14146). [Codes]()
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17391). [Codes]()
- [Knowledge-Aware Iterative Retrieval for Multi-Agent Systems](https://arxiv.org/abs/2503.13275). [Codes]()
- [PaSa: An LLM Agent for Comprehensive Academic Paper Search](https://arxiv.org/abs/2501.10120). [Codes]()
- [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.01441). [Codes]()
- [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470). [Codes](https://github.com/Agent-RL/ReCall)
- [O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering](https://arxiv.org/abs/2505.16582). [Codes]()
- [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.17005). [Codes]()
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)
- [WebDancer: Towards Autonomous Information Seeking Agency](https://arxiv.org/abs/2505.22648). [Codes]()
- [DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2503.00223). [Codes]()
- [Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent](https://arxiv.org/abs/2505.07596). [Codes]()
- [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/abs/2410.23214). [Codes]()
- [Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieval](https://arxiv.org/abs/2410.23214). [Codes]()
- [Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366). [Codes]()
- [Writing-RL: Advancing Long-form Writing via Adaptive Curriculum Reinforcement Learning](https://arxiv.org/abs/2506.05760). [Codes]()
- [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536). [Codes]()
- [StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization](https://arxiv.org/abs/2505.15107). [Codes](https://github.com/Zillwang/StepSearch)
- [LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning](https://arxiv.org/abs/2506.18841). [Codes]()
- [R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning](https://arxiv.org/abs/2506.04185). [Codes]()
- [MMSearch-R1: Incentivizing LMMs to Search](https://arxiv.org/abs/2506.20670). [Codes]()
- [SAGE: Strategy-Adaptive Generation Engine for Query Rewriting](https://www.arxiv.org/abs/2506.19783). [Codes]()
- [Go-Browse: Training Web Agents with Structured Exploration](https://arxiv.org/abs/2506.03533). [Codes]()
- [Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards](https://arxiv.org/abs/2506.11425). [Codes]()
- [Lessons from Training Grounded LLMs with Verifiable Rewards](https://arxiv.org/abs/2506.15522). [Codes]()
- [KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality](https://www.arxiv.org/abs/2506.19807). [Codes]()
- [WebSailor: Navigating Super-human Reasoning for Web Agent](https://arxiv.org/abs/2507.02592). [Codes]()
- []()
- []()


# SFT-based Methods
- [ComposeRAG: A Modular and Composable RAG for Corpus-Grounded Multi-Hop Question Answering](https://arxiv.org/abs/2506.00232). [Codes]()
- [Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2505.22571). [Codes]()
- [Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents](https://arxiv.org/abs/2505.09970). [Codes]()
- [ChatCite: LLM Agent with Human Workflow Guidance for Comparative Literature Summary](https://arxiv.org/abs/2403.02574). [Codes]()
- [HopRAG: Multi-Hop Reasoning for Logic-Aware Retrieval-Augmented Generation](https://arxiv.org/abs/2502.12442). [Codes]()
- [Self-DC: When to Retrieve and When to Generate? Self Divide-and-Conquer for Compositional Unknown Questions](https://arxiv.org/abs/2402.13514). [Codes]()
- [Retrieval Augmented Iterative Self-Feedback](https://arxiv.org/abs/2403.06840). [Codes]()
- [Corrective Retrieval-Augmented Generation](https://arxiv.org/abs/2401.15884). [Codes]()
- [MaskSearch: A Universal Pre-Training Framework to Enhance Agentic Search Capability](https://arxiv.org/abs/2505.20285). [Codes]()
- [AI for Climate Finance: Agentic Retrieval and Multi-Step Reasoning for Early Warning System Investments](https://arxiv.org/abs/2504.05104). [Codes]()
- [Patience is all you need! An Agentic System for Performing Scientific Literature Review](https://arxiv.org/abs/2504.08752). [Codes]()
- [Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research](https://arxiv.org/abs/2502.04644). [Codes]()
- [ManuSearch: Democratizing Deep Search in LLMs with a Transparent and Open Multi-Agent Framework](https://arxiv.org/abs/2505.18105). [Codes]()
- [Search-o1: Agentic Search-Enhanced Large Reasoning Model](https://arxiv.org/abs/2501.05366). [Codes]()
- [WebThinker: Empowering Large Reasoning Models with Deep Research Capability](https://arxiv.org/abs/2504.21776). [Codes]()
- [SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis](https://arxiv.org/abs/2505.16834). [Codes]()
- [NNetNav: Unsupervised Learning of Browser Agents Through Environment Interaction in the Wild](https://arxiv.org/abs/2410.02907). [Codes]()
- []()
- []()



# DPO
- [Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieval](https://arxiv.org/abs/2410.23214). [Codes]()
- [PrefRAG: Preference-Driven Multi-Source Retrieval Augmented Generation](https://arxiv.org/abs/2411.00689). [Codes]()
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511). [Codes]()
- [Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning](https://arxiv.org/abs/2505.14069). [Codes]()
- [RAG-Gym: Systematic Optimization of Language Agents for Retrieval-Augmented Generation](https://arxiv.org/abs/2502.13957). [Codes]()
- []()
- []()




# To be categorized
- [WebWalker: Benchmarking LLMs in Web Traversal](https://moonshotai.github.io/Kimi-Researcher/)



# technical reports
- [Kimi-Researcher]()



## Overview

Deep research systems require sophisticated capabilities beyond basic language understanding and generation. This survey focuses on post-training techniques that enhance these systems' ability to conduct research, reason about complex problems, and interact with various tools and data sources effectively.

Post-training enhancements include but are not limited to:
- Prompt engineering techniques
- Fine-tuning approaches
- Reinforcement learning approaches

## Core Capability Areas Covered in this Survey

This survey organizes the core capabilities of deep research systems into the following seven main areas, reflecting the sequential steps in advanced research workflows:

### Query Understanding and Reasoning

Techniques and enhancements that improve the system's ability to comprehend complex user queries, interpret intent, and perform initial reasoning to frame the research problem.

**Key Papers:**
- [Papers will be added here]
- [Reinforcement Fine-Tuning for Reasoning towards Multi-Step Multi-Source Search in Large Language Models](https://arxiv.org/abs/2506.08352). [Codes]()
- [RL-of-Thoughts: Inference-Time Reinforcement Learning for Reasoning](https://arxiv.org/abs/2505.14140v1). [Codes]()
- [Offline Reinforcement Learning for LLM Multi-Step Reasoning](https://arxiv.org/abs/2412.16145). [Codes]()
- [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992). [Codes]()
- [DeepSeek-R1: Large-Scale RL for Reasoning Alignment](https://arxiv.org/abs/2501.12948). [Codes]()
- [Kimi k1.5: Scaling RLHF for Planning and Reasoning](https://arxiv.org/abs/2501.12599). [Codes]()
- [Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for Large Language Models](https://arxiv.org/abs/2403.12881). [Codes]()
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761). [Codes](https://github.com/conceptofmind/toolformer)
- [Tool Learning in the Wild: Empowering Language Models as Automatic Tool Agents](https://arxiv.org/abs/2405.16533). [Codes]()
- [Reflect-RL: Two-Player Online RL Fine-Tuning for LMs](https://arxiv.org/abs/2402.12621). [Codes]()
- [MARFT: Multi-Agent Reinforcement Fine-Tuning](https://arxiv.org/abs/2504.16129). [Codes]()
- [SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning
](https://arxiv.org/abs/2506.01713). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()

---

### Search Strategy Planning & Query Decomposition

Methods for breaking down complex queries into sub-questions, planning multi-step search strategies, and orchestrating the research process.

**Key Papers:**
- [Papers will be added here]
#### RL-based Methods
- [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470). [Codes](https://github.com/Agent-RL/ReCall)
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). [Codes](https://github.com/PeterGriffinJin/Search-R1)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592). [Codes](https://github.com/RUCAIBox/R1-Searcher)
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)
- [StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization](https://arxiv.org/abs/2505.15107). [Codes](https://github.com/Zillwang/StepSearch)
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160). [Codes](https://github.com/GAIR-NLP/DeepResearcher)
- [RAG-RL: Advancing Retrieval-Augmented Generation via RL and Curriculum Learning](https://arxiv.org/abs/2503.12759).
- [SAGE: Strategy-Adaptive Generation Engine for Query Rewriting]()
#### Other Methods
- [MindSearch: Mimicking Human Minds Elicits Deep AI Searcher](https://arxiv.org/abs/2407.20183). [Codes]()
- [An Agent Framework for Real-Time Financial Information Searching with Large Language Models](https://arxiv.org/abs/2502.15684). [Codes]()
- [Enhancing LLMs' Reasoning-Intensive Multimedia Search Capabilities through Fine-Tuning and Reinforcement Learning](https://arxiv.org/abs/2505.18831). [Codes]()
- [Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents](https://arxiv.org/abs/2505.12065). [Codes](https://github.com/tiannuo-yang/SearchAgent-X)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761). [Codes](https://github.com/conceptofmind/toolformer)
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()


---

### Tool Invocation

Approaches that enable the system to select, invoke, and coordinate external tools, APIs, or computational resources as part of the research workflow.

**Key Papers:**
- [Papers will be added here]
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()

---

### Information Retrieval and Processing

Techniques for retrieving relevant information from various sources, filtering and processing data, and integrating retrieved content into the research pipeline.

**Key Papers:**
- [Papers will be added here]
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()

---

### Iterative Reasoning and Refinement (Feedback Loop)

Enhancements that support multi-step reasoning, self-reflection, verification, and iterative refinement of intermediate results to improve research quality.

**Key Papers:**
- [Papers will be added here]
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()

---

### Knowledge Fusion and Integration

Methods for synthesizing, integrating, and reconciling information from multiple sources and modalities to form coherent research insights.

**Key Papers:**
- papers below are too old. Should find more recent papers.
- [Knowledge Fusion of Large Language Models](https://arxiv.org/abs/2401.10491). [Codes]()
- [REPLUG: Retrieval-Augmented Black-Box Language Models](https://arxiv.org/abs/2301.12652). [Codes]()
- [FineTuneBench: How well do commercial fine-tuning APIs infuse knowledge into LLMs?](https://arxiv.org/abs/2411.05059). [Codes]()
- [KBLaM: Knowledge Base augmented Language Model](https://arxiv.org/abs/2410.10450). [Codes]()
- [Injecting Knowledge Graphs into Large Language Models](https://arxiv.org/abs/2505.07554). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()

---

### Answer Synthesis and Generation

Techniques that enable the system to generate well-structured, comprehensive, and contextually appropriate research outputs based on integrated findings.

**Key Papers:**
- [Papers will be added here]
- [Enhancing LLMs' Reasoning-Intensive Multimedia Search Capabilities through Fine-Tuning and Reinforcement Learning](https://arxiv.org/abs/2505.18831). [Codes]()
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). [Codes](https://github.com/PeterGriffinJin/Search-R1)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592). [Codes](https://github.com/RUCAIBox/R1-Searcher)
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)
- [Reinforcement Fine-Tuning for Reasoning towards Multi-Step Multi-Source Search in Large Language Models](https://arxiv.org/abs/2506.08352). [Codes]()
- [RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism](https://arxiv.org/abs/2507.02962). [Codes](https://github.com/inclusionAI/AgenticLearning/tree/main/RAG-R1)
- [Evidence-Driven Retrieval Augmented Response Generation for Online Misinformation](https://arxiv.org/abs/2403.14952). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()

---

## Data Curation

Data curation is foundational for all post-training enhancements. This section covers improvements in data sourcing, quality control, synthetic data generation, and augmentation, which are critical for effective training and evaluation of research systems.

- **Data Sources:** Methods for identifying and collecting high-quality data relevant to research tasks.
- **Data Quality Control:** Techniques for filtering, cleaning, and validating data to ensure reliability.
- **Synthetic Data Generation:** Approaches for creating artificial data to augment training sets and cover edge cases.
- **Data Augmentation:** Strategies for expanding and diversifying datasets to improve model robustness.

**Key Papers:**
- [Papers will be added here]
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()

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
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
- [](). [Codes]()
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
