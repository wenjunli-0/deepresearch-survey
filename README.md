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







# RL-based Methods
##### Codes Available
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160). [Codes](https://github.com/GAIR-NLP/DeepResearcher)
  - RL (end-to-end GRPO)
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). [Codes](https://github.com/PeterGriffinJin/Search-R1)
  - RL (PPO/GRPO)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592). [Codes](https://github.com/RUCAIBox/R1-Searcher)
  - RL (two-stage outcome-driven RL)
- [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470). [Codes](https://github.com/Agent-RL/ReCall)
  - RL (GRPO without step supervision)
- [Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment](https://arxiv.org/abs/2505.11821). [Codes](https://github.com/SiliangZeng/Multi-Turn-RL-Agent)
  - RL with turn-level credit assignment (MDP + GRPO variant).
- [s3: You Don’t Need That Much Data to Train a Search Agent via RL](https://arxiv.org/abs/2505.14146). [Codes](https://github.com/pat-jj/s3)
  - RL (policy-based RL with Gain Beyond RAG reward)
- [PaSa: An LLM Agent for Comprehensive Academic Paper Search](https://arxiv.org/abs/2501.10120). [Codes](https://github.com/bytedance/pasahttps://github.com/bytedance/pasa)
  - Reinforcement Learning (session-level PPO within AGILE)
- [O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering](https://arxiv.org/abs/2505.16582). [Codes](https://github.com/KnowledgeXLab/O2-Searcher)
  - Reinforcement Learning — RL in a simulated search environment with unified uni-modal reward design for multiple task types.
- [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.17005). [Codes](https://github.com/RUCAIBox/R1-Searcher-plus)
  - Supervised Fine‑Tuning (cold-start SFT); Reinforcement Learning (outcome-driven with internal/external reward and memorization)
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)
  - Supervised Fine-Tuning (to create the retrieval simulator); Reinforcement Learning (policy tuning with curriculum rollout)
- [DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2503.00223). [Codes](https://github.com/pat-jj/DeepRetrieval)
  - Reinforcement Learning (end-to-end RL on retrieval metrics)
- [Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent](https://arxiv.org/abs/2505.07596). [Codes](https://github.com/hzy312/knowledge-r1)
  - Reinforcement Learning (PPO with knowledge-boundary aware rewards)
- [Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieval](https://arxiv.org/abs/2410.23214). [Codes](https://github.com/sher222/LeReThttps://github.com/sher222/LeReT)
  - Reinforcement Learning; Preference Optimization
- [Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366). [Codes](https://github.com/sunnynexus/Search-o1)
  - Reinforcement Learning—building on RL-trained base models to learn agentic retrieval and refined document reasoning
- [Writing-RL: Advancing Long-form Writing via Adaptive Curriculum Reinforcement Learning](https://arxiv.org/abs/2506.05760). [Codes](https://github.com/Tongyi-Zhiwen/Writing-RL)
  - Reinforcement Learning (adaptive curriculum)
- [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536). [Codes](https://retool-rl.github.io/)
  - Supervised Fine-Tuning (cold-start via synthetic examples); Reinforcement Learning (outcome-driven tool-use policy refinement)
- [KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality](https://www.arxiv.org/abs/2506.19807). [Codes](https://github.com/zjunlp/KnowRL)
  - to be written
- [StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization](https://arxiv.org/abs/2505.15107). [Codes](https://github.com/Zillwang/StepSearch)
  - to be written
- [Go-Browse: Training Web Agents with Structured Exploration](https://arxiv.org/abs/2506.03533). [Codes](https://github.com/ApGa/Go-Browse)
  - to be written
- [MMSearch-R1: Incentivizing LMMs to Search](https://arxiv.org/abs/2506.20670). [Codes](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
  - to be written
- [LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning](https://arxiv.org/abs/2506.18841). [Data and Model](https://huggingface.co/THU-KEG/)
  - to be written
- [R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning](https://arxiv.org/abs/2506.04185). [Codes](https://github.com/QingFei1/R-Search)
  - to be written
- [WebDancer: Towards Autonomous Information Seeking Agency](https://arxiv.org/abs/2505.22648). [Codes](https://github.com/Alibaba-NLP/WebAgent)
  - to be written
- [WebSailor: Navigating Super-human Reasoning for Web Agent](https://arxiv.org/abs/2507.02592). [Codes](https://github.com/Alibaba-NLP/WebAgent)
  - to be written
  

##### No codes available
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501).
  - SFT and RL (iterative policy optimization between SFT and RL)
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17391).
  - RL (curriculum-guided, step-level rewards); Preference Optimization (DPO via reward model).
- [Knowledge-Aware Iterative Retrieval for Multi-Agent Systems](https://arxiv.org/abs/2503.13275).
  - Supervised Fine-Tuning (SFT)
- [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.01441).
  - Reinforcement Learning (end‑to‑end, outcome‑based)
- [Iterative Self-Incentivization Empowers Large Language Models as Agentic Searchers](https://arxiv.org/abs/2505.20128).
  - RL via iterative self-incentivization
- [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/pdf/2504.04736).
  - to be written
- [SAGE: Strategy-Adaptive Generation Engine for Query Rewriting](https://www.arxiv.org/abs/2506.19783).
  - to be written
- [Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards](https://arxiv.org/abs/2506.11425).
  - to be written
- [Lessons from Training Grounded LLMs with Verifiable Rewards](https://arxiv.org/abs/2506.15522).
  - to be written


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
- [RQ-RAG: Learning to Refine Queries for Retrieval Augmented Generation](). [Codes]()
- [MaFeRw: Query Rewriting with Multi-Aspect Feedbacks for Retrieval-Augmented Large Language Models](). [Codes]()




# DPO
- [Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieval](https://arxiv.org/abs/2410.23214). [Codes]()
- [PrefRAG: Preference-Driven Multi-Source Retrieval Augmented Generation](https://arxiv.org/abs/2411.00689). [Codes]()
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511). [Codes]()
- [Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning](https://arxiv.org/abs/2505.14069). [Codes]()
- [RAG-Gym: Systematic Optimization of Language Agents for Retrieval-Augmented Generation](https://arxiv.org/abs/2502.13957). [Codes]()


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

### 1. Query Understanding and Reasoning

Techniques and enhancements that improve the system's ability to comprehend complex user queries, interpret intent, and perform initial reasoning to frame the research problem.

**Key Papers:**
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


#### Non-RL-based Papers (e.g., SFT/Instruction Tuning/Preference Optimization/etc)
- [](). [Codes]()
- [](). [Codes]()

#### RL-based Papers
- [](). [Codes]()
- [](). [Codes]()

---

### 2. Search Strategy Planning & Query Decomposition

Methods for breaking down complex queries into sub-questions, planning multi-step search strategies, and orchestrating the research process.

**Key Papers:**
<!-- - [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470). [Codes](https://github.com/Agent-RL/ReCall)
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). [Codes](https://github.com/PeterGriffinJin/Search-R1)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592). [Codes](https://github.com/RUCAIBox/R1-Searcher)
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)
- [StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization](https://arxiv.org/abs/2505.15107). [Codes](https://github.com/Zillwang/StepSearch)
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160). [Codes](https://github.com/GAIR-NLP/DeepResearcher)
- [RAG-RL: Advancing Retrieval-Augmented Generation via RL and Curriculum Learning](https://arxiv.org/abs/2503.12759).
- [SAGE: Strategy-Adaptive Generation Engine for Query Rewriting]()

- [MindSearch: Mimicking Human Minds Elicits Deep AI Searcher](https://arxiv.org/abs/2407.20183). [Codes]()
- [An Agent Framework for Real-Time Financial Information Searching with Large Language Models](https://arxiv.org/abs/2502.15684). [Codes]()
- [Enhancing LLMs' Reasoning-Intensive Multimedia Search Capabilities through Fine-Tuning and Reinforcement Learning](https://arxiv.org/abs/2505.18831). [Codes]()
- [Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents](https://arxiv.org/abs/2505.12065). [Codes](https://github.com/tiannuo-yang/SearchAgent-X)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761). [Codes](https://github.com/conceptofmind/toolformer) -->

#### Non-RL-based Papers (e.g., SFT/Instruction Tuning/Preference Optimization/etc)
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501).
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17391).
- [Knowledge-Aware Iterative Retrieval for Multi-Agent Systems](https://arxiv.org/abs/2503.13275).
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)

#### RL-based Papers
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160). [Codes](https://github.com/GAIR-NLP/DeepResearcher)
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). [Codes](https://github.com/PeterGriffinJin/Search-R1)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592). [Codes](https://github.com/RUCAIBox/R1-Searcher)
- [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470). [Codes](https://github.com/Agent-RL/ReCall)- [Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment](https://arxiv.org/abs/2505.11821). [Codes](https://github.com/SiliangZeng/Multi-Turn-RL-Agent)
- [Iterative Self-Incentivization Empowers Large Language Models as Agentic Searchers](https://arxiv.org/abs/2505.20128).
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501).
- [s3: You Don’t Need That Much Data to Train a Search Agent via RL](https://arxiv.org/abs/2505.14146). [Codes](https://github.com/pat-jj/s3)
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17391).
- [PaSa: An LLM Agent for Comprehensive Academic Paper Search](https://arxiv.org/abs/2501.10120). [Codes](https://github.com/bytedance/pasahttps://github.com/bytedance/pasa)
- [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.01441).
- [O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering](https://arxiv.org/abs/2505.16582). [Codes](https://github.com/KnowledgeXLab/O2-Searcher)
- [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.17005). [Codes](https://github.com/RUCAIBox/R1-Searcher-plus)
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)
- [DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2503.00223). [Codes](https://github.com/pat-jj/DeepRetrieval)


---

### 3. Tool Invocation

Approaches that enable the system to select, invoke, and coordinate external tools, APIs, or computational resources as part of the research workflow.

**Key Papers:**
#### Non-RL-based Papers (e.g., SFT/Instruction Tuning/Preference Optimization/etc)
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501).
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17391).
- [Knowledge-Aware Iterative Retrieval for Multi-Agent Systems](https://arxiv.org/abs/2503.13275).
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)

#### RL-based Papers
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160). [Codes](https://github.com/GAIR-NLP/DeepResearcher)
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). [Codes](https://github.com/PeterGriffinJin/Search-R1)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592). [Codes](https://github.com/RUCAIBox/R1-Searcher)
- [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470). [Codes](https://github.com/Agent-RL/ReCall)
- [Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment](https://arxiv.org/abs/2505.11821). [Codes](https://github.com/SiliangZeng/Multi-Turn-RL-Agent)
- [Iterative Self-Incentivization Empowers Large Language Models as Agentic Searchers](https://arxiv.org/abs/2505.20128).
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501).
- [s3: You Don’t Need That Much Data to Train a Search Agent via RL](https://arxiv.org/abs/2505.14146). [Codes](https://github.com/pat-jj/s3)
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17391).
- [PaSa: An LLM Agent for Comprehensive Academic Paper Search](https://arxiv.org/abs/2501.10120). [Codes](https://github.com/bytedance/pasahttps://github.com/bytedance/pasa)
- [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.01441).
- [O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering](https://arxiv.org/abs/2505.16582). [Codes](https://github.com/KnowledgeXLab/O2-Searcher)
- [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.17005). [Codes](https://github.com/RUCAIBox/R1-Searcher-plus)
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)
- [DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2503.00223). [Codes](https://github.com/pat-jj/DeepRetrieval)


---

### 4. Information Retrieval and Processing

Techniques for retrieving relevant information from various sources, filtering and processing data, and integrating retrieved content into the research pipeline.

**Key Papers:**
#### Non-RL-based Papers (e.g., SFT/Instruction Tuning/Preference Optimization/etc)
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501).
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17391).
- [Knowledge-Aware Iterative Retrieval for Multi-Agent Systems](https://arxiv.org/abs/2503.13275).
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)

#### RL-based Papers
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160). [Codes](https://github.com/GAIR-NLP/DeepResearcher)
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). [Codes](https://github.com/PeterGriffinJin/Search-R1)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592). [Codes](https://github.com/RUCAIBox/R1-Searcher)
- [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470). [Codes](https://github.com/Agent-RL/ReCall)
- [Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment](https://arxiv.org/abs/2505.11821). [Codes](https://github.com/SiliangZeng/Multi-Turn-RL-Agent)
- [Iterative Self-Incentivization Empowers Large Language Models as Agentic Searchers](https://arxiv.org/abs/2505.20128).
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501).
- [s3: You Don’t Need That Much Data to Train a Search Agent via RL](https://arxiv.org/abs/2505.14146). [Codes](https://github.com/pat-jj/s3)
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17391).
- [PaSa: An LLM Agent for Comprehensive Academic Paper Search](https://arxiv.org/abs/2501.10120). [Codes](https://github.com/bytedance/pasahttps://github.com/bytedance/pasa)
- [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.01441).
- [O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering](https://arxiv.org/abs/2505.16582). [Codes](https://github.com/KnowledgeXLab/O2-Searcher)
- [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.17005). [Codes](https://github.com/RUCAIBox/R1-Searcher-plus)
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)
- [DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2503.00223). [Codes](https://github.com/pat-jj/DeepRetrieval)

---

### 5. Iterative Reasoning and Refinement (Feedback Loop)

Enhancements that support multi-step reasoning, self-reflection, verification, and iterative refinement of intermediate results to improve research quality.

**Key Papers:**
#### Non-RL-based Papers (e.g., SFT/Instruction Tuning/Preference Optimization/etc)
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501).
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17391).
- [Knowledge-Aware Iterative Retrieval for Multi-Agent Systems](https://arxiv.org/abs/2503.13275).
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)

#### RL-based Papers
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160). [Codes](https://github.com/GAIR-NLP/DeepResearcher)
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). [Codes](https://github.com/PeterGriffinJin/Search-R1)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592). [Codes](https://github.com/RUCAIBox/R1-Searcher)
- [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470). [Codes](https://github.com/Agent-RL/ReCall)
- [Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment](https://arxiv.org/abs/2505.11821). [Codes](https://github.com/SiliangZeng/Multi-Turn-RL-Agent)
- [Iterative Self-Incentivization Empowers Large Language Models as Agentic Searchers](https://arxiv.org/abs/2505.20128).
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501).
- [s3: You Don’t Need That Much Data to Train a Search Agent via RL](https://arxiv.org/abs/2505.14146). [Codes](https://github.com/pat-jj/s3)
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17391).
- [PaSa: An LLM Agent for Comprehensive Academic Paper Search](https://arxiv.org/abs/2501.10120). [Codes](https://github.com/bytedance/pasahttps://github.com/bytedance/pasa)
- [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.01441).
- [O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering](https://arxiv.org/abs/2505.16582). [Codes](https://github.com/KnowledgeXLab/O2-Searcher)
- [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.17005). [Codes](https://github.com/RUCAIBox/R1-Searcher-plus)
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)

---

### 6. Knowledge Fusion and Integration

Methods for synthesizing, integrating, and reconciling information from multiple sources and modalities to form coherent research insights.

**Key Papers:**
- [Knowledge Fusion of Large Language Models](https://arxiv.org/abs/2401.10491). [Codes]()
- [REPLUG: Retrieval-Augmented Black-Box Language Models](https://arxiv.org/abs/2301.12652). [Codes]()
- [FineTuneBench: How well do commercial fine-tuning APIs infuse knowledge into LLMs?](https://arxiv.org/abs/2411.05059). [Codes]()
- [KBLaM: Knowledge Base augmented Language Model](https://arxiv.org/abs/2410.10450). [Codes]()
- [Injecting Knowledge Graphs into Large Language Models](https://arxiv.org/abs/2505.07554). [Codes]()

#### Non-RL-based Papers (e.g., SFT/Instruction Tuning/Preference Optimization/etc)
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501).
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17391).
- [Knowledge-Aware Iterative Retrieval for Multi-Agent Systems](https://arxiv.org/abs/2503.13275).
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)

#### RL-based Papers
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160). [Codes](https://github.com/GAIR-NLP/DeepResearcher)
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). [Codes](https://github.com/PeterGriffinJin/Search-R1)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592). [Codes](https://github.com/RUCAIBox/R1-Searcher)
- [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470). [Codes](https://github.com/Agent-RL/ReCall)
- [Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment](https://arxiv.org/abs/2505.11821). [Codes](https://github.com/SiliangZeng/Multi-Turn-RL-Agent)
- [Iterative Self-Incentivization Empowers Large Language Models as Agentic Searchers](https://arxiv.org/abs/2505.20128).
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501).
- [s3: You Don’t Need That Much Data to Train a Search Agent via RL](https://arxiv.org/abs/2505.14146). [Codes](https://github.com/pat-jj/s3)
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17391).
- [PaSa: An LLM Agent for Comprehensive Academic Paper Search](https://arxiv.org/abs/2501.10120). [Codes](https://github.com/bytedance/pasahttps://github.com/bytedance/pasa)
- [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.01441).
- [O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering](https://arxiv.org/abs/2505.16582). [Codes](https://github.com/KnowledgeXLab/O2-Searcher)
- [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.17005). [Codes](https://github.com/RUCAIBox/R1-Searcher-plus)
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)

---

### 7. Answer Synthesis and Generation

Techniques that enable the system to generate well-structured, comprehensive, and contextually appropriate research outputs based on integrated findings.

**Key Papers:**
<!-- - [Enhancing LLMs' Reasoning-Intensive Multimedia Search Capabilities through Fine-Tuning and Reinforcement Learning](https://arxiv.org/abs/2505.18831). [Codes]()
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). [Codes](https://github.com/PeterGriffinJin/Search-R1)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592). [Codes](https://github.com/RUCAIBox/R1-Searcher)
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)
- [Reinforcement Fine-Tuning for Reasoning towards Multi-Step Multi-Source Search in Large Language Models](https://arxiv.org/abs/2506.08352). [Codes]()
- [RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism](https://arxiv.org/abs/2507.02962). [Codes](https://github.com/inclusionAI/AgenticLearning/tree/main/RAG-R1)
- [Evidence-Driven Retrieval Augmented Response Generation for Online Misinformation](https://arxiv.org/abs/2403.14952). [Codes]() -->


#### Non-RL-based Papers (e.g., SFT/Instruction Tuning/Preference Optimization/etc)
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501).
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17391).
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)

#### RL-based Papers
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160). [Codes](https://github.com/GAIR-NLP/DeepResearcher)
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). [Codes](https://github.com/PeterGriffinJin/Search-R1)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592). [Codes](https://github.com/RUCAIBox/R1-Searcher)
- [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470). [Codes](https://github.com/Agent-RL/ReCall)
- [Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment](https://arxiv.org/abs/2505.11821). [Codes](https://github.com/SiliangZeng/Multi-Turn-RL-Agent)
- [Iterative Self-Incentivization Empowers Large Language Models as Agentic Searchers](https://arxiv.org/abs/2505.20128).
- [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501).
- [s3: You Don’t Need That Much Data to Train a Search Agent via RL](https://arxiv.org/abs/2505.14146). [Codes](https://github.com/pat-jj/s3)
- [Curriculum Guided Reinforcement Learning for Efficient Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17391).
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588). [Codes](https://github.com/Alibaba-NLP/ZeroSearch)

---

## Data Curation

Data curation is foundational for all post-training enhancements. This section covers improvements in data sourcing, quality control, synthetic data generation, and augmentation, which are critical for effective training and evaluation of research systems.

- **Data Sources:** Methods for identifying and collecting high-quality data relevant to research tasks.
- **Data Quality Control:** Techniques for filtering, cleaning, and validating data to ensure reliability.
- **Synthetic Data Generation:** Approaches for creating artificial data to augment training sets and cover edge cases.
- **Data Augmentation:** Strategies for expanding and diversifying datasets to improve model robustness.

**Key Papers:**
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

---

## Papers Outside the Scope of Enhancing Core Capabilities But Covers Other Aspects of Deep Research Systems
- [Heterogeneous Group-Based Reinforcement Learning for LLM-based Multi-Agent Systems](https://arxiv.org/abs/2506.02718).
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
