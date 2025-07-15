prompt = """I am conducting a survey on the **Core Capabilities of Deep Research Systems**—systems that iteratively process complex user queries using retrieval, reasoning, and synthesis. These capabilities reflect sequential stages in the research pipeline, from initial interpretation to final answer generation.

For this survey, I define seven core capabilities:

1. Query Understanding and Reasoning
2. Search Strategy Planning & Query Decomposition
3. Tool Invocation
4. Information Retrieval and Processing
5. Iterative Reasoning and Refinement (Feedback Loop)
6. Knowledge Fusion and Integration
7. Answer Synthesis and Generation

The focus is on **post-training techniques** that enhance one or more of these capabilities. These techniques include—but are not limited to—**instruction tuning**, **supervised fine-tuning**, **preference optimization** (e.g., DPO), and **reinforcement learning**.

Please help categorize the following paper:
- Indicate which of the **core capabilities** it aims to improve (one or more).
- Identify which **post-training technique(s)** it employs (one or more).

Paper link: {paper_link}
"""
