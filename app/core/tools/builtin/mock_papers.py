RAG_ARXIV_PAPERS = [
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni", "Vladimir Karpukhin"],
        "year": 2020,
        "abstract": "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) models that combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever.",
        "arxiv_id": "2005.11401",
        "url": "http://arxiv.org/abs/2005.11401",
        "categories": ["cs.CL", "cs.AI"],
    },
    {
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "authors": ["Akari Asai", "Zeqiu Wu", "Yizhong Wang", "Avirup Sil", "Hannaneh Hajishirzi"],
        "year": 2023,
        "abstract": "We introduce Self-RAG, a new framework that trains an LM to adaptively retrieve passages on-demand, and generate and reflect on retrieved passages and its own generations using special reflection tokens. Self-RAG trains a single arbitrary LM to generate text with reflection tokens that indicate retrieval necessity and generation quality.",
        "arxiv_id": "2310.11511",
        "url": "http://arxiv.org/abs/2310.11511",
        "categories": ["cs.CL", "cs.AI"],
    },
    {
        "title": "REALM: Retrieval-Augmented Language Model Pre-Training",
        "authors": ["Kelvin Guu", "Kenton Lee", "Zora Tung", "Panupong Pasupat", "Ming-Wei Chang"],
        "year": 2020,
        "abstract": "We explore a novel pre-training approach called Retrieval-Augmented Language Model (REALM) which augments language model pre-training with a learned textual knowledge retriever. The key intuition is that we can improve the model by retrieving and attending over documents from a large corpus during pre-training.",
        "arxiv_id": "2002.08909",
        "url": "http://arxiv.org/abs/2002.08909",
        "categories": ["cs.CL", "cs.AI"],
    },
    {
        "title": "Active Retrieval Augmented Generation",
        "authors": ["Zhengbao Jiang", "Frank F. Xu", "Luyu Gao", "Zhiqing Sun", "Qian Liu"],
        "year": 2023,
        "abstract": "Despite the remarkable ability of large language models, they still struggle with knowledge-intensive tasks. Retrieval-augmented generation aims to address this by retrieving relevant documents. We propose FLARE, an active retrieval augmented generation method that iteratively uses a prediction of the upcoming sentence to anticipate future content.",
        "arxiv_id": "2305.06983",
        "url": "http://arxiv.org/abs/2305.06983",
        "categories": ["cs.CL"],
    },
    {
        "title": "Benchmarking Large Language Models in Retrieval-Augmented Generation",
        "authors": ["Jiawei Chen", "Hongyu Lin", "Xianpei Han", "Le Sun"],
        "year": 2024,
        "abstract": "Retrieval-Augmented Generation (RAG) is a promising approach for mitigating the hallucination of large language models. This paper systematically investigates the impact of RAG on LLMs by establishing a comprehensive benchmark across different RAG scenarios including noise robustness, negative rejection, information integration, and counterfactual robustness.",
        "arxiv_id": "2309.01431",
        "url": "http://arxiv.org/abs/2309.01431",
        "categories": ["cs.CL", "cs.IR"],
    },
]
RAG_SCHOLAR_PAPERS = [
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni", "Vladimir Karpukhin"],
        "year": 2020,
        "abstract": "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) models that combine pre-trained parametric and non-parametric memory for language generation.",
        "citation_count": 3842,
        "url": "https://www.semanticscholar.org/paper/RAG-Lewis/58ed1fbaabe027345e5c21b6b752aeaff6e42e18",
        "venue": "NeurIPS",
        "paper_id": "58ed1fba",
    },
    {
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "authors": ["Vladimir Karpukhin", "Barlas Oguz", "Sewon Min", "Patrick Lewis", "Ledell Wu"],
        "year": 2020,
        "abstract": "Open-domain question answering relies on efficient passage retrieval to select candidate contexts. We introduce Dense Passage Retriever (DPR) using dense representations with a simple dual-encoder framework trained with question and passage pairs.",
        "citation_count": 4215,
        "url": "https://www.semanticscholar.org/paper/DPR-Karpukhin/58a1a483c6e41fba7b24e75cb09a0e2f39e3049f",
        "venue": "EMNLP",
        "paper_id": "58a1a483",
    },
    {
        "title": "Improving Language Models by Retrieving from Trillions of Tokens",
        "authors": ["Sebastian Borgeaud", "Arthur Mensch", "Jordan Hoffmann", "Trevor Cai", "Eliza Rutherford"],
        "year": 2022,
        "abstract": "We enhance auto-regressive language models by conditioning on document chunks retrieved from a large corpus, based on local similarity with preceding tokens. Our Retrieval-Enhanced Transformer (RETRO) obtains comparable performance to GPT-3 with 25x fewer parameters.",
        "citation_count": 1523,
        "url": "https://www.semanticscholar.org/paper/RETRO-Borgeaud/2a89c9cc3db7d46b5e28e6e50aa3b9e53b6f5b0e",
        "venue": "ICML",
        "paper_id": "2a89c9cc",
    },
    {
        "title": "Atlas: Few-shot Learning with Retrieval Augmented Language Models",
        "authors": ["Gautier Izacard", "Patrick Lewis", "Maria Lomeli", "Lucas Hosseini", "Fabio Petroni"],
        "year": 2023,
        "abstract": "We present Atlas, a carefully designed and pre-trained retrieval augmented language model able to learn knowledge intensive tasks with very few training examples. We jointly pre-train the retriever and language model using a novel attention distillation mechanism.",
        "citation_count": 892,
        "url": "https://www.semanticscholar.org/paper/Atlas-Izacard/a1c23b7a5a6d29e3e48b5b723c9c1f8e93fb1db3",
        "venue": "JMLR",
        "paper_id": "a1c23b7a",
    },
    {
        "title": "When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories",
        "authors": ["Alex Mallen", "Akari Asai", "Victor Zhong", "Rajarshi Das", "Daniel Khashabi"],
        "year": 2023,
        "abstract": "We investigate when retrieval-augmentation can help LLMs. We find that the popularity of an entity is a strong indicator of whether the LLM will rely on parametric or non-parametric memory, and retrieval augmentation is most beneficial for less popular entities.",
        "citation_count": 567,
        "url": "https://www.semanticscholar.org/paper/Mallen-Asai/b4a5c97e6c3f4a0e86e7d2b1d3c5f8a7e9b0d2c4",
        "venue": "ACL",
        "paper_id": "b4a5c97e",
    },
]
MULTI_AGENT_ARXIV_PAPERS = [
    {
        "title": "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
        "authors": ["Qingyun Wu", "Gagan Bansal", "Jieyu Zhang", "Yiran Wu", "Beibin Li"],
        "year": 2023,
        "abstract": "AutoGen is an open-source framework that allows developers to build LLM applications via multiple agents that can converse with each other to solve tasks. AutoGen agents are customizable, conversable, and seamlessly allow human participation.",
        "arxiv_id": "2308.08155",
        "url": "http://arxiv.org/abs/2308.08155",
        "categories": ["cs.AI", "cs.MA"],
    },
    {
        "title": "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework",
        "authors": ["Sirui Hong", "Mingchen Zhuge", "Jonathan Chen", "Xiawu Zheng", "Yuheng Cheng"],
        "year": 2023,
        "abstract": "We introduce MetaGPT, an innovative meta-programming framework incorporating efficient human workflows into LLM-based multi-agent collaboration. MetaGPT encodes Standardized Operating Procedures into prompt sequences for more streamlined workflows.",
        "arxiv_id": "2308.00352",
        "url": "http://arxiv.org/abs/2308.00352",
        "categories": ["cs.AI", "cs.SE"],
    },
    {
        "title": "Communicative Agents for Software Development",
        "authors": ["Chen Qian", "Xin Cong", "Wei Liu", "Cheng Yang", "Weize Chen"],
        "year": 2023,
        "abstract": "We present ChatDev, a chat-powered software development framework in which specialized agents collaboratively participate in various phases of the software development lifecycle through natural language communication.",
        "arxiv_id": "2307.07924",
        "url": "http://arxiv.org/abs/2307.07924",
        "categories": ["cs.SE", "cs.AI"],
    },
]
LLM_KG_ARXIV_PAPERS = [
    {
        "title": "Unifying Large Language Models and Knowledge Graphs: A Roadmap",
        "authors": ["Shirui Pan", "Linhao Luo", "Yufei Wang", "Chen Chen", "Jiapu Wang"],
        "year": 2024,
        "abstract": "This paper presents a forward-looking roadmap for the unification of LLMs and KGs. We review existing efforts in three main categories: KG-enhanced LLMs, LLM-augmented KGs, and synergized LLMs + KGs, analyzing their technical contributions and limitations.",
        "arxiv_id": "2306.08302",
        "url": "http://arxiv.org/abs/2306.08302",
        "categories": ["cs.CL", "cs.AI"],
    },
    {
        "title": "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph",
        "authors": ["Jiashuo Sun", "Chengjin Xu", "Lumingyuan Tang", "Saizhuo Wang", "Chen Lin"],
        "year": 2024,
        "abstract": "We propose Think-on-Graph (ToG), a new paradigm that allows LLMs to perform deep and responsible reasoning on knowledge graphs. ToG treats the LLM as an agent that iteratively explores related entities and relations on KGs.",
        "arxiv_id": "2307.07697",
        "url": "http://arxiv.org/abs/2307.07697",
        "categories": ["cs.CL", "cs.AI"],
    },
    {
        "title": "KnowledGPT: Enhancing Large Language Models with Retrieval and Storage Access on Knowledge Bases",
        "authors": ["Xin Wang", "Yasheng Wang", "Fei Mi", "Pengcheng He", "Lifeng Shang"],
        "year": 2023,
        "abstract": "We introduce KnowledGPT, a framework that bridges LLMs and various knowledge bases to enhance factual accuracy. It includes a search-then-generate pipeline with code generation for complex KG queries and a knowledge storage for personalized retrieval.",
        "arxiv_id": "2308.11761",
        "url": "http://arxiv.org/abs/2308.11761",
        "categories": ["cs.CL"],
    },
]
GENERAL_ARXIV_PAPERS = [
    {
        "title": "A Survey on Large Language Model based Autonomous Agents",
        "authors": ["Lei Wang", "Chen Ma", "Xueyang Feng", "Zeyu Zhang", "Hao Yang"],
        "year": 2024,
        "abstract": "This paper presents a comprehensive survey on LLM-based autonomous agents. We propose a unified framework consisting of profiling, memory, planning, and action modules, and systematically review the literature on agent construction and evaluation.",
        "arxiv_id": "2308.11432",
        "url": "http://arxiv.org/abs/2308.11432",
        "categories": ["cs.AI"],
    },
    {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": ["Jason Wei", "Xuezhi Wang", "Dale Schuurmans", "Maarten Bosma", "Brian Ichter"],
        "year": 2022,
        "abstract": "We explore how generating a chain of thought -- a series of intermediate reasoning steps -- significantly improves the ability of large language models to perform complex reasoning.",
        "arxiv_id": "2201.11903",
        "url": "http://arxiv.org/abs/2201.11903",
        "categories": ["cs.CL", "cs.AI"],
    },
    {
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "authors": ["Shunyu Yao", "Jeffrey Zhao", "Dian Yu", "Nan Du", "Izhak Shafran"],
        "year": 2023,
        "abstract": "We introduce ReAct, a general paradigm that synergizes reasoning and acting in language models for solving diverse language reasoning and decision making tasks. ReAct prompts LLMs to generate both verbal reasoning traces and task-specific actions in an interleaved manner.",
        "arxiv_id": "2210.03629",
        "url": "http://arxiv.org/abs/2210.03629",
        "categories": ["cs.CL", "cs.AI"],
    },
]
def get_mock_papers(query: str, source: str = "arxiv") -> list[dict] | None:
    q = query.lower()
    if any(kw in q for kw in ["retrieval augmented", "rag", "retrieval-augmented"]):
        return RAG_ARXIV_PAPERS if source == "arxiv" else RAG_SCHOLAR_PAPERS
    elif any(kw in q for kw in ["multi-agent", "multi agent", "autogen", "collaboration"]):
        return MULTI_AGENT_ARXIV_PAPERS if source == "arxiv" else None
    elif any(kw in q for kw in ["knowledge graph", "knowledge base", "llm kg", "llm and kg"]):
        return LLM_KG_ARXIV_PAPERS if source == "arxiv" else None
    else:
        return GENERAL_ARXIV_PAPERS if source == "arxiv" else RAG_SCHOLAR_PAPERS
