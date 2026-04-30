# Mapeo de términos para expansión semántica y normalización


GLOSARIO_ACRONIMOS = {
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning",
    "RAG": "Retrieval Augmented Generation",
    "LLM": "Large Language Model",
    "NLP": "Natural Language Processing",
    "DL": "Deep Learning",
    "GenAI": "Generative Artificial Intelligence",
    "RLHF": "Reinforcement Learning from Human Feedback",
    "LoRA": "Low-Rank Adaptation",
    "EDA": "Event-Driven Architecture",
    "CI/CD": "Continuous Integration Continuous Deployment",
    "API": "Application Programming Interface",
    "SDK": "Software Development Kit",
    "IaC": "Infrastructure as Code",
    "SaaS": "Software as a Service",
    "PaaS": "Platform as a Service"
}

CATEGORIES_LIST = [
    "React", "Angular", "Node.js", "Next.js", "TypeScript",
    "AI & Machine Learning", "LLMs", "RAG", "Data Science",
    "DevOps", "Kubernetes", "AWS", "Terraform", "Infrastructure as Code",
    "E2E Testing", "Playwright", "Cypress", "Unit Testing",
    "State Management", "Web Components", "Performance", "Module Federation",
    "Open Source", "Project Management", "Product Design", "UX/UI", "Agile"
]


TECH_MAPPING = {
    "Frontend": {
        "JS": "JavaScript, ES6+, ECMAScript, Web APIs",
        "TS": "TypeScript, Static Typing, Type Safety",
        "A11y": "Accessibility, WCAG, ARIA, Screen Readers",
        "i18n": "Internationalization, Localization, l10n, Translation",
        "E2E": "End-to-End Testing, Playwright, Cypress, Vitest", # Vitest es clave ahora
        "State Mgmt": "Redux, NgRx, Signals, Zustand, Context API, Store",
        "Styles": "CSS, Tailwind, SASS, Design Systems, Shadow DOM"
    },
    "Backend & Infrastructure": {
        "K8s": "Kubernetes, Orchestration, Clusters, Containers, Helm",
        "K3s": "Lightweight Kubernetes, Edge Computing",
        "CI/CD": "Continuous Integration, Continuous Deployment, Pipelines, Automation, GitHub Actions",
        "Serverless": "AWS Lambda, Cloud Functions, FaaS, Cold Start",
        "IaC": "Infrastructure as Code, Terraform, Pulumi, CloudFormation",
        "Auth": "Authentication, Authorization, OAuth2, JWT, RBAC, IAM",
        "API": "REST, GraphQL, gRPC, Webhooks"
    },
    "AI & Data": {
        "LLM": "Large Language Models, GPT, Claude, Llama, Foundation Models",
        "RAG": "Retrieval-Augmented Generation, Vector Search, Knowledge Base, Context Injection",
        "MCP": "Model Context Protocol, AI Tools, Agentic Workflows, MCP Servers",
        "GenAI": "Generative AI, Artificial Intelligence, Multimodal AI",
        "NLP": "Natural Language Processing, Tokenization, Sentiment Analysis",
        "Vector DB": "Chroma, Pinecone, Milvus, Vector Store, Embeddings, Similarity Search"
    },
    "Architecture": {
        "EDA": "Event-Driven Architecture, Pub/Sub, Message Queues, Kafka, RabbitMQ",
        "DDD": "Domain-Driven Design, Bounded Context, Ubiquitous Language",
        "Microservices": "Distributed Systems, Service Mesh, Sidecar, API Gateway"
    }
}

# Función para formatear el mapeo para el prompt
def get_tech_mapping_str():
    mapping_str = ""
    for category, terms in TECH_MAPPING.items():
        mapping_str += f"\n--- {category} ---\n"
        for acronym, full_term in terms.items():
            mapping_str += f"- {acronym}: {full_term}\n"
    return mapping_str