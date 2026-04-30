from enum import Enum 


from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator


from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum

class RetrievalRoute(str, Enum):
    FAST = "Fast"                # Pure metadata: Chronological order/Counts
    CONVENCIONAL = "Convencional"  # Semantic Search: Topics/Concepts/Knowledge

class RestrievalStragegy(BaseModel):
    route: RetrievalRoute = Field('Convencional',
        description="The retrieval strategy. 'Fast' for purely chronological requests or generic listings. 'Convencional' for topic-based questions or conceptual searches."
    )

    justification: str = Field(
        description="Technical explanation for choosing this route based on user intent."
    )

class SearchQueryOutput(BaseModel):
    search_query: str = Field(description="2-6 technical keywords for BM25/Vector search. No Bitovi, no fluff.")

class TaskType(str, Enum):
    LISTING = "listing"        # "Muéstrame todos", "Dame una lista de...", "Qué hay de..."
    SINTESIS = "synthesis"     # "Cómo se hace...", "Qué es...", "Explícame..."
    REASONING = "reasoning"    # "Por qué...", "Compara X con Y..."

class IntentOutput(BaseModel):
    intent: TaskType = Field(
        description="'synthesis' is for extracting information, summaries, or explanations FROM documents. "
                    "'listing' is for discovering OR pointing to the documents themselves."
    )
    
    reasoning: str = Field(
        description="A brief explanation of why this intent was chosen based on specific keywords "
                    "or the grammatical structure of the user's request (e.g., presence of 'all', 'how', or 'why')."
    )


class SearchControl(BaseModel):
    sort_by: Optional[Literal["date_ts"]] = Field(
        None, 
        description="Si el usuario pide lo 'último', 'reciente', 'nuevo' o 'last/latest', debe ser 'date_ts'. De lo contrario, None."
    )
    
    top_k: int = Field(        
        ge=1, 
        le=50, 
        description="Cantidad exacta de documentos a recuperar. Si el usuario pide un número (ej: 'last X'), poner X. Si pide 'lo último' (en singular), poner 1. Si no especifica cantidad, el default."
    )

    reasoning: str = Field(
        description="Breve explicación de por qué se eligió ese orden y esa cantidad."
    )



class BlogCategory(str, Enum):
    # --- Frontend ---
    REACT = "React"
    ANGULAR = "Angular"
    NODE_JS = "Node.js"
    NEXT_JS = "Next.js"
    TYPESCRIPT = "TypeScript"
    STATE_MANAGEMENT = "State Management"
    WEB_COMPONENTS = "Web Components"
    PERFORMANCE = "Performance"
    MODULE_FEDERATION = "Module Federation"
    
    # --- AI & Data ---    
    AI_ML = "AI & Machine Learning"
    LLMS = "LLMs"
    RAG = "RAG"
    DATA_SCIENCE = "Data Science"
    
    # --- DevOps & Infrastructure ---
    DEVOPS = "DevOps"
    KUBERNETES = "Kubernetes"
    AWS = "AWS"
    TERRAFORM = "Terraform"
    IAC = "Infrastructure as Code"
    
    # --- Testing ---
    TESTING = "E2E Testing"
    PLAYWRIGHT = "Playwright"
    CYPRESS = "Cypress"
    UNIT_TESTING = "Unit Testing"
    
    # --- Management & Architecture ---
    PROJECT_MANAGEMENT = "Project Management"
    PRODUCT_DESIGN = "Product Design"
    UX_UI = "UX/UI"
    AGILE = "Agile"
    ARCHITECTURE = "Architecture"
    OPEN_SOURCE = "Open Source"

    @property
    def tech_mapping(self) -> str:
        mappings = {
            # Frontend
            BlogCategory.REACT: "React.js, Hooks, JSX, React Router, Context API",
            BlogCategory.ANGULAR: "Angular, RxJS, NgRx, Templates, Dependency Injection",
            BlogCategory.NODE_JS: "Node.js, Express, Fastify, Backend JS, NPM, Bun",
            BlogCategory.NEXT_JS: "Next.js, SSR, SSG, App Router, Vercel",
            BlogCategory.TYPESCRIPT: "TS, TypeScript, Interface, Generics, Type Safety",
            BlogCategory.STATE_MANAGEMENT: "Signals, Redux, Zustand, MobX, State Mgmt, Store",
            BlogCategory.WEB_COMPONENTS: "Lit, Stencil, Custom Elements, Shadow DOM",
            BlogCategory.PERFORMANCE: "Core Web Vitals, LCP, CLS, Optimization, Bundle Size",
            BlogCategory.MODULE_FEDERATION: "Micro-frontends, MFE, Webpack Federation, Remote Modules",
            
            # AI & Data
            BlogCategory.AI_ML: "AI, IA, Artificial Intelligence, Inteligencia Artificial, Generative AI, Machine Learning",
            BlogCategory.LLMS: "Large Language Models, GPT, Claude, Llama, Fine-tuning",
            BlogCategory.RAG: "Retrieval Augmented Generation, Vector Search, Chroma, Pinecone, Embeddings, Semantic Search, MCP, Model Context Protocol",
            BlogCategory.DATA_SCIENCE: "Pandas, Jupyter, Data Analysis, Visualization, Python Data",
            
            # DevOps & Infra
            BlogCategory.DEVOPS: "CI/CD, Pipelines, GitHub Actions, Automation, StackStorm",
            BlogCategory.KUBERNETES: "K8s, Clusters, Helm, K3s, Orchestration, Containers",
            BlogCategory.AWS: "Amazon Web Services, S3, EC2, Cloud, Lambda",
            BlogCategory.TERRAFORM: "Terraform, HCL, Infrastructure as Code, TF Plans",
            BlogCategory.IAC: "Pulumi, CloudFormation, IaC, Resource Provisioning",
            
            # Testing
            BlogCategory.TESTING: "E2E, End-to-End, Integration Testing, QA",
            BlogCategory.PLAYWRIGHT: "Playwright, Browser Automation, E2E Scripts",
            BlogCategory.CYPRESS: "Cypress.io, Dashboard, E2E Testing",
            BlogCategory.UNIT_TESTING: "Jest, Vitest, Mocha, Component Testing",
            
            # Management & Architecture
            BlogCategory.PROJECT_MANAGEMENT: "Roadmaps, Resource Planning, Delivery, Stakeholders",
            BlogCategory.PRODUCT_DESIGN: "Product Strategy, Discovery, User Research",
            BlogCategory.UX_UI: "Design Systems, CSS, Tailwind, Figma, Styles, UI/UX",
            BlogCategory.AGILE: "Scrum, Kanban, Sprints, Retrospectives, Lean",
            BlogCategory.ARCHITECTURE: "EDA, Event-Driven, DDD, Microservices, Service Mesh, Kafka",
            BlogCategory.OPEN_SOURCE: "OSS, Contributing, Community, GitHub Repos"
        }
        return mappings.get(self, self.value)

    @classmethod
    def get_prompt_context(cls):
        # Mismo método de antes para generar el texto del prompt...
        output = []
        for c in cls:
            output.append(f"- {c.value}: {c.tech_mapping}")
        return "\n".join(output)

    @classmethod
    def get_all_values(cls) -> List[str]:
        """
        Retorna la lista oficial de categorías para que el Agente
        sepa exactamente qué opciones tiene permitidas.
        """
        return [c.value for c in cls]


# ============================================================================
# PYDANTIC MODELS
# ============================================================================
# 
class ChunkMetadata(BaseModel):
    # Usamos el Enum directamente. 
    # Al tener use_enum_values=True, Pydantic lo validará contra el Enum 
    # pero devolverá el string (ej. "DevOps") al llamar a .dict() o .json()
    category: Optional[BlogCategory] = Field(
        default=None, 
        description="The technical category of the post (e.g., 'DevOps', 'AI & Machine Learning')"
    )
    
    year: Optional[int] = Field(
        default=None, 
        ge=2000, 
        le=2030, 
        description="The year the article was published"
    )
    
    author: Optional[str] = Field(
        default=None, 
        description="The name of the blog post author"
    )

    model_config = {
        "use_enum_values": True,
        "populate_by_name": True
    }


# class RankingKeywords(BaseModel):
    # keywords: List[str] = Field(..., description="Generate Exactly 5 technical keywords related to user query", min_length=0, max_length=5)    

class RankingKeywords(BaseModel):
    keywords: List[str] = Field(
        ...,
        description="List of exactly 5 technical keywords derived from the mapping. "
                    "If the query is irrelevant, return an empty list.",
        min_length=0, # por si la query es off-topic,         
        max_length=5 
    )

    @field_validator("keywords")
    @classmethod
    def validate_no_forbidden_terms(cls, v: List[str]) -> List[str]:
        # Fail-safe por si el LLM ignora el prompt y mete el nombre de la empresa
        forbidden = {"bitovi"}
        filtered = [k for k in v if k.lower() not in forbidden]
        
        # Eliminar duplicados manteniendo el orden
        return list(dict.fromkeys(filtered))