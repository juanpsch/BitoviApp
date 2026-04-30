from langgraph.graph import StateGraph, START, END
from .state import AgentState
from langgraph.prebuilt import ToolNode, tools_condition
from .nodes import (
    intent_analyzer_node,
    analysis_node,
    retrieval_node,
    generator_node,
    expansion_node,
    grade_retrieval,
    listing_generator_node,
    query_optimizer_node
)
from scripts.my_tools import retrieve_docs
from agent.routers import route_generator


# =============================================================================
# ROUTERS LÓGICOS
# =============================================================================




# =============================================================================
# CONFIGURACIÓN DEL WORKFLOW
# =============================================================================


# Configuración del Grafo

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Definimos una función mínima para el nodo router
# Solo sirve para que el grafo tenga un punto físico donde aterrizar
def router_node(state: AgentState):
    return state

workflow = StateGraph(AgentState)

# 2. Agregamos los nodos (Incluyendo el nodo router)
workflow.add_node("intent_analyzer", intent_analyzer_node)
workflow.add_node("analizar", analysis_node)
workflow.add_node("query_optimizer", query_optimizer_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("tools", ToolNode([retrieve_docs]))
workflow.add_node("expansion", expansion_node)
workflow.add_node("generator", generator_node)
workflow.add_node("listing_generator", listing_generator_node)

# REGISTRAMOS EL NODO FÍSICO
workflow.add_node("puntos_de_decision", router_node)

# 3. Flujo Inicial
workflow.add_edge(START, "intent_analyzer")
workflow.add_edge("intent_analyzer", "analizar")
workflow.add_edge("analizar", "query_optimizer")
workflow.add_edge("query_optimizer", "retrieval")

# 4. De Retrieval a la Pasarela de decisión
workflow.add_conditional_edges(
    "retrieval",
    tools_condition,
    {
        "tools": "tools",
        "__end__": "puntos_de_decision" # <--- Aterriza en el nodo real
    }
)

# 5. De Tools a la Pasarela de decisión
workflow.add_conditional_edges(
    "tools",
    grade_retrieval,
    {
        "expand": "expansion",
        "end": "puntos_de_decision",
        "fail": "puntos_de_decision"
    }
)

# 6. EL ROUTER REAL (Lógica de salida de la pasarela)
workflow.add_conditional_edges(
    "puntos_de_decision", # Ahora sí existe porque lo agregamos con add_node
    route_generator,      # Tu función lógica
    {
        "generator": "generator",
        "listing_generator": "listing_generator"
    }
)

workflow.add_edge("expansion", "retrieval")
workflow.add_edge("generator", END)
workflow.add_edge("listing_generator", END)

app = workflow.compile()


