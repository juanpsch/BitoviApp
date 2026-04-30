from agent.state import AgentState
from scripts.schemas import TaskType

# agent/routers.py
def route_generator(state: AgentState):
    task_type = state.get("task_type")
    # Es buena práctica imprimirlo para debuguear
    print(f"--- [ROUTER] Decidiendo generador para: {task_type} ---")
    
    if task_type == TaskType.LISTING:
        return "listing_generator"
    return "generator"