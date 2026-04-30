from typing import List, Optional, Annotated, Dict, Any, Literal
import operator
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from enum import Enum
from scripts.schemas import TaskType, IntentOutput


# =============================================================================
# State Definition
# =============================================================================

class AgentState(TypedDict):
    # 'operator.add' permite que los mensajes se acumulen en lugar de sobrescribirse
    messages: Annotated[list, operator.add]
    
    # Aquí guardamos la decisión de 'fast' o 'convencional'
    # La definimos como str para mayor compatibilidad con los prompts
    selected_route: str
    search_params: dict
    retry_count: int  # Para controlar el bucle
    task_type: TaskType    
    search_query: str
