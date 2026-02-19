from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1) Define the state shape
class State(TypedDict):
    count: int

# 2) Create the graph builder
builder = StateGraph(State)

# 3) Define a node (a pure state update function)
def increment(state: State) -> State:
    return {"count": state["count"] + 1}

# 4) Add the node to the graph
builder.add_node("increment", increment)

# 5) Define edges: start -> increment -> end
builder.set_entry_point("increment")
builder.add_edge("increment", END)

# 6) Compile into a runnable graph
graph = builder.compile()

# 7) Invoke the graph with initial state
result = graph.invoke({"count": 0})

print(result)  # {'count': 1}
