import json
import logging
from typing import List, Dict, Any, Union, TypedDict
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from agents import PerceptorAgent, ExecutorAgent, VerifierAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load RAG content (Guidelines)
with open('sepsis_guidelines.txt', 'r') as f:
    guidelines_text = f.read()

# Initialize LLM
llm = Ollama(model="gemma")

# State Management
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, SystemMessage]]
    subject_id: str
    visit_id: str
    clinical_data: Dict[str, Any]
    alert_triggered: bool
    plan: List[str]
    execution_result: List[str]
    explanation: str

# 1. Perceptor Node
def perceptor_node(state: AgentState):
    logger.info("--- PERCEPTOR AGENT ---")
    perceptor = PerceptorAgent()
    # We construct the input format expected by Perceptor
    patient_data = {
        'subject_id': state['subject_id'],
        'visits': [{
            'hadm_id': state['visit_id'],
            'admittime': 'unknown',
            'events': [
                {'itemid': '8867-4', 'valuenum': state['clinical_data']['HR']},
                {'itemid': '9279-1', 'valuenum': state['clinical_data']['RR']},
                {'itemid': '8310-5', 'valuenum': state['clinical_data']['Temp']},
                {'itemid': '32693-4', 'valuenum': state['clinical_data']['Lactate']}
            ]
        }]
    }

    alerts = perceptor.monitor(patient_data)
    triggered = len(alerts) > 0
    return {"alert_triggered": triggered}

# 2. Planner Node (RAG + LLM)
def planner_node(state: AgentState):
    logger.info("--- PLANNER AGENT ---")
    if not state['alert_triggered']:
        return {"plan": []}

    # RAG Context
    context = guidelines_text

    # Prompt
    prompt = f"""
    You are a medical planner agent. Based on the following Sepsis-3 guidelines and the patient's data, generate a list of specific clinical orders to be executed.

    Guidelines:
    {context}

    Patient Data:
    HR: {state['clinical_data']['HR']}
    RR: {state['clinical_data']['RR']}
    Temp: {state['clinical_data']['Temp']}
    Lactate: {state['clinical_data']['Lactate']}

    Output ONLY a JSON list of strings representing the orders (e.g., ["Order Lactate Redraw", "Administer 30mL/kg Crystalloid"]). Do not include any other text.
    """

    response = llm.invoke(prompt)

    # Parse LLM response (robustness check needed in prod)
    try:
        # Simple string cleaning to extract the list
        start = response.find('[')
        end = response.rfind(']') + 1
        if start != -1 and end != -1:
            plan_json = response[start:end]
            plan = json.loads(plan_json)
        else:
            plan = ["Consult Specialist (Parse Error)"]
    except Exception as e:
        logger.error(f"Failed to parse plan: {e}")
        plan = ["Consult Specialist (JSON Error)"]

    return {"plan": plan}

# 3. Executor Node
def executor_node(state: AgentState):
    logger.info("--- EXECUTOR AGENT ---")
    if not state['plan']:
        return {"execution_result": ["No actions required."]}

    executor = ExecutorAgent()
    results = executor.execute_orders(state['plan'], state['visit_id'])
    return {"execution_result": results}

# 4. Verifier Node
def verifier_node(state: AgentState):
    logger.info("--- VERIFIER AGENT ---")
    if not state['alert_triggered']:
        return {"explanation": "No sepsis alert triggered."}

    verifier = VerifierAgent()
    # Create the alert structure needed by Verifier
    alert = {
        'clinical_data': state['clinical_data']
    }
    explanation = verifier.explain(alert)
    return {"explanation": explanation}

# Build the Graph
def create_agent_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("perceptor", perceptor_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("verifier", verifier_node)

    # Add Edges
    workflow.set_entry_point("perceptor")

    # Conditional logic after Perceptor
    def should_continue(state: AgentState):
        if state['alert_triggered']:
            return "planner"
        else:
            return END

    workflow.add_conditional_edges(
        "perceptor",
        should_continue,
        {
            "planner": "planner",
            END: END
        }
    )

    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "verifier")
    workflow.add_edge("verifier", END)

    return workflow.compile()

# Orchestrator Function
def run_orchestrator(patient_file='output/harmonized_data.json'):
    try:
        with open(patient_file, 'r') as f:
            patients = json.load(f)
    except FileNotFoundError:
        print("Data file not found.")
        return

    app = create_agent_graph()

    all_results = []

    for patient in patients:
        for visit in patient['visits']:
            # Prepare initial state
            hr = next((e['valuenum'] for e in visit['events'] if e['itemid'] == '8867-4'), 0)
            rr = next((e['valuenum'] for e in visit['events'] if e['itemid'] == '9279-1'), 0)
            temp = next((e['valuenum'] for e in visit['events'] if e['itemid'] == '8310-5'), 0)
            lactate = next((e['valuenum'] for e in visit['events'] if e['itemid'] == '32693-4'), 0)

            initial_state = {
                "subject_id": patient['subject_id'],
                "visit_id": visit['hadm_id'],
                "clinical_data": {
                    "HR": hr,
                    "RR": rr,
                    "Temp": temp,
                    "Lactate": lactate
                },
                "alert_triggered": False,
                "plan": [],
                "execution_result": [],
                "explanation": ""
            }

            # Run the graph
            result = app.invoke(initial_state)
            all_results.append(result)

            # Output for this patient
            if result['alert_triggered']:
                print(f"\n--- Result for Patient {result['subject_id']} (Visit {result['visit_id']}) ---")
                print(f"Alert: YES")
                print(f"Plan: {result['plan']}")
                print(f"Execution: {result['execution_result']}")
                print(f"Explanation:\n{result['explanation']}")
            else:
                # Optional: print non-septic cases to verify coverage
                pass

    # Save all results for evaluation
    with open('output/orchestration_results.json', 'w') as f:
        # Convert non-serializable objects to string/dict representation if necessary
        # Assuming simple types for now
        json.dump(all_results, f, default=str, indent=2)

if __name__ == "__main__":
    run_orchestrator()
