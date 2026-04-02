import json
import logging
from typing import List, Dict, Any, Union, TypedDict
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from agents import PerceptorAgent, ExecutorAgent, EvaluatorAgent, VerifierAgent, MedCATPipeline
from nlp_model_stub import FineTunedClinicalBERT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load RAG content (Guidelines)
with open('sepsis_guidelines.txt', 'r') as f:
    guidelines_text = f.read()

# Fallback mechanism if Ollama is not running in CI
class MockOllama:
    def invoke(self, prompt: str) -> str:
        if "evaluator agent" in prompt.lower():
            return "Evaluation Summary: The proposed treatment plan appropriately addresses the identified Sepsis and Hypotension conditions according to Sepsis-3 guidelines."
        return '["Order Lactate Redraw", "Administer 30mL/kg Crystalloid"]'

try:
    import requests
    requests.get('http://localhost:11434/api/tags', timeout=1)
    llm = Ollama(model="gemma")
except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
    logger.warning("Ollama not reachable, using MockOllama for CI testing.")
    llm = MockOllama()

# Initialize NLP Model (Method 3)
nlp_model = FineTunedClinicalBERT()

# Initialize MedCAT Pipeline
medcat_pipeline = MedCATPipeline()

# State Management
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, SystemMessage]]
    subject_id: str
    visit_id: str
    clinical_data: Dict[str, Any]
    clinical_note: str
    extracted_entities: List[Dict[str, str]]
    nlp_sepsis_score: float
    alert_triggered: bool
    plan: List[str]
    evaluation_result: str
    execution_result: List[str]
    explanation: str

# 1. NLP Perceptor Node (Method 3 + MedCAT)
def nlp_perceptor_node(state: AgentState):
    logger.info("--- NLP PERCEPTOR AGENT (Bio_ClinicalBERT & MedCAT) ---")
    note = state.get('clinical_note', '')

    # Extract entities using MedCAT
    entities = medcat_pipeline.get_entities(note)

    # Use the NLP model to predict sepsis probability
    score = nlp_model.predict_sepsis_probability(note)

    # Trigger alert if score is high OR if we found severe sepsis / septic shock concepts
    critical_cuis = {"C1090680", "C0151744"} # Severe Sepsis, Septic Shock
    found_critical_cui = any(ent['cui'] in critical_cuis for ent in entities)

    triggered = score >= 0.5 or found_critical_cui

    if triggered:
        logger.info(f"NLP Alert triggered for {state['subject_id']} (Score: {score:.2f}, Entities found: {len(entities)})")

    return {"nlp_sepsis_score": score, "extracted_entities": entities, "alert_triggered": triggered}

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

# 3. Evaluator Node (LLM Judge)
def evaluator_node(state: AgentState):
    logger.info("--- EVALUATOR AGENT (LLM Judge) ---")
    if not state['plan']:
        return {"evaluation_result": "No plan to evaluate."}

    evaluator = EvaluatorAgent(llm=llm)
    result = evaluator.evaluate(state['plan'], state['extracted_entities'])
    return {"evaluation_result": result}

# 4. Executor Node
def executor_node(state: AgentState):
    logger.info("--- EXECUTOR AGENT ---")
    if not state['plan']:
        return {"execution_result": ["No actions required."]}

    executor = ExecutorAgent()
    results = executor.execute_orders(state['plan'], state['visit_id'])
    return {"execution_result": results}

# 5. Verifier Node
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
    workflow.add_node("nlp_perceptor", nlp_perceptor_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("verifier", verifier_node)

    # Add Edges
    workflow.set_entry_point("nlp_perceptor")

    # Conditional logic after Perceptor
    def should_continue(state: AgentState):
        if state['alert_triggered']:
            return "planner"
        else:
            return END

    workflow.add_conditional_edges(
        "nlp_perceptor",
        should_continue,
        {
            "planner": "planner",
            END: END
        }
    )

    workflow.add_edge("planner", "evaluator")
    workflow.add_edge("evaluator", "executor")
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

            # Extract clinical note
            notes = visit.get('clinical_notes', [])
            note_text = notes[0]['text'] if len(notes) > 0 else "No clinical note available."

            initial_state = {
                "subject_id": patient['subject_id'],
                "visit_id": visit['hadm_id'],
                "clinical_data": {
                    "HR": hr,
                    "RR": rr,
                    "Temp": temp,
                    "Lactate": lactate
                },
                "clinical_note": note_text,
                "extracted_entities": [],
                "nlp_sepsis_score": 0.0,
                "alert_triggered": False,
                "plan": [],
                "evaluation_result": "",
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
                print(f"Extracted CUI Entities: {result['extracted_entities']}")
                print(f"Plan: {result['plan']}")
                print(f"LLM Judge Evaluation:\n{result['evaluation_result']}")
                print(f"Execution: {result['execution_result']}")
                print(f"Explanation:\n{result['explanation']}")
            else:
                pass

    # Save all results for evaluation
    with open('output/orchestration_results.json', 'w') as f:
        # Convert non-serializable objects to string/dict representation if necessary
        json.dump(all_results, f, default=str, indent=2)

if __name__ == "__main__":
    run_orchestrator()
