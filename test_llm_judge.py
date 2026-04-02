from agents import EvaluatorAgent

class MockLLM:
    def invoke(self, prompt: str) -> str:
        return "Evaluation Summary: The plan is good."

llm = MockLLM()
evaluator = EvaluatorAgent(llm)
print(evaluator.evaluate(["Test plan"], [{"cui": "C0243026", "cui_name": "Sepsis"}]))
