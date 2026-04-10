from app.guardrails import parse_llm_output
from app.schemas.workflow import WorkflowResponse

def test_valid():
    raw = '''
    {
      "workflow": {
        "steps": [
          {
            "step_id": 1,
            "tool": "restart_rollout",
            "params": {
              "namespace": "prod",
              "deployment_name": "api"
            }
          }
        ]
      }
    }
    '''

    parsed = parse_llm_output(raw)
    validated = WorkflowResponse.model_validate(parsed)

    assert validated is not None
    print("VALID TEST PASSED")