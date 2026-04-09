"""
Smoke test for the vLLM Track B generator endpoint.

Run with:
    python tests/smoke_test.py

The vLLM server must be running on localhost:8001 before executing this script.
"""

import json
import sys

import requests

ENDPOINT = "http://localhost:8001/v1/chat/completions"

PAYLOAD = {
    "model": "/models/fused_model",
    "messages": [
        {
            "role": "system",
            "content": (
                "You are AutoMend. Available tools: scale_deployment, restart_rollout, "
                "undo_rollout, send_notification, request_approval, trigger_webhook. "
                "Respond with a JSON workflow."
            ),
        },
        {
            "role": "user",
            "content": "Scale my deployment to 5 replicas",
        },
    ],
    "temperature": 0.0,
    "max_tokens": 512,
}


def run():
    print(f"POST {ENDPOINT}")
    print()

    try:
        response = requests.post(ENDPOINT, json=PAYLOAD, timeout=60)
    except requests.exceptions.ConnectionError:
        print("FAIL: could not connect to vLLM server at localhost:8001")
        print("Make sure the server is running: docker compose up vllm-generator")
        sys.exit(1)

    # Check 1: HTTP 200
    assert response.status_code == 200, (
        f"FAIL: expected HTTP 200, got {response.status_code}\n{response.text}"
    )
    print(f"[PASS] HTTP status: {response.status_code}")

    body = response.json()

    # Check 2: choices array present and non-empty
    assert "choices" in body and len(body["choices"]) > 0, (
        f"FAIL: response has no choices\n{json.dumps(body, indent=2)}"
    )
    print(f"[PASS] choices array present ({len(body['choices'])} item(s))")

    content = body["choices"][0]["message"]["content"]
    print()
    print("Raw assistant content:")
    print(content)
    print()

    # Check 3: content is valid JSON
    try:
        parsed = json.loads(content.strip())
    except json.JSONDecodeError as exc:
        print(f"FAIL: assistant content is not valid JSON -- {exc}")
        sys.exit(1)
    print("[PASS] assistant content is valid JSON")

    # Check 4: JSON contains workflow.steps
    assert "workflow" in parsed, (
        f"FAIL: JSON missing 'workflow' key\n{json.dumps(parsed, indent=2)}"
    )
    assert "steps" in parsed["workflow"], (
        f"FAIL: 'workflow' missing 'steps' key\n{json.dumps(parsed, indent=2)}"
    )
    print("[PASS] JSON contains workflow.steps")

    print()
    print("Full parsed workflow:")
    print(json.dumps(parsed, indent=2))

    print()
    print("Full raw response body:")
    print(json.dumps(body, indent=2))

    print()
    print("All checks passed.")


if __name__ == "__main__":
    run()
