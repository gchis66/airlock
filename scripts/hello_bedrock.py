"""
hello_bedrock.py: smoke test for Bedrock access.

Sends one message to Claude Haiku 4.5 through the Converse API and prints the
reply, the stop reason and the token counts. Nothing else imports this file.
It proves the local credentials work and the IAM policy allows the model,
before any agent code runs.

Run from the repo root with the venv active:
    py scripts/hello_bedrock.py
"""

import boto3

# An inference profile ID, not a plain model ID. The "us." profile lets Bedrock
# serve the request from us-east-1, us-east-2 or us-west-2, so the IAM policy
# must allow the model in all three. The plain model ID is rejected for
# on-demand calls.
#
# Haiku 4.5 rather than 5.5: this account's Haiku 5.5 quota is zero and the
# increase was declined. Switching later is this one line.
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = "us-east-1"


def main() -> None:
    """Send one message to the model and print what comes back."""
    # No keys in code. boto3 reads AWS_PROFILE locally and the function's IAM
    # role on Lambda, so this line works unchanged in both places.
    client = boto3.client("bedrock-runtime", region_name=REGION)

    response = client.converse(
        modelId=MODEL_ID,
        # Content is a list of blocks (text, image, tool call, tool result).
        # That shape is what makes tool calling possible later.
        messages=[
            {
                "role": "user",
                "content": [{"text": "In one sentence, what is Amazon Bedrock?"}],
            }
        ],
        # Caps reply length, which also caps cost per call.
        inferenceConfig={"maxTokens": 200},
    )

    reply = response["output"]["message"]["content"][0]["text"]

    # "end_turn" = finished, "max_tokens" = hit the cap, "tool_use" = the
    # model is asking our code to run a tool.
    stop_reason = response["stopReason"]

    # Billed token counts. The spend meter will sum these per call.
    usage = response["usage"]

    print(f"Reply: {reply}")
    print(f"Stop reason: {stop_reason}")
    print(f"Tokens: {usage['inputTokens']} in, {usage['outputTokens']} out")


if __name__ == "__main__":
    main()
