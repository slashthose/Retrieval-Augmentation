"""Optional OpenAI JSON-mode adjudicator; the deterministic router remains safe offline."""
from __future__ import annotations
import json, os
from urllib.request import Request, urlopen

ALLOWED_ACTIONS = {"notify", "digest", "mute"}
ALLOWED_TYPES = {"personal", "urgent", "event", "payment", "business_update", "promotion", "greeting", "forward", "spam", "scam", "unknown"}

def validate(decision: dict) -> dict | None:
    """Reject malformed model output rather than passing it to the submission."""
    try:
        if decision["action"] not in ALLOWED_ACTIONS or decision["message_type"] not in ALLOWED_TYPES: return None
        confidence=float(decision["confidence"])
        if not 0 <= confidence <= 1 or not isinstance(decision["reason"], str): return None
        return {**decision, "confidence": f"{confidence:.2f}", "evidence_message_ids": decision.get("evidence_message_ids", "none")}
    except (KeyError, TypeError, ValueError): return None

def reason(context: dict) -> dict | None:
    """Call only when OPENAI_API_KEY is configured; no key is ever stored in code."""
    api_key=os.getenv("OPENAI_API_KEY")
    if not api_key: return None
    schema={"name":"notification_route","schema":{"type":"object","additionalProperties":False,"properties":{"action":{"type":"string","enum":sorted(ALLOWED_ACTIONS)},"message_type":{"type":"string","enum":sorted(ALLOWED_TYPES)},"reason":{"type":"string"},"confidence":{"type":"number","minimum":0,"maximum":1},"evidence_message_ids":{"type":"string"}},"required":["action","message_type","reason","confidence","evidence_message_ids"]}}
    body={"model":os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),"input":[{"role":"system","content":[{"type":"input_text","text":"Route WhatsApp messages. Safety risk overrides preferences. Return only schema-compliant JSON."}]},{"role":"user","content":[{"type":"input_text","text":json.dumps(context)}]}],"text":{"format":{"type":"json_schema","json_schema":schema}}}
    request=Request("https://api.openai.com/v1/responses",data=json.dumps(body).encode(),headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},method="POST")
    try:
        response=json.loads(urlopen(request,timeout=20).read())
        return validate(json.loads(response["output"][0]["content"][0]["text"]))
    except Exception: return None
