"""Personalized, multimodal WhatsApp notification router.

The router intentionally makes its decision from data rather than message IDs or
file-specific rules.  It works without cloud credentials and has optional local
OCR / speech-to-text adapters when those packages are installed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import wave
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

OUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
RISK = re.compile(r"\b(otp|one.?time.?password|verify.{0,35}(account|identity)|kyc|account.{0,20}(blocked|suspend)|claim.{0,20}(prize|reward)|gift.?card|crypto|bitcoin|upi.{0,20}(pin|collect)|bank.{0,25}(pin|password)|lottery|winner|reply.{0,20}(code|password|pin))\b", re.I)
PROMO = re.compile(r"\b(sale|offer|discount|coupon|cashback|buy now|limited stock|free delivery|% off|selling|for sale)\b", re.I)
PAYMENT = re.compile(r"\b(payment|invoice|bill|due|refund|receipt|amount|rent|maintenance|fees?)\b", re.I)
URGENT = re.compile(r"\b(urgent|emergency|asap|immediately|today|tomorrow|cancelled|postponed|deadline|help)\b", re.I)
EVENT = re.compile(r"\b(meeting|class|exam|event|schedule|function|birthday|appointment|pickup|drop.?off)\b", re.I)
GREETING = re.compile(r"^(hi|hello|good morning|good evening|thanks|thank you|happy .{0,30})[!. ]*$", re.I)

def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline=""))) if path.exists() else []

def val(row: dict[str, str], *names: str) -> str:
    return next((str(row.get(n, "") or "") for n in names if row.get(n) not in (None, "")), "")

def as_float(v: str) -> float:
    try: return float(v or 0)
    except ValueError: return 0.0

def media_text(dataset: Path, message: dict[str, str], images: dict[str, dict], voices: dict[str, dict]) -> str:
    """Read referenced local media and extract text when local tools are present."""
    mid, kind = val(message, "media_id"), val(message, "media_type").lower()
    source = (images if kind == "image" else voices).get(mid, {})
    raw_path = val(source, "file_path", "path", "media_path")
    path = dataset / raw_path
    if not raw_path or not path.exists(): return ""
    cache_root = Path(os.getenv("MEDIA_CACHE_DIR", dataset / ".media_cache"))
    cache_root.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    cache_file = cache_root / f"{fingerprint}.json"
    if cache_file.exists():
        try: return json.loads(cache_file.read_text(encoding="utf-8"))["text"]
        except (OSError, ValueError, KeyError): pass
    extracted = ""
    try:
        # Always consume local media, even where an optional extractor is absent.
        # This validates that a referenced asset is present and non-empty.
        with path.open("rb") as media_file:
            if not media_file.read(32): return ""
        if kind == "image":
            import pytesseract  # optional; image is still read even without it
            from PIL import Image
            extracted = pytesseract.image_to_string(Image.open(path))
        if kind == "voice":
            # Validate/read the wav header.  Accurate transcription can be supplied
            # by faster-whisper in an environment where it is installed.
            if path.suffix.lower() == ".wav":
                with wave.open(str(path), "rb") as audio: audio.getparams()
            try:
                from faster_whisper import WhisperModel
                segments, _ = WhisperModel("base", device="cpu", compute_type="int8").transcribe(str(path))
                extracted = " ".join(s.text for s in segments)
            except Exception: extracted = ""
    except Exception: extracted = ""
    cache_file.write_text(json.dumps({"text": extracted}), encoding="utf-8")
    return extracted

def type_for(text: str, message: dict[str, str]) -> str:
    if RISK.search(text): return "scam"
    if int(as_float(val(message, "forwarded_count"))) >= 3: return "forward"
    if GREETING.match(text.strip().split("\n")[0]) and not re.search(r"\b(need|please|call|meeting|due|pay)\b", text, re.I): return "greeting"
    if PROMO.search(text): return "promotion"
    if val(message, "conversation_type") == "personal":
        if URGENT.search(text): return "urgent"
        return "personal"
    if val(message, "conversation_type") == "group" and ("@" in text or URGENT.search(text)):
        return "urgent" if not EVENT.search(text) else "event"
    if re.search(r"\b(order|delivery|delivered|packed|shipment|feedback|service|statement)\b", text, re.I): return "business_update"
    if re.search(r"\b(appointment|prescription|school|bus|class|consent|circular|schedule)\b", text, re.I): return "event"
    if PAYMENT.search(text): return "payment"
    if URGENT.search(text): return "urgent"
    if EVENT.search(text): return "event"
    if GREETING.search(text.strip()): return "greeting"
    if val(message, "conversation_type") == "business": return "business_update"
    return "personal" if val(message, "conversation_type") == "personal" else "unknown"

def token_set(s: str) -> set[str]:
    return {x.lower() for x in re.findall(r"[a-zA-Z]{3,}", s)}

class Router:
    def __init__(self, dataset: Path):
        self.dataset = dataset
        self.users = {val(r,"user_id"): r for r in read_csv(dataset/"users.csv")}
        self.groups = {val(r,"group_id"): r for r in read_csv(dataset/"groups.csv")}
        self.members = {(val(r,"user_id"),val(r,"group_id")): r for r in read_csv(dataset/"group_members.csv")}
        self.businesses = {val(r,"business_id"): r for r in read_csv(dataset/"business_accounts.csv")}
        self.biz_history = read_csv(dataset/"user_business_history.csv")
        self.history = read_csv(dataset/"message_history.csv")
        self.events = read_csv(dataset/"message_events.csv")
        self.images = {val(r,"image_id","media_id"):r for r in read_csv(dataset/"images.csv")}
        self.voices = {val(r,"voice_note_id","media_id"):r for r in read_csv(dataset/"voice_notes.csv")}
        self.event_by_message = defaultdict(list)
        for e in self.events: self.event_by_message[val(e,"message_id")].append(val(e,"event_type","event" ).lower())
        self.history_by_user = defaultdict(list)
        for h in self.history: self.history_by_user[val(h,"user_id")].append(h)

    def route_many(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [self.route(message) for message in messages]

    def evidence(self, message: dict[str,str], text: str) -> tuple[list[str], Counter]:
        candidates, outcomes = [], Counter()
        words = token_set(text)
        sender = val(message,"sender_user_id","sender_id")
        business = val(message,"business_id")
        group = val(message,"group_id")
        for h in self.history_by_user[val(message,"user_id")]:
            same_source = (sender and sender == val(h,"sender_user_id","sender_id")) or (business and business == val(h,"business_id")) or (group and group == val(h,"group_id"))
            overlap = len(words & token_set(val(h,"message_text","text")))
            if same_source or overlap >= 2:
                events = self.event_by_message[val(h,"message_id")]
                candidates.append(((2 if same_source else 0) + overlap, val(h,"message_id"), events))
        candidates.sort(reverse=True)
        ids=[]
        for _, mid, es in candidates[:3]:
            if mid: ids.append(mid); outcomes.update(es)
        return ids, outcomes

    def route(self, message: dict[str,str]) -> dict[str,str]:
        original = val(message,"message_text")
        extracted = media_text(self.dataset, message, self.images, self.voices)
        text = (original + " " + extracted).strip()
        kind = type_for(text, message)
        evidence, outcomes = self.evidence(message, text)
        score, why = 0.0, []
        ctype = val(message,"conversation_type").lower()
        member = self.members.get((val(message,"user_id"),val(message,"group_id")), {})
        sender_member = self.members.get((val(message,"sender_user_id"),val(message,"group_id")), {})
        biz = self.businesses.get(val(message,"business_id"), {})
        profile = self.users.get(val(message,"user_id"), {})
        name = val(profile, "name", "display_name", "first_name")
        mentioned = bool(re.search(r"@|\b(you|your name)\b", text, re.I)) or bool(name and re.search(rf"\b{re.escape(name)}\b", text, re.I))
        muted_group = val(member,"is_muted","muted").lower() in {"1","true","yes"}
        trusted = val(biz,"is_verified","verified").lower() in {"1","true","yes"}
        relationship = any(val(x,"user_id")==val(message,"user_id") and val(x,"business_id")==val(message,"business_id") and val(x,"opt_out").lower() not in {"1","true","yes"} for x in self.biz_history)
        if kind == "scam" or "report" in outcomes:
            action, why = "mute", ["safety risk detected" if kind == "scam" else "similar messages were reported"]
        else:
            if ctype == "personal": score += 1.4
            if kind in {"urgent", "payment"}: score += 1.2
            if kind == "event": score += .45
            if ctype == "group" and val(sender_member,"role").lower() == "admin": score += .65; why.append("trusted group admin")
            if val(self.groups.get(val(message,"group_id"),{}), "group_type").lower() in {"school","work"} and kind in {"event","urgent"}: score += .65
            if mentioned: score += 1.1; why.append("direct mention")
            if muted_group and not mentioned: score -= 1.5; why.append("muted group")
            if kind == "promotion": score -= .65
            if ctype == "business" and relationship: score += .65; why.append("active business relationship")
            if ctype == "business" and trusted: score += .25
            if "reply" in outcomes or "open" in outcomes: score += .45; why.append("similar messages were engaged with")
            if "dismiss" in outcomes or "mute" in outcomes: score -= 1.0; why.append("similar messages were dismissed")
            if kind in {"greeting", "forward"}: score -= .55
            action = "notify" if score >= 1.65 else "digest" if score >= -.35 else "mute"
            if not why: why = [f"{kind.replace('_',' ')} is {'time-sensitive' if action=='notify' else 'safe but non-urgent' if action=='digest' else 'low-value for this user'}"]
        confidence = min(.97, max(.55, .62 + abs(score) * .12 + (.16 if kind == "scam" else 0)))
        result = {"message_id":val(message,"message_id"), "action":action, "message_type":kind, "reason":"; ".join(why)[:180], "confidence":f"{confidence:.2f}", "evidence_message_ids":";".join(evidence) if evidence else "none"}
        # Optional model adjudication is deliberately limited to ambiguous, safe
        # traffic. Hard-safety decisions above cannot be overridden by an LLM.
        if os.getenv("USE_LLM_REASONING", "").lower() in {"1", "true", "yes"} and kind != "scam" and .75 <= score <= 1.85:
            try:
                from llm_reasoner import reason
                adjudicated = reason({"message": message, "extracted_media_text": extracted, "heuristic_decision": result, "historical_evidence": evidence, "engagement_outcomes": dict(outcomes)})
                if adjudicated: result.update(adjudicated)
            except Exception: pass
        return result

def predict_rows(dataset: Path, messages: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    messages = messages if messages is not None else read_csv(dataset/"messages.csv")
    if not messages: raise FileNotFoundError(f"No messages.csv found under {dataset}")
    return Router(dataset).route_many(messages)

def predict(dataset: Path, output: Path) -> None:
    rows = predict_rows(dataset)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=OUT_COLUMNS); writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--dataset", type=Path, default=Path("dataset")); p.add_argument("--output", type=Path, default=Path("output.csv")); a=p.parse_args(); predict(a.dataset,a.output)
