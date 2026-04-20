import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


BLOCKLIST = {
    "direct_threat": [
        re.compile(r"\bi\s*(?:will|'ll|am going to|gonna)\s*(kill|murder|shoot|stab|hurt)\s+you\b", re.IGNORECASE),
        re.compile(r"\byou(?:'re| are)\s+going\s+to\s+die\b", re.IGNORECASE),
        re.compile(r"\bsomeone\s+should\s+(?:kill|shoot|stab)\s+you\b", re.IGNORECASE),
        re.compile(r"\bi\s*(?:will|'ll|gonna)\s+find\s+(?:where\s+)?you\s+live\b", re.IGNORECASE),
        re.compile(r"\bi\s*(?:want|hope)\s+you\s+dead\b", re.IGNORECASE),
    ],
    "self_harm_directed": [
        re.compile(r"\b(?:go\s+)?kill\s+yourself\b", re.IGNORECASE),
        re.compile(r"\byou\s+should\s+(?:just\s+)?(?:die|kill\s+yourself)\b", re.IGNORECASE),
        re.compile(r"\bnobody\s+would\s+miss\s+you\s+if\s+you\s+died\b", re.IGNORECASE),
        re.compile(r"\bdo\s+everyone\s+a\s+favo[u]?r\s+and\s+disappear\b", re.IGNORECASE),
    ],
    "doxxing_stalking": [
        re.compile(r"\bi\s*(?:know|found)\s+where\s+you\s+live\b", re.IGNORECASE),
        re.compile(r"\bi\s*(?:will|'ll|gonna)\s+post\s+your\s+(?:address|phone|location)\b", re.IGNORECASE),
        re.compile(r"\bi\s*(?:found|know)\s+your\s+real\s+name\b", re.IGNORECASE),
        re.compile(r"\beveryone\s+will\s+know\s+who\s+you\s+really\s+are\b", re.IGNORECASE),
    ],
    "dehumanization": [
        re.compile(r"\b(?:they|those\s+people|[a-z]+)\s+are\s+not\s+(?:human|people|person)\b", re.IGNORECASE),
        re.compile(r"\b(?:they|those\s+people|[a-z]+)\s+are\s+animals\b", re.IGNORECASE),
        re.compile(r"\b(?:they|those\s+people|[a-z]+)\s+should\s+be\s+exterminated\b", re.IGNORECASE),
        re.compile(r"\b(?:they|those\s+people|[a-z]+)\s+are\s+a\s+disease\b", re.IGNORECASE),
    ],
    "coordinated_harassment": [
        re.compile(r"\beveryone\s+report\s+@\w+\b", re.IGNORECASE),
        re.compile(r"\blet'?s\s+all\s+go\s+after\b", re.IGNORECASE),
        re.compile(r"\bmass\s+report\s+(?:this\s+)?account\b", re.IGNORECASE),
        re.compile(r"\braid\b(?=.*\bprofile\b)", re.IGNORECASE),
    ],
}


def input_filter(text: str) -> Optional[Dict[str, Any]]:
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {
                    "decision": "block",
                    "layer": "input_filter",
                    "category": category,
                    "confidence": 1.0,
                }
    return None


@dataclass
class ModerationPipeline:
    model: Any
    tokenizer: Any
    calibrator: Any
    low_threshold: float = 0.4
    high_threshold: float = 0.6
    device: str = "cpu"

    def _raw_score(self, text: str) -> float:
        import torch

        encoded = self.tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]
        return float(probs[0])

    def _calibrated_score(self, raw_prob: float) -> float:
        x = np.array([[raw_prob]])
        calibrated = self.calibrator.predict_proba(x)[:, 1]
        return float(calibrated[0])

    def predict(self, text: str) -> Dict[str, Any]:
        blocked = input_filter(text)
        if blocked is not None:
            return blocked

        raw_prob = self._raw_score(text)
        conf = self._calibrated_score(raw_prob)

        if conf >= self.high_threshold:
            return {"decision": "block", "layer": "model", "confidence": conf}
        if conf <= self.low_threshold:
            return {"decision": "allow", "layer": "model", "confidence": conf}
        return {"decision": "review", "layer": "model", "confidence": conf}


def count_filter_hits(texts: List[str]) -> Dict[str, int]:
    counts = {k: 0 for k in BLOCKLIST}
    for t in texts:
        decision = input_filter(t)
        if decision is not None:
            counts[decision["category"]] += 1
    return counts
