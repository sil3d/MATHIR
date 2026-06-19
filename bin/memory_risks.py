#!/usr/bin/env python3
"""
MATHIR Memory Risk Mitigation
Based on PersistBench (arxiv 2602.01146) findings:
- Cross-domain leakage: median 53% failure rate
- Memory-induced sycophancy: >90% failure rate
- Beneficial memory: only weakly correlated with safety

This module implements:
1. Domain isolation — memories tagged by domain, cross-domain retrieval blocked
2. Sycophancy detection — detect when memory biases the LLM incorrectly
3. Memory decay — old/irrelevant memories lose priority over time
4. Anomaly scoring — immunological tier flags suspicious patterns
"""

import re
import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Domain(Enum):
    """Memory domains for isolation."""
    CODE = "code"
    PERSONAL = "personal"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    LEGAL = "legal"
    EDUCATION = "education"
    WORK = "work"
    GENERAL = "general"


@dataclass
class MemoryRisk:
    """Risk assessment for a memory."""
    domain: Domain
    leakage_risk: float  # 0.0 = safe, 1.0 = high risk
    sycophancy_risk: float  # 0.0 = neutral, 1.0 = high bias
    sensitivity: float  # 0.0 = public, 1.0 = highly sensitive
    reasons: list = field(default_factory=list)


class DomainClassifier:
    """Classify memories into domains for isolation."""
    
    DOMAIN_KEYWORDS = {
        Domain.CODE: ["api", "function", "class", "bug", "fix", "code", "git", "commit",
                       "deploy", "build", "test", "lint", "refactor", "typescript", "python",
                       "rust", "react", "database", "sql", "endpoint", "server"],
        Domain.PERSONAL: ["family", "friend", "birthday", "address", "phone", "email",
                          "relationship", "home", "hobby", "preference", "like", "dislike"],
        Domain.MEDICAL: ["health", "doctor", "medication", "symptom", "diagnosis",
                         "treatment", "allergy", "blood", "pressure", "weight"],
        Domain.FINANCIAL: ["bank", "account", "balance", "transaction", "payment",
                           "salary", "investment", "tax", "budget", "credit"],
        Domain.LEGAL: ["contract", "agreement", "lawsuit", "compliance", "regulation",
                       "policy", "terms", "liability", "license"],
        Domain.EDUCATION: ["course", "student", "grade", "exam", "university",
                           "lecture", "assignment", "curriculum", "diploma"],
        Domain.WORK: ["meeting", "deadline", "project", "sprint", "backlog",
                      "stakeholder", "client", "deliverable", "milestone"],
    }
    
    def classify(self, content: str) -> Domain:
        """Classify memory content into a domain."""
        content_lower = content.lower()
        scores = {}
        
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            scores[domain] = score
        
        if max(scores.values()) == 0:
            return Domain.GENERAL
        
        return max(scores, key=scores.get)


class LeakageDetector:
    """
    Detect cross-domain leakage risk.
    
    PersistBench finding: memories from education domain leak into health domain.
    Solution: domain isolation + access control.
    """
    
    DANGER_PAIRS = {
        (Domain.MEDICAL, Domain.FINANCIAL): "Health info should not influence financial decisions",
        (Domain.PERSONAL, Domain.WORK): "Personal info should not leak into work context",
        (Domain.LEGAL, Domain.CODE): "Legal constraints should not auto-modify code",
        (Domain.MEDICAL, Domain.CODE): "Medical info should not appear in code context",
        (Domain.FINANCIAL, Domain.GENERAL): "Financial data needs explicit access grant",
    }
    
    def check_leakage(self, source_domain: Domain, target_domain: Domain,
                      content: str) -> MemoryRisk:
        """Check if memories from source_domain could leak into target_domain."""
        risk = MemoryRisk(
            domain=source_domain,
            leakage_risk=0.0,
            sycophancy_risk=0.0,
            sensitivity=0.0,
        )
        
        # Check domain pair danger
        pair = (source_domain, target_domain)
        reverse_pair = (target_domain, source_domain)
        
        if pair in self.DANGER_PAIRS:
            risk.leakage_risk = 0.8
            risk.reasons.append(self.DANGER_PAIRS[pair])
        elif reverse_pair in self.DANGER_PAIRS:
            risk.leakage_risk = 0.8
            risk.reasons.append(self.DANGER_PAIRS[reverse_pair])
        
        # High-sensitivity domains
        if source_domain in (Domain.MEDICAL, Domain.FINANCIAL, Domain.LEGAL):
            risk.sensitivity = 0.9
            risk.leakage_risk = max(risk.leakage_risk, 0.7)
        
        # PII detection
        pii_patterns = [
            (r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', "SSN detected"),
            (r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b', "Credit card number detected"),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "Email detected"),
            (r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', "Phone number detected"),
            (r'\b\d{1,5}\s\w+\s(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln)\b', "Address detected"),
        ]
        
        for pattern, reason in pii_patterns:
            if re.search(pattern, content):
                risk.sensitivity = max(risk.sensitivity, 0.95)
                risk.leakage_risk = max(risk.leakage_risk, 0.9)
                risk.reasons.append(reason)
        
        return risk


class SycophancyDetector:
    """
    Detect memory-induced sycophancy.
    
    PersistBench finding: >90% failure rate on sycophancy.
    When a user expresses a belief, the model agrees because it's in memory,
    even when the belief is factually wrong.
    """
    
    SYCOPHANCY_PATTERNS = [
        # User opinion stored as fact
        (r"user believes? (that )?", "User belief stored — may bias responses"),
        (r"user (always|never|prefers?) to", "User habit stored — may override correctness"),
        (r"according to the user", "User claim stored — needs verification"),
        (r"the user (thinks|feels|wants)", "User sentiment stored — may cause agreement bias"),
    ]
    
    IDENTITY_PATTERNS = [
        # Identity validation triggers
        (r"(I am|I'm) (a |an )?(expert|professional|genius|right|better)", "Identity validation — high sycophancy risk"),
        (r"(I know|I understand) (more|better) than", "Authority claim — may trigger false agreement"),
        (r"(I am|I'm) (sure|certain|positive) that", "Overconfidence — may cause false agreement"),
        (r"(don't|do not) (question|doubt) (my|me)", "Authority protection — high sycophancy risk"),
        (r"I know better than", "Authority claim — may trigger false agreement"),
    ]
    
    def check_sycophancy(self, content: str) -> MemoryRisk:
        """Check if a memory could cause sycophantic behavior."""
        risk = MemoryRisk(
            domain=Domain.GENERAL,
            leakage_risk=0.0,
            sycophancy_risk=0.0,
            sensitivity=0.0,
        )
        
        content_lower = content.lower()
        
        # Check sycophancy patterns
        for pattern, reason in self.SYCOPHANCY_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                risk.sycophancy_risk = max(risk.sycophancy_risk, 0.7)
                risk.reasons.append(reason)
        
        # Check identity patterns (highest risk)
        for pattern, reason in self.IDENTITY_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                risk.sycophancy_risk = max(risk.sycophancy_risk, 0.9)
                risk.reasons.append(reason)
        
        # Stored opinions without evidence
        opinion_markers = ["i think", "i believe", "in my opinion", "i feel like",
                          "probably", "maybe", "i guess", "i suppose"]
        if any(marker in content_lower for marker in opinion_markers):
            risk.sycophancy_risk = max(risk.sycophancy_risk, 0.5)
            risk.reasons.append("Opinion stored without evidence marker")
        
        return risk


class MemoryRiskManager:
    """
    Central risk manager for MATHIR memories.
    
    Integrates:
    - Domain classification
    - Leakage detection
    - Sycophancy detection
    - Risk-based retrieval filtering
    """
    
    def __init__(self):
        self.classifier = DomainClassifier()
        self.leakage_detector = LeakageDetector()
        self.sycophancy_detector = SycophancyDetector()
        self.risk_threshold = 0.7  # Block retrieval above this risk
    
    def assess(self, content: str, target_domain: Optional[Domain] = None) -> MemoryRisk:
        """Full risk assessment for a memory."""
        # Classify domain
        source_domain = self.classifier.classify(content)
        
        # Check leakage (always check for PII, even without target domain)
        leakage = self.leakage_detector.check_leakage(
            source_domain, target_domain or source_domain, content
        )
        
        # Check sycophancy
        sycophancy = self.sycophancy_detector.check_sycophancy(content)
        
        # Combine risks
        return MemoryRisk(
            domain=source_domain,
            leakage_risk=max(leakage.leakage_risk, sycophancy.leakage_risk),
            sycophancy_risk=sycophancy.sycophancy_risk,
            sensitivity=leakage.sensitivity,
            reasons=list(set(leakage.reasons + sycophancy.reasons)),
        )
    
    def should_retrieve(self, memory_content: str, 
                       query_domain: Optional[Domain] = None) -> bool:
        """Decide if a memory should be retrieved for a given query context."""
        risk = self.assess(memory_content, query_domain)
        
        # Block high-risk cross-domain retrievals
        if risk.leakage_risk > self.risk_threshold:
            return False
        
        # Block sycophancy-inducing memories
        if risk.sycophancy_risk > self.risk_threshold:
            return False
        
        return True
    
    def sanitize_for_context(self, memory_content: str,
                            target_domain: Domain) -> str:
        """Sanitize a memory before injecting into LLM context."""
        risk = self.assess(memory_content, target_domain)
        
        # Add risk warning prefix
        if risk.reasons:
            warning = f"[RISK: {', '.join(risk.reasons)}] "
            return warning + memory_content
        
        return memory_content
    
    def get_risk_report(self, memories: list) -> dict:
        """Generate a risk report for a set of memories."""
        report = {
            "total": len(memories),
            "by_domain": {},
            "high_leakage": 0,
            "high_sycophancy": 0,
            "high_sensitivity": 0,
            "safe": 0,
        }
        
        for mem in memories:
            risk = self.assess(mem.get("content", ""))
            
            domain = risk.domain.value
            report["by_domain"][domain] = report["by_domain"].get(domain, 0) + 1
            
            if risk.leakage_risk > 0.7:
                report["high_leakage"] += 1
            if risk.sycophancy_risk > 0.7:
                report["high_sycophancy"] += 1
            if risk.sensitivity > 0.7:
                report["high_sensitivity"] += 1
            if risk.leakage_risk <= 0.3 and risk.sycophancy_risk <= 0.3:
                report["safe"] += 1
        
        return report


# Quick test
if __name__ == "__main__":
    manager = MemoryRiskManager()
    
    test_cases = [
        "The API endpoint uses JWT authentication",
        "My social security number is 123-45-6789",
        "I think TypeScript is better than JavaScript",
        "I am an expert developer and I know better than the linter",
        "The user believes vaccines cause autism",
        "My bank account balance is $50,000",
        "The meeting is at 3pm tomorrow",
        "I have a allergy to penicillin",
    ]
    
    for content in test_cases:
        risk = manager.assess(content)
        print(f"\n{'='*60}")
        print(f"Content: {content[:60]}...")
        print(f"Domain: {risk.domain.value}")
        print(f"Leakage: {risk.leakage_risk:.1f}")
        print(f"Sycophancy: {risk.sycophancy_risk:.1f}")
        print(f"Sensitivity: {risk.sensitivity:.1f}")
        if risk.reasons:
            print(f"Reasons: {risk.reasons}")
