"""Safety guardrail agent for clinical emergency detection and response validation."""
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re

from loguru import logger

from src.caution_keywords import CAUTION_KEYWORDS as SHARED_CAUTION_KEYWORDS
from src.caution_keywords import caution_keyword_hits
from src.llm import LLMProvider


class RiskLevel(Enum):
    """Risk level classification for queries."""
    NONE = "none"  # No clinical risk
    LOW = "low"  # Minor concern, proceed with caution
    MEDIUM = "medium"  # Moderate concern, add warnings
    HIGH = "high"  # Serious concern, recommend professional help
    CRITICAL = "critical"  # Emergency, immediate action needed


@dataclass
class SafetyAssessment:
    """Result of safety assessment."""
    risk_level: RiskLevel
    is_emergency: bool
    clinical_concerns: list[str]
    recommended_action: str
    warning_message: Optional[str] = None
    reasoning: str = ""


class SafetyGuard:
    """
    Safety guardrail agent for clinical emergency detection.

    This subagent provides an additional layer of safety beyond keyword matching
    by using LLM reasoning to detect:
    - Clinical emergencies that may not contain obvious keywords
    - Situations requiring immediate medical attention
    - Queries that could lead to harmful self-treatment
    - When to strongly recommend professional medical consultation
    """

    CAUTION_KEYWORDS = SHARED_CAUTION_KEYWORDS

    # Clinical emergency indicators (beyond simple keywords)
    EMERGENCY_PATTERNS = [
        # # Respiratory distress
        # r"(trouble|difficulty|can'?t|unable to)\s*(breath|breathing)",
        # r"(choking|gasping|wheezing)\s*(severely|badly)?",
        # r"blue\s*(lips|skin|face|fingernails)",
        # r"not\s*breathing",

        # # Cardiovascular
        # r"(severe|crushing|sudden)\s*chest\s*pain",
        # r"heart\s*(attack|stopped|racing uncontrollably)",
        # r"(fainting|passed out|unconscious)",

        # # Bleeding
        # r"(heavy|uncontrolled|won'?t stop)\s*bleeding",
        # r"blood\s*(everywhere|won'?t stop|pooling)",

        # # Neurological
        # r"(seizure|convulsion|fitting)",
        # r"(sudden|severe)\s*headache",
        # r"(stroke|paralysis|can'?t move)",
        # r"(confused|disoriented)\s*and\s*(weak|dizzy)",

        # # Allergic/Anaphylaxis
        # r"(severe|serious)\s*allergic\s*reaction",
        # r"throat\s*(swelling|closing)",
        # r"(can'?t swallow|swelling up)",

        # # Infection/Fever
        # r"(very high|dangerous)\s*fever",
        # r"(fever|temperature)\s*over\s*\d{2,3}",
        # r"(stiff neck|rash|spots)\s*with\s*fever",

        # # Post-procedure complications
        # r"(site|wound|incision)\s*(bleeding heavily|infected|pus)",
        # r"(sudden|severe)\s*pain\s*after\s*(procedure|surgery)",
        # r"(catheter|line|tube)\s*(fell out|came out|blocked)",
    ]

    SAFETY_CHECK_PROMPT = """You are a clinical safety assessment agent for a pediatric interventional radiology chatbot.
Your role is to identify queries that may indicate a medical emergency or require immediate professional attention.

Analyze this query for clinical safety concerns:

Query: {query}

Consider:
1. Is this describing an emergency situation (even if not using obvious keywords)?
2. Are there signs of serious complications (post-procedure bleeding, infection, etc.)?
3. Could following general advice be harmful without professional assessment?
4. Does this describe symptoms requiring urgent medical evaluation?

Respond in JSON only:
{{
    "risk_level": "none|low|medium|high|critical",
    "is_emergency": true/false,
    "clinical_concerns": ["list of specific concerns"],
    "recommended_action": "proceed|add_warning|recommend_professional|emergency_response",
    "reasoning": "brief explanation"
}}

IMPORTANT: Err on the side of caution. If unsure, escalate the risk level."""

    RESPONSE_CHECK_PROMPT = """Review this chatbot response for clinical safety.

User Query: {query}
Generated Response: {response}

Check for:
1. Any advice that could be harmful if followed without professional guidance
2. Missing safety warnings for procedures or symptoms mentioned
3. Information that could delay necessary emergency care
4. Appropriate disclaimers present

Respond in JSON only:
{{
    "is_safe": true/false,
    "concerns": ["list of safety concerns if any"],
    "suggested_additions": "any additional warnings to add",
    "should_block": true/false
}}"""

    # Standard warning messages
    WARNING_MESSAGES = {
        RiskLevel.LOW: "",
        RiskLevel.MEDIUM: (
            "\n\n⚠️ **Note**: Based on what you've described, it may be helpful "
            "to discuss this with your medical team for personalized guidance. "
            "If this is not an emergency, you can also contact the HKCH IR nurse "
            "coordinator at 3513 6099."
        ),
        RiskLevel.HIGH: (
            "\n\n⚠️ **Important**: The symptoms or situation you've described "
            "may require prompt medical attention. Please contact your doctor, "
            "the HKCH IR nurse contact at 3513 6099, or visit your nearest "
            "A&E if you are concerned."
        ),
        RiskLevel.CRITICAL: None,  # Use full emergency response instead
    }

    EMERGENCY_RESPONSE_EN = """🚨 **This sounds like it could be a medical emergency.**

**Please take immediate action:**
1. **Call 999** or go to your nearest Accident & Emergency department immediately
2. If your child has a PICC line, central line, or other IR device and it's causing the problem, bring any relevant information with you
3. Do NOT wait for a callback or appointment

**While waiting for help:**
- Keep your child calm and still
- Do not remove any medical devices unless instructed by emergency services
- Note the time symptoms started

📞 **HKCH IR Emergency Contact**: [Contact IR Nurse Coordinator]

Please remember: This chatbot cannot assess emergencies. Always err on the side of caution with children."""

    EMERGENCY_RESPONSE_ZH = """🚨 **這聽起來可能是醫療緊急情況。**

**請立即採取行動：**
1. **致電999** 或立即前往最近的急症室
2. 如果您的孩子有PICC導管、中心靜脈導管或其他介入放射科裝置，請帶上相關資料
3. 請勿等待回電或預約

**等待幫助時：**
- 保持孩子冷靜和靜止
- 除非急救人員指示，否則不要移除任何醫療器材
- 記錄症狀開始的時間

📞 **HKCH介入放射科緊急聯絡**：5741 3238 / 35136099

請記住：此聊天機器人無法評估緊急情況。對於兒童，請務必謹慎行事。"""

    def __init__(self, llm_provider: LLMProvider, use_llm_check: bool = True):
        """
        Initialize the safety guard.

        Args:
            llm_provider: LLM for safety assessment
            use_llm_check: Whether to use LLM for safety checks (vs pattern-only)
        """
        self.llm = llm_provider
        self.use_llm_check = use_llm_check
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.EMERGENCY_PATTERNS]
        logger.info(f"Initialized SafetyGuard (LLM check: {use_llm_check})")

    def _check_patterns(self, text: str) -> list[str]:
        """Check text against emergency patterns."""
        matches = []
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)
        return matches

    def _check_caution_keywords(self, text: str) -> list[str]:
        """Return caution keywords that match post-procedure infection concerns."""
        return caution_keyword_hits(text)

    def _is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any(ord(char) > 127 for char in text)

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def assess_query(self, query: str) -> SafetyAssessment:
        """
        Assess a query for clinical safety concerns.

        Args:
            query: User query to assess

        Returns:
            SafetyAssessment with risk level and recommendations
        """
        # Fast path: Pattern matching
        pattern_matches = self._check_patterns(query)

        if pattern_matches:
            logger.warning(f"Emergency patterns detected: {pattern_matches}")
            return SafetyAssessment(
                risk_level=RiskLevel.CRITICAL,
                is_emergency=True,
                clinical_concerns=pattern_matches,
                recommended_action="emergency_response",
                reasoning="Emergency patterns detected in query"
            )

        caution_hits = self._check_caution_keywords(query)
        if caution_hits:
            logger.info(f"Caution keywords detected: {caution_hits}")
            return SafetyAssessment(
                risk_level=RiskLevel.HIGH,
                is_emergency=False,
                clinical_concerns=caution_hits,
                recommended_action="add_warning",
                warning_message=self.WARNING_MESSAGES[RiskLevel.HIGH],
                reasoning="Post-procedure infection/wound caution keywords detected",
            )

        # If not using LLM or for simple cases, return no risk
        if not self.use_llm_check:
            return SafetyAssessment(
                risk_level=RiskLevel.NONE,
                is_emergency=False,
                clinical_concerns=[],
                recommended_action="proceed",
                reasoning="No emergency patterns detected (LLM check disabled)"
            )

        # LLM-based assessment for nuanced detection
        try:
            prompt = self.SAFETY_CHECK_PROMPT.format(query=query)
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.generate(messages, temperature=0.1)

            parsed = self._parse_json_response(response)
            if parsed:
                risk_level = RiskLevel(parsed.get("risk_level", "none"))
                is_emergency = parsed.get("is_emergency", False)

                # Determine warning message
                warning = self.WARNING_MESSAGES.get(risk_level)

                return SafetyAssessment(
                    risk_level=risk_level,
                    is_emergency=is_emergency,
                    clinical_concerns=parsed.get("clinical_concerns", []),
                    recommended_action=parsed.get("recommended_action", "proceed"),
                    warning_message=warning,
                    reasoning=parsed.get("reasoning", "")
                )

        except Exception as e:
            logger.warning(f"LLM safety check failed: {e}")

        # Default to safe if LLM fails
        return SafetyAssessment(
            risk_level=RiskLevel.NONE,
            is_emergency=False,
            clinical_concerns=[],
            recommended_action="proceed",
            reasoning="LLM check failed, defaulting to safe"
        )

    def get_emergency_response(self, query: str) -> str:
        """Get appropriate emergency response based on query language."""
        if self._is_chinese(query):
            return self.EMERGENCY_RESPONSE_ZH
        return self.EMERGENCY_RESPONSE_EN

    def validate_response(self, query: str, response: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a generated response for safety.

        Args:
            query: Original user query
            response: Generated response to validate

        Returns:
            Tuple of (is_safe, additional_warning_if_any)
        """
        if not self.use_llm_check:
            return True, None

        try:
            prompt = self.RESPONSE_CHECK_PROMPT.format(query=query, response=response)
            messages = [{"role": "user", "content": prompt}]
            llm_response = self.llm.generate(messages, temperature=0.1)

            parsed = self._parse_json_response(llm_response)
            if parsed:
                is_safe = parsed.get("is_safe", True)
                additions = parsed.get("suggested_additions", "")

                if not is_safe and parsed.get("should_block", False):
                    logger.warning(f"Response blocked by safety check: {parsed.get('concerns')}")
                    return False, None

                # Handle if additions is a list (LLM sometimes returns list)
                if isinstance(additions, list):
                    if additions:
                        # Join list items into formatted string
                        additions = "\n".join(f"• {item}" for item in additions if item)
                    else:
                        additions = None
                elif isinstance(additions, str):
                    additions = additions.strip() if additions.strip() else None
                else:
                    additions = None

                return is_safe, additions

        except Exception as e:
            logger.warning(f"Response validation failed: {e}")

        return True, None

    def add_safety_wrapper(self, response: str, assessment: SafetyAssessment) -> str:
        """
        Add appropriate safety warnings to a response based on assessment.

        Args:
            response: Original response
            assessment: Safety assessment result

        Returns:
            Response with added warnings if needed
        """
        if assessment.warning_message:
            return response + assessment.warning_message
        return response
