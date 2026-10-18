"""
Medical Q&A Service Orchestrator.
Coordinates:
- Emergency condition detection and warnings
- Medical entity recognition (symptoms, diseases, medications, treatments)
- Local TF-IDF retrieval of evidence documents from MedQuAD
- Free retrieval-based answering (no API key required)
- Optional Gemini generative answering when API key is provided
- Medical disclaimers and source attribution
"""

from typing import Dict, List, Optional, Tuple
import config
from services.entity_service import entity_service
from services.retrieval_service import retrieval_service


class MedicalQAService:
    """
    Core medical Q&A engine providing truthful, evidence-grounded answers
    from the MedQuAD dataset.
    """

    def __init__(self):
        self.retrieval = retrieval_service
        self.entity_extractor = entity_service

    def answer_query(
        self,
        query: str,
        mode: str = "retrieval_free",
        top_k: int = config.DEFAULT_TOP_K
    ) -> Dict:
        """
        Process user medical question and return structured response.

        Args:
            query: User's health/medical question string.
            mode: "retrieval_free" (default, free & local) or "gemini" (optional).
            top_k: Number of reference evidence documents to retrieve.

        Returns:
            Dictionary with answer, recognized_entities, references, mode_label,
            is_emergency, emergency_alert, and disclaimer.
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "answer": "Please enter a valid medical question to search the MedQuAD knowledge base.",
                "mode_used": "retrieval_free",
                "mode_label": "Retrieval-Based Answering (Free Local Mode)",
                "recognized_entities": {"symptoms": [], "diseases": [], "medications": [], "treatments": []},
                "references": [],
                "is_emergency": False,
                "emergency_alert": None,
                "has_sufficient_evidence": False,
                "disclaimer": config.MEDICAL_DISCLAIMER
            }

        # 1. Emergency Detection
        is_emergency = self.entity_extractor.check_emergency(clean_query)
        emergency_alert = config.EMERGENCY_WARNING if is_emergency else None

        # 2. Clinical Entity Recognition
        entities = self.entity_extractor.extract_entities(clean_query)

        # 3. Document Retrieval
        retrieved_docs, has_evidence = self.retrieval.search(clean_query, top_k=top_k)

        # 4. Handle Insufficient Evidence
        if not has_evidence or not retrieved_docs:
            return {
                "answer": (
                    "No relevant medical evidence was found in the MedQuAD knowledge base for your inquiry. "
                    "The system does not invent or hallucinate medical information. "
                    "Please try rephrasing your question or consult a qualified healthcare professional."
                ),
                "mode_used": "retrieval_free",
                "mode_label": "Retrieval-Based Answering (Free Local Mode)",
                "recognized_entities": entities,
                "references": retrieved_docs,
                "is_emergency": is_emergency,
                "emergency_alert": emergency_alert,
                "has_sufficient_evidence": False,
                "disclaimer": config.MEDICAL_DISCLAIMER
            }

        # 5. Answer Generation based on Mode
        if mode == "gemini" and config.is_gemini_configured():
            answer_text, mode_label = self._generate_gemini_answer(clean_query, retrieved_docs)
            mode_used = "gemini"
        else:
            # Free API-key-free retrieval mode (Default)
            answer_text = self._synthesize_retrieval_answer(clean_query, retrieved_docs)
            mode_used = "retrieval_free"
            mode_label = "Retrieval-Based Answering (Free Local Mode — No API Key Required)"

        return {
            "answer": answer_text,
            "mode_used": mode_used,
            "mode_label": mode_label,
            "recognized_entities": entities,
            "references": retrieved_docs,
            "is_emergency": is_emergency,
            "emergency_alert": emergency_alert,
            "has_sufficient_evidence": True,
            "disclaimer": config.MEDICAL_DISCLAIMER
        }

    def _synthesize_retrieval_answer(self, query: str, docs: List[Dict]) -> str:
        """
        Synthesize answer directly from the top retrieved MedQuAD evidence document.
        Does not require external APIs or keys; 100% grounded in authentic NIH literature.
        """
        top_doc = docs[0]
        answer_text = top_doc.get("answer", "").strip()
        focus = top_doc.get("focus", "Medical Condition")
        source = top_doc.get("source", "NIH / MedQuAD")
        matched_question = top_doc.get("question", "")

        response = (
            f"**Information from {source} regarding {focus}:**\n\n"
            f"{answer_text}\n\n"
            f"*Matched Reference Question:* \"{matched_question}\""
        )
        return response

    def _generate_gemini_answer(self, query: str, docs: List[Dict]) -> Tuple[str, str]:
        """
        Generate grounded response using Google GenAI SDK when configured.
        Falls back gracefully to retrieval mode if the API call fails.
        """
        api_key = config.get_gemini_api_key()
        if not api_key:
            return self._synthesize_retrieval_answer(query, docs), "Retrieval-Based Answering (Free Local Mode)"

        # Prepare evidence context
        context_blocks = []
        for i, d in enumerate(docs[:3], 1):
            context_blocks.append(
                f"[Evidence #{i} from {d.get('source', 'NIH')}] Topic: {d.get('focus', '')}\n"
                f"Question: {d.get('question', '')}\n"
                f"Answer: {d.get('answer', '')}"
            )
        context_str = "\n\n".join(context_blocks)

        prompt = f"""You are a professional medical information assistant.
You are provided with verified medical evidence from the National Institutes of Health (NIH) MedQuAD dataset.

USER QUESTION: {query}

VERIFIED RETRIEVED EVIDENCE:
{context_str}

CRITICAL MEDICAL GUIDELINES:
1. Ground your answer strictly in the provided evidence. Do NOT invent diagnoses, unverified facts, or treatments.
2. NEVER prescribe medications or fabricate dosages.
3. Be clear, empathetic, and objective.
4. If the evidence does not directly answer parts of the question, acknowledge the limitation.
5. Provide a direct, synthesized summary of the verified evidence.
"""

        try:
            # Try new google.genai client if installed
            import google.genai as genai_new
            client = genai_new.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text, "Gemini Generative AI (Grounded in MedQuAD Evidence)"
        except Exception:
            pass

        try:
            # Fallback to legacy google.generativeai if available
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=api_key)
            model = genai_legacy.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text, "Gemini Generative AI (Grounded in MedQuAD Evidence)"
        except Exception as e:
            # Graceful fallback to retrieval-based mode if Gemini fails
            retrieval_fallback = self._synthesize_retrieval_answer(query, docs)
            return (
                f"{retrieval_fallback}\n\n*(Note: Gemini API unavailable or quota exceeded; automatically fell back to verified local retrieval)*",
                "Retrieval-Based Answering (Free Local Mode — Automatic Fallback)"
            )


# Global singleton instance
medical_service = MedicalQAService()
