
import json
import re
import time
import logging
import asyncio
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod

from loguru import logger
from google import genai
from google.genai import types

from app.core.config import settings
from app.services.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT,
    VISION_EXTRACTION_PROMPT,
    VALIDATION_PROMPT
)
from app.models.schemas import MarksheetExtraction
from app.utils.exceptions import LLMExtractionError, LLMConnectionError


class BaseLLMExtractor(ABC):
    """Abstract base class for LLM extractors"""
    
    @abstractmethod
    async def extract(
        self, 
        images: List[Tuple[bytes, str]]
    ) -> Dict[str, Any]:
        """Extract data from images using LLM"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if LLM connection is healthy"""
        pass
    
    def _clean_json_response(self, response: str) -> str:
        response = response.strip()
        
        # Handle ```json ... ``` format
        if response.startswith("```"):
            # Find the first newline after ```
            start = response.find("\n")
            if start != -1:
                # Find the closing ```
                end = response.rfind("```")
                if end > start:
                    response = response[start:end].strip()
        
        # Handle cases where response might have extra text before/after JSON
        # Find first { and last }
        json_start = response.find("{")
        json_end = response.rfind("}")
        
        if json_start != -1 and json_end != -1:
            response = response[json_start:json_end + 1]
        
        return response
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response with error handling"""
        cleaned = self._clean_json_response(response)
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Raw response: {response[:500]}...")
            raise LLMExtractionError(
                f"Failed to parse LLM response as JSON: {str(e)}",
                details={"raw_response": response[:1000]}
            )
    
    def _calculate_overall_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate weighted average confidence score"""
        confidences = []
        weights = []
        
        def extract_confidences(obj, weight=1.0):
            if isinstance(obj, dict):
                if "confidence" in obj and "value" in obj:
                    if obj.get("value") is not None:
                        confidences.append(obj["confidence"])
                        weights.append(weight)
                else:
                    for key, value in obj.items():
                        # Higher weight for important fields
                        field_weight = weight
                        if key in ["name", "roll_number", "result_status"]:
                            field_weight = 2.0
                        elif key in ["obtained_marks", "total_marks"]:
                            field_weight = 1.5
                        extract_confidences(value, field_weight)
            elif isinstance(obj, list):
                for item in obj:
                    extract_confidences(item, weight)
        
        extract_confidences(data)
        
        if not confidences:
            return 0.5
        
        weighted_sum = sum(c * w for c, w in zip(confidences, weights))
        total_weight = sum(weights)
        
        return round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.5



logger = logging.getLogger(__name__)

class GeminiExtractor:
 
    def __init__(self, api_key: Optional[str] = None):

        self.api_key = api_key or settings.google_api_key
        
        if not self.api_key:
            raise Exception("Google API key not configured. Provide via request or set GOOGLE_API_KEY in environment.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = settings.gemini_model
        
        key_source = "user-provided" if api_key else "system"
        logger.info(f"Initialized Gemini extractor with model: {self.model_name} (using {key_source} API key)")

    async def health_check(self) -> bool:
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents="Say OK",
            )
            return response.text and "ok" in response.text.lower()
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return False

    async def extract(self, images: list[tuple[bytes, str]]):
        """
        Extract marksheet data from images using Gemini Vision API.
        
        This is the PRIMARY extraction method - highest accuracy.
        
        Args:
            images: List of (image_bytes, mime_type) tuples
            
        Returns:
            dict: Extracted marksheet data with confidence scores
            
        Performance:
            - Single image: ~15-25 seconds
            - Multiple images: ~20-35 seconds
        """
        try:
            # Convert images to Gemini format
            image_parts = [
                types.Part.from_bytes(data=img_bytes, mime_type=mime)
                for img_bytes, mime in images
            ]

            # Call Gemini API (MAIN BOTTLENECK - 15-30 seconds)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=VISION_EXTRACTION_PROMPT),
                            *image_parts,
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    temperature=0.1,  # Low = deterministic, high accuracy
                    max_output_tokens=10000,
                    response_mime_type="application/json",  # Forces valid JSON
                ),
            )

            if not response.text:
                raise Exception("Empty response from Gemini")
            
            # Parse JSON response (fast - <100ms)
            data = self._parse_json_response(response.text)
            
            # Calculate weighted confidence score
            data["extraction_confidence"] = self._calculate_confidence(data)
            return data

        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            raise Exception(f"Extraction Error: {str(e)}")

    def _calculate_confidence(self, data: dict) -> float:
        """
        Calculate weighted average confidence score.
        
        Weight priorities:
        - Critical fields (name, roll_number, result_status): 2.0x
        - Important fields (marks, subject_name): 1.5x
        - Other fields: 1.0x
        
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        confidences = []
        weights = []
        
        def extract_confidences(obj, weight=1.0):
            if isinstance(obj, dict):
                if "confidence" in obj and "value" in obj:
                    if obj.get("value") is not None:
                        confidences.append(obj["confidence"])
                        weights.append(weight)
                else:
                    for key, value in obj.items():
                        # Assign weights based on field importance
                        field_weight = weight
                        if key in ["name", "roll_number", "result_status"]:
                            field_weight = 2.0  # Critical fields
                        elif key in ["obtained_marks", "total_marks", "subject_name"]:
                            field_weight = 1.5  # Important fields
                        extract_confidences(value, field_weight)
            elif isinstance(obj, list):
                for item in obj:
                    extract_confidences(item, weight)
        
        extract_confidences(data)
        
        if not confidences:
            return 0.5  # Default confidence
        
        # Weighted average calculation
        weighted_sum = sum(c * w for c, w in zip(confidences, weights))
        total_weight = sum(weights)
        
        return round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.5

    async def extract_from_text(self, ocr_text: str):
     
        try:
            from app.services.prompts import get_extraction_prompt
            
            # Build prompt with OCR text embedded
            prompt = get_extraction_prompt(ocr_text)

            # Call Gemini API (text-only, slightly faster)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=prompt),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Low = deterministic output
                    max_output_tokens=10000,
                    response_mime_type="application/json",
                ),
            )

            if not response.text:
                raise Exception("Empty response from Gemini")
            
            # Parse and calculate confidence
            data = self._parse_json_response(response.text)
            data["extraction_confidence"] = self._calculate_confidence(data)
            return data

        except Exception as e:
            logger.error(f"Gemini text extraction failed: {e}")
            raise Exception(f"Text Extraction Error: {str(e)}")

    def _parse_json_response(self, text: str) -> dict:

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        clean = re.sub(r"```json\s*", "", text)
        clean = re.sub(r"```\s*", "", clean).strip()
        
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response")
        
        json_text = match.group(0)
        
        json_text = re.sub(r",\s*([}\]])", r"\1", json_text)
        json_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', json_text)
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            logger.warning("JSON malformed, using manual extraction")
            return self._extract_json_manually(json_text)
    
    def _extract_json_manually(self, json_text: str) -> dict:
    
        result = {
            "candidate": {},
            "examination": {},
            "subjects": [],
            "result": {},
            "metadata": {},
            "processing_notes": ["JSON was malformed, extracted manually with fallback parser"]
        }
        
        # Helper: Extract string field with confidence
        def extract_field(pattern, text, default_conf=0.7):
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1)
                # Try to get confidence if present
                conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', match.group(0) if hasattr(match, 'group') else "")
                conf = float(conf_match.group(1)) if conf_match else default_conf
                return {"value": value, "confidence": conf}
            return None
        
        def extract_numeric_field(pattern, text, default_conf=0.7):
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    return {"value": value, "confidence": default_conf}
                except:
                    pass
            return None
        
        # ============ CANDIDATE FIELDS ============
        # Name
        name = extract_field(r'"name"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if name:
            result["candidate"]["name"] = name
        
        # Father name
        father = extract_field(r'"father_name"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if father:
            result["candidate"]["father_name"] = father
        
        # Mother name
        mother = extract_field(r'"mother_name"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if mother:
            result["candidate"]["mother_name"] = mother
        
        # Roll number
        roll = extract_field(r'"roll_number"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if roll:
            result["candidate"]["roll_number"] = roll
        
        # Registration number
        reg = extract_field(r'"registration_number"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if reg:
            result["candidate"]["registration_number"] = reg
        
        # Date of birth
        dob = extract_field(r'"date_of_birth"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if dob:
            result["candidate"]["date_of_birth"] = dob
        
        # ============ EXAMINATION FIELDS ============
        # Exam name
        exam_name = extract_field(r'"exam_name"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if exam_name:
            result["examination"]["exam_name"] = exam_name
        
        # Exam year
        exam_year = extract_field(r'"exam_year"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if exam_year:
            result["examination"]["exam_year"] = exam_year
        
        # Board/University
        board = extract_field(r'"board_university"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if board:
            result["examination"]["board_university"] = board
        
        # Institution name
        inst = extract_field(r'"institution_name"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if inst:
            result["examination"]["institution_name"] = inst
        
        # Course name
        course = extract_field(r'"course_name"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if course:
            result["examination"]["course_name"] = course
        
        # ============ SUBJECTS - Most Important ============
        # Find each subject block and extract all fields
        # Pattern to find subject blocks - more flexible pattern
        subject_blocks = re.finditer(
            r'\{\s*"subject_code".*?"is_pass"[^}]*\}[^}]*\}',
            json_text,
            re.DOTALL
        )
        
        subject_blocks_list = list(subject_blocks)
        
        # If the first pattern didn't find subjects, try an alternative
        if not subject_blocks_list:
            subject_blocks_list = list(re.finditer(
                r'\{\s*"subject_name"\s*:\s*\{\s*"value"[^{]*?(?:"is_pass"|"components")',
                json_text,
                re.DOTALL
            ))
        
        for block_match in subject_blocks_list:
            block = block_match.group(0)
            # Extend block to capture complete subject
            block_start = block_match.start()
            block_end = block_match.end()
            # Find the complete block by counting braces
            depth = 0
            extended_end = block_start
            for i, char in enumerate(json_text[block_start:]):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        extended_end = block_start + i + 1
                        break
            block = json_text[block_start:extended_end]
            
            subj = {}
            
            # Subject name
            subj_name = extract_field(r'"subject_name"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', block)
            if subj_name:
                subj["subject_name"] = subj_name
            
            # Subject code
            subj_code = extract_field(r'"subject_code"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', block)
            if subj_code:
                subj["subject_code"] = subj_code
            
            # Subject group
            subj_group = extract_field(r'"subject_group"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', block)
            if subj_group:
                subj["subject_group"] = subj_group
            
            # Max marks
            max_marks = extract_numeric_field(r'"max_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', block)
            if max_marks:
                subj["max_marks"] = max_marks
            
            # Obtained marks
            obt_marks = extract_numeric_field(r'"obtained_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', block)
            if obt_marks:
                subj["obtained_marks"] = obt_marks
            
            # Grade
            grade = extract_field(r'"grade"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', block)
            if grade:
                subj["grade"] = grade
            
            # Theory marks
            theory = extract_numeric_field(r'"theory_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', block)
            if theory:
                subj["theory_marks"] = theory
            
            # Practical marks
            practical = extract_numeric_field(r'"practical_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', block)
            if practical:
                subj["practical_marks"] = practical
            
            # Oral marks
            oral = extract_numeric_field(r'"oral_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', block)
            if oral:
                subj["oral_marks"] = oral
            
            # Written marks
            written = extract_numeric_field(r'"written_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', block)
            if written:
                subj["written_marks"] = written
            
            # Credits
            credits = extract_numeric_field(r'"credits"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', block)
            if credits:
                subj["credits"] = credits
            
            # Grade point
            gp = extract_numeric_field(r'"grade_point"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', block)
            if gp:
                subj["grade_point"] = gp
            
            # is_pass
            is_pass = re.search(r'"is_pass"\s*:\s*\{\s*"value"\s*:\s*(true|false)', block, re.IGNORECASE)
            if is_pass:
                subj["is_pass"] = {"value": is_pass.group(1).lower() == "true", "confidence": 0.7}
            
            # ============ COMPONENTS (DETAILED EVALUATION) ============
            # Extract components array for detailed breakdowns (papers, oral, written, etc.)
            components = []
            components_match = re.search(r'"components"\s*:\s*\[(.*?)\]', block, re.DOTALL)
            if components_match:
                components_text = components_match.group(1)
                # Find each component by looking for component_name patterns
                # More robust pattern that handles multi-line
                comp_pattern = r'"component_name"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"[^}]*\}[^}]*"max_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)[^}]*\}[^}]*"obtained_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)'
                for comp_match in re.finditer(comp_pattern, components_text, re.DOTALL):
                    comp = {
                        "component_name": {"value": comp_match.group(1), "confidence": 0.7},
                        "max_marks": {"value": float(comp_match.group(2)), "confidence": 0.7},
                        "obtained_marks": {"value": float(comp_match.group(3)), "confidence": 0.7}
                    }
                    components.append(comp)
                
                # If first pattern didn't work, try alternate order (obtained before max)
                if not components:
                    comp_pattern2 = r'"component_name"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"[^}]*\}[^}]*"obtained_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)[^}]*\}[^}]*"max_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)'
                    for comp_match in re.finditer(comp_pattern2, components_text, re.DOTALL):
                        comp = {
                            "component_name": {"value": comp_match.group(1), "confidence": 0.7},
                            "obtained_marks": {"value": float(comp_match.group(2)), "confidence": 0.7},
                            "max_marks": {"value": float(comp_match.group(3)), "confidence": 0.7}
                        }
                        components.append(comp)
                
                # Fallback: extract just component names
                if not components:
                    comp_names = re.finditer(r'"component_name"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', components_text)
                    for comp_match in comp_names:
                        comp = {"component_name": {"value": comp_match.group(1), "confidence": 0.7}}
                        # Try to find marks nearby
                        start = comp_match.start()
                        end = min(start + 300, len(components_text))
                        nearby = components_text[start:end]
                        
                        max_m = re.search(r'"max_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', nearby)
                        if max_m:
                            comp["max_marks"] = {"value": float(max_m.group(1)), "confidence": 0.7}
                        
                        obt_m = re.search(r'"obtained_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', nearby)
                        if obt_m:
                            comp["obtained_marks"] = {"value": float(obt_m.group(1)), "confidence": 0.7}
                        
                        components.append(comp)
            
            if components:
                subj["components"] = components
            
            if subj.get("subject_name"):
                result["subjects"].append(subj)
        
        # If no subjects found with block pattern, try simpler pattern
        if not result["subjects"]:
            # Simpler pattern for subject extraction
            simple_pattern = re.finditer(
                r'"subject_name"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"[^}]*"confidence"\s*:\s*([\d.]+)[^}]*\}',
                json_text
            )
            for match in simple_pattern:
                subj = {"subject_name": {"value": match.group(1), "confidence": float(match.group(2))}}
                # Try to find marks nearby
                start_pos = match.start()
                nearby_text = json_text[start_pos:start_pos+500]
                
                obt = extract_numeric_field(r'"obtained_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', nearby_text)
                if obt:
                    subj["obtained_marks"] = obt
                
                max_m = extract_numeric_field(r'"max_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', nearby_text)
                if max_m:
                    subj["max_marks"] = max_m
                
                grade = extract_field(r'"grade"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', nearby_text)
                if grade:
                    subj["grade"] = grade
                
                result["subjects"].append(subj)
        
        # ============ RESULT FIELDS ============
        # Total marks
        total = extract_numeric_field(r'"total_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', json_text)
        if total:
            result["result"]["total_marks"] = total
        
        # Max total marks
        max_total = extract_numeric_field(r'"max_total_marks"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', json_text)
        if max_total:
            result["result"]["max_total_marks"] = max_total
        
        # Percentage
        percentage = extract_numeric_field(r'"percentage"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', json_text)
        if percentage:
            result["result"]["percentage"] = percentage
        
        # CGPA
        cgpa = extract_numeric_field(r'"cgpa"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', json_text)
        if cgpa:
            result["result"]["cgpa"] = cgpa
        
        # SGPA
        sgpa = extract_numeric_field(r'"sgpa"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', json_text)
        if sgpa:
            result["result"]["sgpa"] = sgpa
        
        # Division
        division = extract_field(r'"division"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if division:
            result["result"]["division"] = division
        
        # Result status (PASS/FAIL)
        status = extract_field(r'"result_status"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if status:
            result["result"]["result_status"] = status
        
        # Grade
        result_grade = extract_field(r'"result"[^}]*"grade"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if result_grade:
            result["result"]["grade"] = result_grade
        
        # Total credits
        total_credits = extract_numeric_field(r'"total_credits"\s*:\s*\{\s*"value"\s*:\s*(\d+(?:\.\d+)?)', json_text)
        if total_credits:
            result["result"]["total_credits"] = total_credits
        
        # ============ METADATA ============
        doc_type = extract_field(r'"document_type"\s*:\s*\{\s*"value"\s*:\s*"([^"]+)"', json_text)
        if doc_type:
            result["metadata"]["document_type"] = doc_type
        
        return result

class OpenAIExtractor(BaseLLMExtractor):
    
    def __init__(self):
        try:
            from openai import OpenAI
            
            if not settings.openai_api_key:
                raise LLMConnectionError("OpenAI API key not configured")
            
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.model = settings.openai_model
            logger.info(f"Initialized OpenAI extractor with model: {self.model}")
            
        except ImportError:
            raise LLMConnectionError("openai package not installed")
        except Exception as e:
            raise LLMConnectionError(f"Failed to initialize OpenAI: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check OpenAI API connectivity"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=10
            )
            return "ok" in response.choices[0].message.content.lower()
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
    
    async def extract(
        self, 
        images: List[Tuple[bytes, str]]
    ) -> Dict[str, Any]:
        try:
            import base64
            
            # Build messages with images
            content = [{"type": "text", "text": EXTRACTION_USER_PROMPT}]
            
            for image_bytes, mime_type in images:
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "high"
                    }
                })
            
            logger.info(f"Sending {len(images)} image(s) to OpenAI for extraction")
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": content}
                ],
                max_tokens=8192,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            
            if not response_text:
                raise LLMExtractionError("OpenAI returned empty response")
            
            # Parse response
            data = self._parse_json_response(response_text)
            
            # Calculate overall confidence
            data["extraction_confidence"] = self._calculate_overall_confidence(data)
            
            return data
            
        except LLMExtractionError:
            raise
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            raise LLMExtractionError(f"OpenAI extraction failed: {str(e)}")

class LLMExtractionService:
    def __init__(self):
        """Initialize service (extractor created lazily on first use)."""
        self.extractor: Optional[BaseLLMExtractor] = None
        self.provider: str = settings.default_llm_provider
        self._initialized = False
        self._current_api_key: Optional[str] = None  # Track current key for re-init
    
    async def initialize(self, user_api_key: Optional[str] = None):
 
        if self._initialized and user_api_key == self._current_api_key:
            return
        
        if self._initialized and user_api_key != self._current_api_key:
            self._initialized = False
            self.extractor = None
        
        try:
            if self.provider == "gemini":
                self.extractor = GeminiExtractor(api_key=user_api_key)
            elif self.provider == "openai":
                self.extractor = OpenAIExtractor()
            else:
                # Try Gemini first, then OpenAI
                try:
                    self.extractor = GeminiExtractor(api_key=user_api_key)
                    self.provider = "gemini"
                except LLMConnectionError:
                    self.extractor = GeminiExtractor(api_key=user_api_key)
                    self.provider = "gemini"
            
            self._initialized = True
            self._current_api_key = user_api_key
            key_info = "user-provided" if user_api_key else "system"
            logger.info(f"LLM Extraction Service initialized with provider: {self.provider} ({key_info} API key)")
            
        except LLMConnectionError as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise

    async def health_check(self) -> Tuple[bool, str]:
        if not self._initialized:
            await self.initialize()
        
        if self.extractor:
            is_healthy = await self.extractor.health_check()
            return is_healthy, self.provider
        
        return False, "none"
    
    async def extract_marksheet(
        self, 
        images: List[Tuple[bytes, str]],
        user_api_key: Optional[str] = None
    ) -> MarksheetExtraction:
        if not self._initialized or user_api_key != self._current_api_key:
            await self.initialize(user_api_key)
        
        if not self.extractor:
            raise LLMExtractionError("LLM extractor not initialized")
        
        start_time = time.time()
        
        raw_data = await self.extractor.extract(images)
        
        try:
            extraction = self._build_extraction_model(raw_data)
            logger.info(
                f"Extraction completed in {time.time() - start_time:.2f}s "
                f"with confidence: {extraction.extraction_confidence}"
            )
            return extraction
            
        except Exception as e:
            logger.error(f"Failed to build extraction model: {e}")
            raise LLMExtractionError(
                f"Failed to structure extracted data: {str(e)}",
                details={"raw_data": raw_data}
            )
    
    async def extract_marksheet_ocr(
        self,
        ocr_results: List[dict],
        user_api_key: Optional[str] = None
    ) -> MarksheetExtraction:
   
        # Initialize extractor if needed
        if not self._initialized or user_api_key != self._current_api_key:
            await self.initialize(user_api_key)

        if not self.extractor:
            raise LLMExtractionError("LLM extractor not initialized")

        start_time = time.time()

        # Convert OCR results into LLM-friendly text pages
        ocr_pages: List[Tuple[bytes, str]] = []

        combined_ocr_text = ""
        for page in ocr_results:
            page_num = page.get("page", "N/A")
            avg_conf = page.get("avg_confidence", 0)
            text = page.get("text", "")
            
            combined_ocr_text += f"""
=== PAGE {page_num} (OCR Confidence: {avg_conf:.2f}) ===
{text}

"""

        if isinstance(self.extractor, GeminiExtractor):
            raw_data = await self.extractor.extract_from_text(combined_ocr_text)
        else:
            # Fallback for other extractors - pass as text content
            raw_data = await self.extractor.extract_from_text([
                (combined_ocr_text.encode("utf-8"), "text/plain")
            ])

        try:
            extraction = self._build_extraction_model(raw_data)

            logger.info(
                f"OCR-based extraction completed in {time.time() - start_time:.2f}s "
                f"with confidence: {extraction.extraction_confidence}"
            )

            return extraction

        except Exception as e:
            logger.error(f"Failed to build OCR extraction model: {e}")
            raise LLMExtractionError(
                f"Failed to structure OCR extracted data: {str(e)}",
                details={"raw_data": raw_data}
            )

    async def extract_marksheet_from_ocr(self, ocr_results: List[dict]) -> MarksheetExtraction:
        return await self.extract_marksheet_ocr(ocr_results)

    def _build_extraction_model(self, data: Dict[str, Any]) -> MarksheetExtraction:

        from app.models.schemas import (
            MarksheetExtraction,
            CandidateDetails,
            ExaminationDetails,
            SubjectMarks,
            ComponentMarks,
            OverallResult,
            DocumentMetadata,
            StringField,
            IntField,
            FloatField,
            DateField,
            ConfidenceField
        )
        
        def safe_dict(val):
            return val if isinstance(val, dict) else {}
        
        def build_string_field(d) -> Optional[StringField]:
            if d is None:
                return None
            if isinstance(d, str):
                return StringField(value=d, confidence=0.5)
            if isinstance(d, (int, float)):
                return StringField(value=str(d), confidence=0.5)
            if not isinstance(d, dict):
                return StringField(value=str(d), confidence=0.5)
            if d.get("value") is None:
                return None
            return StringField(value=str(d.get("value")), confidence=float(d.get("confidence", 0.5)))
        
        def build_int_field(d) -> Optional[IntField]:
            if d is None:
                return None
            if isinstance(d, int):
                return IntField(value=d, confidence=0.5)
            if isinstance(d, float):
                return IntField(value=int(d), confidence=0.5)
            if isinstance(d, str):
                try:
                    return IntField(value=int(d), confidence=0.5)
                except:
                    return None
            if not isinstance(d, dict):
                return None
            if d.get("value") is None:
                return None
            try:
                return IntField(value=int(d.get("value")), confidence=float(d.get("confidence", 0.5)))
            except:
                return None
        
        def build_float_field(d) -> Optional[FloatField]:
            if d is None:
                return None
            if isinstance(d, (int, float)):
                return FloatField(value=float(d), confidence=0.5)
            if isinstance(d, str):
                try:
                    return FloatField(value=float(d), confidence=0.5)
                except:
                    return None
            if not isinstance(d, dict):
                return None
            if d.get("value") is None:
                return None
            try:
                return FloatField(value=float(d.get("value")), confidence=float(d.get("confidence", 0.5)))
            except:
                return None
        
        def build_date_field(d) -> Optional[DateField]:
            if d is None:
                return None
            if isinstance(d, str):
                return DateField(value=d, confidence=0.5)
            if not isinstance(d, dict):
                return DateField(value=str(d), confidence=0.5)
            if d.get("value") is None:
                return None
            return DateField(value=str(d.get("value")), confidence=float(d.get("confidence", 0.5)))
        
        def build_confidence_field(d) -> Optional[ConfidenceField]:
            if d is None:
                return None
            if isinstance(d, (bool, int, float, str)):
                return ConfidenceField(value=d, confidence=0.5)
            if not isinstance(d, dict):
                return None
            if d.get("value") is None:
                return None
            return ConfidenceField(value=d.get("value"), confidence=float(d.get("confidence", 0.5)))
        
        # Build candidate details
        candidate_data = safe_dict(data.get("candidate"))
        candidate = CandidateDetails(
            name=build_string_field(candidate_data.get("name")) or StringField(value="Unknown", confidence=0.0),
            father_name=build_string_field(candidate_data.get("father_name")),
            mother_name=build_string_field(candidate_data.get("mother_name")),
            guardian_name=build_string_field(candidate_data.get("guardian_name")),
            roll_number=build_string_field(candidate_data.get("roll_number")) or StringField(value="Unknown", confidence=0.0),
            registration_number=build_string_field(candidate_data.get("registration_number")),
            date_of_birth=build_date_field(candidate_data.get("date_of_birth")),
            gender=build_string_field(candidate_data.get("gender")),
            category=build_string_field(candidate_data.get("category")),
            photo_id=build_string_field(candidate_data.get("photo_id"))
        )
        
        # Build examination details
        exam_data = safe_dict(data.get("examination"))
        examination = ExaminationDetails(
            exam_name=build_string_field(exam_data.get("exam_name")) or StringField(value="Unknown", confidence=0.0),
            exam_year=build_string_field(exam_data.get("exam_year")) or StringField(value="Unknown", confidence=0.0),
            exam_month=build_string_field(exam_data.get("exam_month")),
            exam_session=build_string_field(exam_data.get("exam_session")),
            board_university=build_string_field(exam_data.get("board_university")) or StringField(value="Unknown", confidence=0.0),
            institution_name=build_string_field(exam_data.get("institution_name")),
            institution_code=build_string_field(exam_data.get("institution_code")),
            course_name=build_string_field(exam_data.get("course_name")),
            semester=build_string_field(exam_data.get("semester"))
        )
        
        # Build subjects
        subjects = []
        subj_list = data.get("subjects", [])
        if not isinstance(subj_list, list):
            subj_list = []
        for subj_data in subj_list:
            if not isinstance(subj_data, dict):
                continue
            # Build components if present
            components = None
            comp_list = subj_data.get("components")
            if comp_list and isinstance(comp_list, list):
                components = []
                for comp_data in comp_list:
                    if not isinstance(comp_data, dict):
                        continue
                    comp = ComponentMarks(
                        component_name=build_string_field(comp_data.get("component_name")) or StringField(value="Unknown", confidence=0.0),
                        max_marks=build_float_field(comp_data.get("max_marks")),
                        obtained_marks=build_float_field(comp_data.get("obtained_marks"))
                    )
                    components.append(comp)
            
            subject = SubjectMarks(
                subject_code=build_string_field(subj_data.get("subject_code")),
                subject_name=build_string_field(subj_data.get("subject_name")) or StringField(value="Unknown", confidence=0.0),
                subject_group=build_string_field(subj_data.get("subject_group")),
                max_marks=build_float_field(subj_data.get("max_marks")),
                obtained_marks=build_float_field(subj_data.get("obtained_marks")),
                credits=build_float_field(subj_data.get("credits")),
                grade=build_string_field(subj_data.get("grade")),
                grade_point=build_float_field(subj_data.get("grade_point")),
                theory_marks=build_float_field(subj_data.get("theory_marks")),
                practical_marks=build_float_field(subj_data.get("practical_marks")),
                oral_marks=build_float_field(subj_data.get("oral_marks")),
                written_marks=build_float_field(subj_data.get("written_marks")),
                internal_marks=build_float_field(subj_data.get("internal_marks")),
                external_marks=build_float_field(subj_data.get("external_marks")),
                is_pass=build_confidence_field(subj_data.get("is_pass")),
                components=components
            )
            subjects.append(subject)
        
        # Build result
        result_data = safe_dict(data.get("result"))
        result = OverallResult(
            total_marks=build_float_field(result_data.get("total_marks")),
            max_total_marks=build_float_field(result_data.get("max_total_marks")),
            percentage=build_float_field(result_data.get("percentage")),
            cgpa=build_float_field(result_data.get("cgpa")),
            sgpa=build_float_field(result_data.get("sgpa")),
            total_credits=build_float_field(result_data.get("total_credits")),
            division=build_string_field(result_data.get("division")),
            result_status=build_string_field(result_data.get("result_status")) or StringField(value="Unknown", confidence=0.0),
            rank=build_int_field(result_data.get("rank")),
            grade=build_string_field(result_data.get("grade")),
            distinction=build_string_field(result_data.get("distinction"))
        )
        
        # Build metadata
        meta_data = safe_dict(data.get("metadata"))
        metadata = DocumentMetadata(
            issue_date=build_date_field(meta_data.get("issue_date")),
            issue_place=build_string_field(meta_data.get("issue_place")),
            certificate_number=build_string_field(meta_data.get("certificate_number")),
            verification_code=build_string_field(meta_data.get("verification_code")),
            document_type=build_string_field(meta_data.get("document_type")) or StringField(value="Marksheet", confidence=0.5),
            signatory_name=build_string_field(meta_data.get("signatory_name")),
            signatory_designation=build_string_field(meta_data.get("signatory_designation"))
        )
        
        return MarksheetExtraction(
            candidate=candidate,
            examination=examination,
            subjects=subjects,
            result=result,
            metadata=metadata,
            extraction_confidence=data.get("extraction_confidence", 0.5),
            processing_notes=data.get("processing_notes")
        )


llm_service = LLMExtractionService()
