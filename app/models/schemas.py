
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import date
from enum import Enum



class ConfidenceField(BaseModel):

    value: Any = Field(..., description="The extracted value")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0 and 1"
    )
    bounding_box: Optional[Dict[str, int]] = Field(
        default=None,
        description="Bounding box coordinates {x, y, width, height} if available"
    )


class StringField(ConfidenceField):
    value: Optional[str] = Field(None, description="String value")


class IntField(ConfidenceField):
    value: Optional[int] = Field(None, description="Integer value")


class FloatField(ConfidenceField):
    value: Optional[float] = Field(None, description="Float value")


class DateField(ConfidenceField):
    value: Optional[str] = Field(None, description="Date value in YYYY-MM-DD format")


# candidate model
class CandidateDetails(BaseModel):
    name: StringField = Field(..., description="Full name of the candidate")
    father_name: Optional[StringField] = Field(None, description="Father's name")
    mother_name: Optional[StringField] = Field(None, description="Mother's name")
    guardian_name: Optional[StringField] = Field(None, description="Guardian's name if applicable")
    roll_number: StringField = Field(..., description="Roll number / Seat number")
    registration_number: Optional[StringField] = Field(None, description="Registration number")
    date_of_birth: Optional[DateField] = Field(None, description="Date of birth")
    gender: Optional[StringField] = Field(None, description="Gender")
    category: Optional[StringField] = Field(None, description="Category (General/OBC/SC/ST etc)")
    photo_id: Optional[StringField] = Field(None, description="Any photo ID number if present")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": {"value": "RAHUL SHARMA", "confidence": 0.95},
                "father_name": {"value": "SURESH SHARMA", "confidence": 0.92},
                "roll_number": {"value": "12345678", "confidence": 0.98},
                "date_of_birth": {"value": "2000-05-15", "confidence": 0.88}
            }
        }


# exam detail
class ExaminationDetails(BaseModel):
    exam_name: StringField = Field(..., description="Name of the examination")
    exam_year: StringField = Field(..., description="Year of examination")
    exam_month: Optional[StringField] = Field(None, description="Month of examination")
    exam_session: Optional[StringField] = Field(None, description="Session (e.g., Annual, Supplementary)")
    board_university: StringField = Field(..., description="Name of Board or University")
    institution_name: Optional[StringField] = Field(None, description="School/College name")
    institution_code: Optional[StringField] = Field(None, description="School/College code")
    course_name: Optional[StringField] = Field(None, description="Course/Stream name")
    semester: Optional[StringField] = Field(None, description="Semester if applicable")
    
    class Config:
        json_schema_extra = {
            "example": {
                "exam_name": {"value": "Higher Secondary Examination", "confidence": 0.95},
                "exam_year": {"value": "2023", "confidence": 0.99},
                "board_university": {"value": "CBSE", "confidence": 0.97}
            }
        }


# subwis marks
class GradeEnum(str, Enum):
    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    C_PLUS = "C+"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    PASS = "PASS"
    FAIL = "FAIL"
    ABSENT = "ABSENT"
    OTHER = "OTHER"


# Component marks for detailed evaluation (papers, oral, written, etc.)
class ComponentMarks(BaseModel):
    component_name: StringField = Field(..., description="Name of component (e.g., Paper 1, Oral, Written)")
    max_marks: Optional[FloatField] = Field(None, description="Maximum marks for this component")
    obtained_marks: Optional[FloatField] = Field(None, description="Marks obtained in this component")
    
    class Config:
        json_schema_extra = {
            "example": {
                "component_name": {"value": "Paper 1 (Written)", "confidence": 0.95},
                "max_marks": {"value": 90, "confidence": 0.95},
                "obtained_marks": {"value": 45, "confidence": 0.94}
            }
        }


class SubjectMarks(BaseModel):
    subject_code: Optional[StringField] = Field(None, description="Subject code")
    subject_name: StringField = Field(..., description="Name of the subject")
    subject_group: Optional[StringField] = Field(None, description="Subject group (e.g., Language Group, Science Group)")
    max_marks: Optional[FloatField] = Field(None, description="Maximum marks")
    obtained_marks: Optional[FloatField] = Field(None, description="Marks obtained")
    credits: Optional[FloatField] = Field(None, description="Credits for the subject")
    grade: Optional[StringField] = Field(None, description="Grade obtained")
    grade_point: Optional[FloatField] = Field(None, description="Grade point")
    theory_marks: Optional[FloatField] = Field(None, description="Theory marks if separate")
    practical_marks: Optional[FloatField] = Field(None, description="Practical marks if separate")
    oral_marks: Optional[FloatField] = Field(None, description="Oral/Viva marks if separate")
    written_marks: Optional[FloatField] = Field(None, description="Written exam marks if separate")
    internal_marks: Optional[FloatField] = Field(None, description="Internal assessment marks")
    external_marks: Optional[FloatField] = Field(None, description="External assessment marks")
    is_pass: Optional[ConfidenceField] = Field(None, description="Whether passed in this subject")
    # Detailed component breakdown (for marksheets with Paper 1, Paper 2, Oral, etc.)
    components: Optional[List[ComponentMarks]] = Field(None, description="Detailed breakdown of marks by component (papers, oral, written, etc.)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "subject_name": {"value": "FIRST LANGUAGE (BENGALI)", "confidence": 0.96},
                "subject_group": {"value": "Language Group", "confidence": 0.9},
                "max_marks": {"value": 200, "confidence": 0.95},
                "obtained_marks": {"value": 115, "confidence": 0.94},
                "components": [
                    {"component_name": {"value": "Paper 1 (Written)", "confidence": 0.95}, "max_marks": {"value": 90, "confidence": 0.95}, "obtained_marks": {"value": 45, "confidence": 0.94}},
                    {"component_name": {"value": "Paper 2 (Written)", "confidence": 0.95}, "max_marks": {"value": 90, "confidence": 0.95}, "obtained_marks": {"value": 52, "confidence": 0.94}},
                    {"component_name": {"value": "Oral", "confidence": 0.95}, "max_marks": {"value": 20, "confidence": 0.95}, "obtained_marks": {"value": 18, "confidence": 0.94}}
                ],
                "grade": {"value": "B", "confidence": 0.92}
            }
        }


# ovral res
class OverallResult(BaseModel):
    total_marks: Optional[FloatField] = Field(None, description="Total marks obtained")
    max_total_marks: Optional[FloatField] = Field(None, description="Maximum total marks")
    percentage: Optional[FloatField] = Field(None, description="Percentage obtained")
    cgpa: Optional[FloatField] = Field(None, description="CGPA if applicable")
    sgpa: Optional[FloatField] = Field(None, description="SGPA if applicable")
    total_credits: Optional[FloatField] = Field(None, description="Total credits earned")
    division: Optional[StringField] = Field(None, description="Division (First/Second/Third)")
    result_status: StringField = Field(..., description="Pass/Fail/Compartment status")
    rank: Optional[IntField] = Field(None, description="Rank if mentioned")
    grade: Optional[StringField] = Field(None, description="Overall grade")
    distinction: Optional[StringField] = Field(None, description="Any distinction/honors")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_marks": {"value": 425, "confidence": 0.95},
                "max_total_marks": {"value": 500, "confidence": 0.95},
                "percentage": {"value": 85.0, "confidence": 0.96},
                "result_status": {"value": "PASS", "confidence": 0.99},
                "division": {"value": "First Division with Distinction", "confidence": 0.94}
            }
        }


# document data
class DocumentMetadata(BaseModel):
    issue_date: Optional[DateField] = Field(None, description="Date when marksheet was issued")
    issue_place: Optional[StringField] = Field(None, description="Place of issue")
    certificate_number: Optional[StringField] = Field(None, description="Certificate/Marksheet number")
    verification_code: Optional[StringField] = Field(None, description="QR code or verification code if present")
    document_type: StringField = Field(..., description="Type of document (Marksheet/Grade Card/Transcript)")
    signatory_name: Optional[StringField] = Field(None, description="Name of the signatory")
    signatory_designation: Optional[StringField] = Field(None, description="Designation of signatory")
    
    class Config:
        json_schema_extra = {
            "example": {
                "issue_date": {"value": "2023-07-15", "confidence": 0.88},
                "document_type": {"value": "Marksheet", "confidence": 0.95},
                "certificate_number": {"value": "MS/2023/123456", "confidence": 0.91}
            }
        }

# sheet extract model
class MarksheetExtraction(BaseModel):
    candidate: CandidateDetails = Field(..., description="Candidate personal details")
    examination: ExaminationDetails = Field(..., description="Examination details")
    subjects: List[SubjectMarks] = Field(..., description="Subject-wise marks")
    result: OverallResult = Field(..., description="Overall result")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    
    extraction_confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Overall extraction confidence (weighted average)"
    )
    processing_notes: Optional[List[str]] = Field(
        None, 
        description="Any notes about extraction quality or issues"
    )
    raw_text: Optional[str] = Field(
        None, 
        description="Raw OCR text if requested"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "candidate": {
                    "name": {"value": "RAHUL SHARMA", "confidence": 0.95},
                    "roll_number": {"value": "12345678", "confidence": 0.98}
                },
                "examination": {
                    "exam_name": {"value": "Class XII Board Examination", "confidence": 0.95},
                    "exam_year": {"value": "2023", "confidence": 0.99},
                    "board_university": {"value": "CBSE", "confidence": 0.97}
                },
                "subjects": [
                    {
                        "subject_name": {"value": "Mathematics", "confidence": 0.96},
                        "max_marks": {"value": 100, "confidence": 0.95},
                        "obtained_marks": {"value": 85, "confidence": 0.94}
                    }
                ],
                "result": {
                    "total_marks": {"value": 425, "confidence": 0.95},
                    "result_status": {"value": "PASS", "confidence": 0.99}
                },
                "metadata": {
                    "document_type": {"value": "Marksheet", "confidence": 0.95}
                },
                "extraction_confidence": 0.93
            }
        }

# api res model
class ExtractionResponse(BaseModel):
    success: bool = Field(..., description="whether extraction was successful")
    data: Optional[MarksheetExtraction] = Field(None, description="extracted data")
    error: Optional[str] = Field(None, description="error message if failed")
    processing_time_ms: float = Field(..., description="processing time in milliseconds")
    file_name: str = Field(..., description="original file name")
    file_size_bytes: Optional[int] = Field(None, description="file size in bytes")
    extraction_method: Optional[str] = Field(
        None, 
        description="Method used: 'ocr_llm' (Tesseract + LLM) or 'direct_llm_vision' (LLM Vision fallback)"
    )
    ocr_confidence: Optional[float] = Field(
        None, 
        description="Average OCR confidence (0.0-1.0). If < 0.4, direct LLM vision was used"
    )


class BatchExtractionResponse(BaseModel):
    success: bool = Field(..., description="Whether batch processing completed")
    total_files: int = Field(..., description="Total files processed")
    successful: int = Field(..., description="Successfully processed files")
    failed: int = Field(..., description="Failed extractions")
    results: List[ExtractionResponse] = Field(..., description="Individual results")
    total_processing_time_ms: float = Field(..., description="Total processing time")

# health res with llm provider
class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    llm_provider: str = Field(..., description="Active LLM provider")
    llm_status: str = Field(..., description="LLM connection status")

# error res
class ErrorResponse(BaseModel):
    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for debugging")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
