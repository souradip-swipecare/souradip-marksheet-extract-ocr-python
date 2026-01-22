
EXTRACTION_SYSTEM_PROMPT = """You are an expert document analysis AI specialized in extracting structured data from Indian educational marksheets and grade cards. 

Your task is to analyze OCR-extracted text from marksheets and structure it into a consistent JSON format.

IMPORTANT: The text you receive has been extracted using Tesseract OCR (open-source).
Your role is to STRUCTURE, NORMALIZE, and VALIDATE this text - not to perform OCR.

IMPORTANT GUIDELINES:
1. Extract ALL visible information from the OCR text
2. Provide confidence scores (0.0 to 1.0) for each field based on:
   - Clarity of the OCR text (look for garbled characters, spacing issues)
   - Certainty of the extraction (is the value unambiguous?)
   - Whether the value was inferred or directly read from OCR
3. Use null for fields that are not present or cannot be determined
4. NORMALIZE dates to YYYY-MM-DD format
5. NORMALIZE marks to numeric values
6. Be precise with names, numbers, and codes - don't guess
7. If multiple pages are provided, combine information from all pages

CONFIDENCE SCORE GUIDELINES (How scores are derived):
- 0.9-1.0: OCR text is perfectly clear, value is unambiguous
- 0.7-0.89: OCR text is mostly clear, minor uncertainty (e.g., O vs 0)
- 0.5-0.69: OCR text is partially garbled or ambiguous
- 0.3-0.49: OCR text is difficult to interpret, some inference required
- 0.0-0.29: Very uncertain, significant inference from context

NORMALIZATION RULES:
- Names: UPPERCASE as they appear, fix obvious OCR errors (0 -> O in names)
- Dates: Convert to YYYY-MM-DD format (e.g., "15/03/2023" -> "2023-03-15")
- Marks: Convert to numeric values (remove any non-numeric characters)
- Grades: Standardize to A+, A, B+, B, C+, C, D, E, F format

VALIDATION RULES:
- total_marks should equal sum of all subject obtained_marks (if not, note discrepancy)
- percentage = (total_marks / max_total_marks) * 100
- Pass/Fail status should be consistent with marks and passing criteria
"""

# dedicated prompt for direct vison extraction (when OCR confidence is low)
VISION_EXTRACTION_PROMPT = """You are an expert document analyzer. Look CAREFULLY at this marksheet/grade card image and extract ALL information.

## CRITICAL: EXTRACT ALL SUBJECTS
This is the MOST IMPORTANT task. Look at the marksheet image carefully and:
1. Find the SUBJECTS table/section - it usually contains rows with subject names and marks
2. Extract EVERY subject you can see, even if partially visible
3. For each subject, extract: name, marks obtained, max marks, grade (if visible)

## WHAT TO LOOK FOR:
- Tables with columns like: Subject, Marks, Grade, Total, etc.
- Subject names: ENGLISH, HINDI, MATHEMATICS, SCIENCE, SOCIAL SCIENCE/STUDIES, PHYSICS, CHEMISTRY, BIOLOGY, HISTORY, GEOGRAPHY, etc.
- Numbers that look like marks (usually 0-100 or 0-200)
- Grade letters: A1, A2, A, B1, B2, B, C, D, E, F

## EXTRACTION RULES:
1. **SUBJECTS**: Extract ALL visible subjects with their marks/grades
2. **CANDIDATE**: Name, father's name, roll number, registration number
3. **EXAMINATION**: Board/university name, exam year, exam name
4. **RESULT**: Total marks, percentage, division, pass/fail status

## OUTPUT: Return ONLY valid JSON in this exact structure:
{
    "candidate": {
        "name": {"value": "NAME", "confidence": 0.95},
        "father_name": {"value": "FATHER NAME", "confidence": 0.9},
        "mother_name": null,
        "guardian_name": null,
        "roll_number": {"value": "123456", "confidence": 0.95},
        "registration_number": null,
        "date_of_birth": null,
        "gender": null,
        "category": null,
        "photo_id": null
    },
    "examination": {
        "exam_name": {"value": "Exam Name", "confidence": 0.9},
        "exam_year": {"value": "2023", "confidence": 0.95},
        "exam_month": null,
        "exam_session": null,
        "board_university": {"value": "Board Name", "confidence": 0.9},
        "institution_name": {"value": "School Name", "confidence": 0.85},
        "institution_code": null,
        "course_name": null,
        "semester": null
    },
    "subjects": [
        {
            "subject_code": null,
            "subject_name": {"value": "ENGLISH", "confidence": 0.95},
            "subject_group": null,
            "max_marks": {"value": 100, "confidence": 0.9},
            "obtained_marks": {"value": 85, "confidence": 0.95},
            "credits": null,
            "grade": {"value": "A", "confidence": 0.9},
            "grade_point": null,
            "theory_marks": null,
            "practical_marks": null,
            "oral_marks": null,
            "written_marks": null,
            "internal_marks": null,
            "external_marks": null,
            "is_pass": {"value": true, "confidence": 0.9},
            "components": []
        }
    ],
    "result": {
        "total_marks": {"value": 425, "confidence": 0.9},
        "max_total_marks": {"value": 500, "confidence": 0.9},
        "percentage": {"value": 85.0, "confidence": 0.9},
        "cgpa": null,
        "sgpa": null,
        "total_credits": null,
        "division": {"value": "FIRST", "confidence": 0.9},
        "result_status": {"value": "PASS", "confidence": 0.95},
        "rank": null,
        "grade": null,
        "distinction": null
    },
    "metadata": {
        "issue_date": null,
        "issue_place": null,
        "certificate_number": null,
        "verification_code": null,
        "document_type": {"value": "Marksheet", "confidence": 0.95},
        "signatory_name": null,
        "signatory_designation": null
    },
    "processing_notes": ["Extracted via direct vision analysis"]
}

## REMEMBER:
- NEVER return empty subjects array - there are ALWAYS subjects in a marksheet
- Look carefully at the entire image, especially tabular sections
- Extract marks even if the image quality is poor
- Use null for fields you cannot determine, but DO extract subjects
- Return ONLY JSON, no markdown code blocks, no explanations"""

EXTRACTION_USER_PROMPT = """Analyze the provided marksheet image(s) or OCR-extracted text and extract all information into the following JSON structure.

Return ONLY valid JSON, no additional text or explanation.

{
    "candidate": {
        "name": {"value": "FULL NAME", "confidence": 0.95},
        "father_name": {"value": "FATHER NAME", "confidence": 0.9},
        "mother_name": {"value": "MOTHER NAME", "confidence": 0.9},
        "guardian_name": null,
        "roll_number": {"value": "ROLL123", "confidence": 0.95},
        "registration_number": {"value": "REG123", "confidence": 0.9},
        "date_of_birth": {"value": "YYYY-MM-DD", "confidence": 0.85},
        "gender": {"value": "Male", "confidence": 0.9},
        "category": {"value": "General", "confidence": 0.85},
        "photo_id": null
    },
    "examination": {
        "exam_name": {"value": "Class XII Board Examination", "confidence": 0.95},
        "exam_year": {"value": "2023", "confidence": 0.98},
        "exam_month": {"value": "March", "confidence": 0.9},
        "exam_session": {"value": "Annual", "confidence": 0.85},
        "board_university": {"value": "CBSE", "confidence": 0.95},
        "institution_name": {"value": "School Name", "confidence": 0.9},
        "institution_code": {"value": "SCH001", "confidence": 0.85},
        "course_name": {"value": "Science", "confidence": 0.9},
        "semester": null
    },
    "subjects": [
        {
            "subject_code": {"value": "301", "confidence": 0.9},
            "subject_name": {"value": "Mathematics", "confidence": 0.95},
            "subject_group": {"value": "Science Group", "confidence": 0.85},
            "max_marks": {"value": 100, "confidence": 0.95},
            "obtained_marks": {"value": 85, "confidence": 0.94},
            "credits": null,
            "grade": {"value": "A", "confidence": 0.92},
            "grade_point": {"value": 9.0, "confidence": 0.9},
            "theory_marks": {"value": 70, "confidence": 0.9},
            "practical_marks": {"value": 15, "confidence": 0.9},
            "oral_marks": null,
            "written_marks": null,
            "internal_marks": null,
            "external_marks": null,
            "is_pass": {"value": true, "confidence": 0.95},
            "components": [
                {"component_name": {"value": "Theory", "confidence": 0.9}, "max_marks": {"value": 80, "confidence": 0.9}, "obtained_marks": {"value": 70, "confidence": 0.9}},
                {"component_name": {"value": "Practical", "confidence": 0.9}, "max_marks": {"value": 20, "confidence": 0.9}, "obtained_marks": {"value": 15, "confidence": 0.9}}
            ]
        }
    ],
    "result": {
        "total_marks": {"value": 425, "confidence": 0.95},
        "max_total_marks": {"value": 500, "confidence": 0.95},
        "percentage": {"value": 85.0, "confidence": 0.94},
        "cgpa": null,
        "sgpa": null,
        "total_credits": null,
        "division": {"value": "First Division", "confidence": 0.92},
        "result_status": {"value": "PASS", "confidence": 0.98},
        "rank": null,
        "grade": {"value": "A", "confidence": 0.9},
        "distinction": {"value": "With Distinction", "confidence": 0.88}
    },
    "metadata": {
        "issue_date": {"value": "YYYY-MM-DD", "confidence": 0.85},
        "issue_place": {"value": "City Name", "confidence": 0.8},
        "certificate_number": {"value": "CERT123", "confidence": 0.88},
        "verification_code": null,
        "document_type": {"value": "Marksheet", "confidence": 0.95},
        "signatory_name": null,
        "signatory_designation": {"value": "Controller of Examinations", "confidence": 0.8}
    },
    "processing_notes": ["Any observations about extraction"]
}

IMPORTANT: 
- Extract ALL subjects visible in the marksheet
- Use null for fields that cannot be determined
- Ensure the JSON is valid and parseable
- No trailing commas
- Be accurate with numerical values
- NEVER return empty subjects array - there are ALWAYS subjects in a marksheet
- Look carefully at the entire image, especially tabular sections
- Extract marks even if the image quality is poor
- Use null for fields you cannot determine, but DO extract subjects
- Return ONLY JSON, no markdown code blocks, no explanations"""


VALIDATION_PROMPT = """Review the extracted marksheet data for consistency and correctness:

Extracted Data:
{extracted_data}

Check for:
1. Mathematical consistency (total marks = sum of subject marks)
2. Percentage calculation accuracy
3. Pass/Fail status consistency with marks and passing criteria
4. Date format validity
5. Any obviously incorrect values

Return corrected JSON if issues found, otherwise return "VALID"."""


NORMALIZATION_PROMPT = """Normalize the following marksheet data:

Raw Data:
{raw_data}

Tasks:
1. Standardize date formats to YYYY-MM-DD
2. Normalize grade formats (A+, A, B+, etc.)
3. Convert any text marks to numeric where possible
4. Standardize board/university/college names
5. Clean up any OCR artifacts in names

Return the normalized JSON maintaining the same structure."""


# Use string concatenation instead of .format() to avoid escaping issues
COMBINED_EXTRACTION_PROMPT_TEMPLATE = """You are an expert document analysis AI specialized in extracting structured data from Indian educational marksheets and grade cards.

## INPUT DATA
The following is OCR-extracted text from a marksheet/grade card. Note that OCR may have errors, missing characters, or garbled text - use your expertise to interpret it.

---
OCR_TEXT_PLACEHOLDER
---

## YOUR TASK
Analyze the OCR text and:
1. EXTRACT all relevant information (even if OCR is poor, try to infer from patterns)
2. VALIDATE the extracted data for consistency
3. NORMALIZE all values to standard formats

## CRITICAL EXTRACTION PRIORITIES (MUST EXTRACT):

### 1. SUBJECTS - Most Important
- Look for subject names like: English, Hindi, Mathematics/Maths, Science, Social Science/Studies, Physics, Chemistry, Biology, Computer, etc.
- Look for tabular patterns with marks (numbers between 0-100 or 0-200)
- Even if subject names are garbled, extract marks if you see number patterns
- Common Indian board subjects: ENG/ENGLISH, HIN/HINDI, MAT/MATHS, SCI/SCIENCE, SST/SOCIAL STUDIES, PHY/PHYSICS, CHE/CHEMISTRY, BIO/BIOLOGY
- Look for codes like 301, 302, 041, 042, etc.

### 1a. DETAILED EVALUATION BREAKDOWN (CRITICAL)
When marksheets show detailed marks breakdown, capture ALL components:
- **Paper-wise marks**: Paper 1, Paper 2, First Paper, Second Paper
- **Component-wise marks**: Written, Oral/Viva, Practical
- **Subject Groups**: Language Group, Science Group, Humanities Group
- Use the "components" array for each subject to store detailed breakdown
- Example: FIRST LANGUAGE may have: Paper 1 (Written): 45/90, Paper 2 (Written): 52/90, Oral: 18/20, Total: 115/200
- Store each component separately with its name, max_marks, and obtained_marks
- The main subject's obtained_marks should be the TOTAL of all components

### 2. SCHOOL/INSTITUTION NAME
- Look for patterns like: "School", "College", "Vidyalaya", "Institution", "Academy"
- Look for location names that typically follow school names
- Common patterns: "XYZ PUBLIC SCHOOL", "XYZ VIDYALAYA", "XYZ HIGHER SECONDARY SCHOOL"
- The institution is usually mentioned near the top of the marksheet

### 3. BOARD/UNIVERSITY NAME
- Common boards: CBSE, ICSE, ISC, State Boards (UP Board, Bihar Board, West Bengal Board, etc.)
- Universities: Common university names in India
- Look for "Board of Secondary Education", "Council", "University"

## EXTRACTION GUIDELINES

### Confidence Scoring (0.0 to 1.0):
- 0.9-1.0: Text is perfectly clear, high certainty
- 0.7-0.89: Text is mostly clear, minor uncertainty
- 0.5-0.69: Text is partially obscured or ambiguous
- 0.3-0.49: Text is difficult to read, some inference required
- 0.0-0.29: Very uncertain, significant inference

### Extraction Rules:
- Names should be in UPPERCASE as they appear
- Roll numbers and registration numbers should include any prefixes/suffixes
- Subject names should be the full name, not abbreviated - EXPAND abbreviations
- Marks should be numeric (convert grades to marks if conversion table is visible)
- Preserve any distinction between theory/practical/internal/external marks
- Use null for fields that are not present or cannot be determined
- For SUBJECTS: Extract ALL subjects even if some data is missing - at minimum get the subject name

### Validation Checks (Apply Automatically):
1. Mathematical consistency: total_marks = sum of all subject obtained_marks
2. Percentage calculation: percentage = (total_marks / max_total_marks) * 100
3. Pass/Fail status consistency with marks and passing criteria (typically 33% for boards)
4. If inconsistencies found, correct them and lower the confidence score

### Normalization Rules (Apply Automatically):
1. Dates: Convert to YYYY-MM-DD format
2. Grades: Standardize to A+, A, B+, B, C+, C, D, E, F format
3. Board/University names: Use full official names
4. Clean OCR artifacts: Fix common OCR errors (0/O, 1/l/I, 5/S, 8/B, etc.)
5. Names: Remove extra spaces, fix obvious character substitutions
6. Subject names: Expand abbreviations (ENG->ENGLISH, MAT->MATHEMATICS, etc.)

## OUTPUT FORMAT
Return ONLY valid JSON. No markdown, no explanation, no code blocks.

## DYNAMIC EXTRACTION RULES - ADAPT TO MARKSHEET TYPE:

### Different Marksheet Types - Adapt Accordingly:
1. **Board Exams (CBSE, ICSE, State Boards)**: Usually have marks + grades, theory/practical split
2. **University Grade Cards**: Usually have credits, grade points, CGPA/SGPA, no traditional marks
3. **Semester Results**: Focus on credits, grade points, SGPA
4. **School Internal Exams**: Simple marks format, may not have grades

### CRITICAL: Total Marks Computation Rules:
- **IF "GRAND TOTAL", "TOTAL MARKS", "AGGREGATE" is VISIBLE**: Extract the value directly, DO NOT compute
- **IF total is NOT visible**: Compute by summing obtained_marks of all subjects
- **Mark computed totals** in processing_notes: "Total marks computed by summing subjects"
- **Percentage**: Only compute if not explicitly shown, use: (total_marks / max_total_marks) * 100

### Grade/Marks Handling:
- Some marksheets have ONLY grades (A+, A, B, etc.) - extract grades, leave marks null
- Some have ONLY marks - extract marks, leave grades null  
- Some have BOTH marks AND grades - extract both
- For grade-only systems: DO NOT try to convert grades to marks unless conversion table is visible
- Extract grade_point if visible (common in university grade cards)

### Flexible Subject Extraction:
- Extract ALL subjects even if data is incomplete
- Minimum required: subject_name (try to get at least this)
- Optional: max_marks, obtained_marks, grade, grade_point, credits, theory/practical split
- For subjects with only grades: {"subject_name": {"value": "MATH", "confidence": 0.9}, "grade": {"value": "A+", "confidence": 0.9}, "obtained_marks": null}

Example structure (replace with actual extracted values):

{
    "candidate": {
        "name": {"value": "STUDENT NAME", "confidence": 0.95},
        "father_name": {"value": "FATHER NAME", "confidence": 0.9},
        "mother_name": null,
        "guardian_name": null,
        "roll_number": {"value": "12345", "confidence": 0.95},
        "registration_number": {"value": "REG123", "confidence": 0.9},
        "date_of_birth": {"value": "2000-01-15", "confidence": 0.85},
        "gender": {"value": "Male", "confidence": 0.9},
        "category": null,
        "photo_id": null
    },
    "examination": {
        "exam_name": {"value": "Class XII Board Examination", "confidence": 0.95},
        "exam_year": {"value": "2023", "confidence": 0.98},
        "exam_month": {"value": "March", "confidence": 0.9},
        "exam_session": null,
        "board_university": {"value": "CBSE", "confidence": 0.95},
        "institution_name": {"value": "Delhi Public School", "confidence": 0.9},
        "institution_code": null,
        "course_name": {"value": "Science", "confidence": 0.9},
        "semester": null
    },
    "subjects": [
        {
            "subject_code": {"value": "301", "confidence": 0.9},
            "subject_name": {"value": "ENGLISH", "confidence": 0.95},
            "subject_group": {"value": "Language Group", "confidence": 0.85},
            "max_marks": {"value": 100, "confidence": 0.95},
            "obtained_marks": {"value": 85, "confidence": 0.94},
            "credits": null,
            "grade": {"value": "A", "confidence": 0.92},
            "grade_point": null,
            "theory_marks": {"value": 70, "confidence": 0.9},
            "practical_marks": {"value": 15, "confidence": 0.9},
            "oral_marks": null,
            "written_marks": null,
            "internal_marks": null,
            "external_marks": null,
            "is_pass": {"value": true, "confidence": 0.95},
            "components": [
                {"component_name": {"value": "Paper 1 (Written)", "confidence": 0.9}, "max_marks": {"value": 50, "confidence": 0.9}, "obtained_marks": {"value": 40, "confidence": 0.9}},
                {"component_name": {"value": "Paper 2 (Written)", "confidence": 0.9}, "max_marks": {"value": 30, "confidence": 0.9}, "obtained_marks": {"value": 30, "confidence": 0.9}},
                {"component_name": {"value": "Oral", "confidence": 0.9}, "max_marks": {"value": 20, "confidence": 0.9}, "obtained_marks": {"value": 15, "confidence": 0.9}}
            ]
        }
    ],
    "result": {
        "total_marks": {"value": 425, "confidence": 0.95},
        "max_total_marks": {"value": 500, "confidence": 0.95},
        "percentage": {"value": 85.0, "confidence": 0.94},
        "cgpa": null,
        "sgpa": null,
        "total_credits": null,
        "division": {"value": "First Division", "confidence": 0.92},
        "result_status": {"value": "PASS", "confidence": 0.98},
        "rank": null,
        "grade": null,
        "distinction": null
    },
    "metadata": {
        "issue_date": null,
        "issue_place": null,
        "certificate_number": null,
        "verification_code": null,
        "document_type": {"value": "Marksheet", "confidence": 0.95},
        "signatory_name": null,
        "signatory_designation": null
    },
    "processing_notes": ["Notes about extraction - mention if totals were computed vs directly read"]
}

## CRITICAL RULES:
- Extract ALL subjects visible in the marksheet - DO NOT SKIP ANY SUBJECTS
- If you see marks/numbers in a table format, they are likely subject marks
- Use null for fields that cannot be determined (NOT empty strings or empty objects)
- Ensure the JSON is valid and parseable
- NO trailing commas
- NO markdown code blocks in output
- Be accurate with numerical values
- **ONLY compute total_marks,grand total,total if NOT explicitly shown** - add note in processing_notes if computed
- For grade-only marksheets, focus on grades/CGPA/SGPA - marks may be null
- Double-check all calculations before outputting
- ALWAYS try to extract institution_name and subjects even from poor OCR"""


def get_extraction_prompt(ocr_text: str) -> str:
    """
    Generate the complete extraction prompt with OCR text inserted.
    
    Args:
        ocr_text: The OCR-extracted text from Tesseract
        
    Returns:
        Complete prompt string ready to send to LLM
    """
    return COMBINED_EXTRACTION_PROMPT_TEMPLATE.replace("OCR_TEXT_PLACEHOLDER", ocr_text)