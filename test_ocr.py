import asyncio
import sys
import json
from pathlib import Path
sys.path.insert(0, '/Users/souradipbiswas/Downloads/python_api_extract')

from app.services.ocr_service import ocr_service
from app.services.extraction import llm_service

OCR_CONFIDENCE_THRESHOLD = 0.40  


def test_ocr_only(img_path):
    """Test just the OCR service without LLM"""
    print(f"\n{'='*60}")
    print(f"Testing OCR: {img_path}")
    
    with open(img_path, 'rb') as f:
        image_bytes = f.read()
    
    # Test OCR extraction with text saving
    ocr_result = ocr_service.extract_text(image_bytes, save_text=True)
    
    print(f"\nOCR Results:")
    print(f"  Confidence: {ocr_result['avg_confidence']:.2%}")
    print(f"  Word Count: {ocr_result.get('word_count', len(ocr_result.get('words', [])))}")
    print(f"  High Conf Words: {ocr_result.get('high_conf_words', 0)}")
    print(f"  Saved to: {ocr_result.get('saved_file', 'N/A')}")
    print(f"\nText preview (first 500 chars):")
    print(f"  {ocr_result['raw_text'][:500]}")
    
    return ocr_result


async def test_full_pipeline(img_path):
    print(f"\n{'='*60}")
    print(f"Testing: {img_path}")
    
    with open(img_path, 'rb') as f:
        image_bytes = f.read()
    
    # Get file extension for mime type
    ext = img_path.split('.')[-1].lower()
    mime_map = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp'}
    mime_type = mime_map.get(ext, 'image/jpeg')
    
    # Test 1: OCR extraction with saving
    print("\n--- OCR Extraction ---")
    ocr_result = ocr_service.extract_text(image_bytes, save_text=True)
    ocr_confidence = ocr_result['avg_confidence']
    print(f"OCR chars: {len(ocr_result['raw_text'])}")
    print(f"OCR confidence: {ocr_confidence:.3f}")
    print(f"OCR word count: {ocr_result.get('word_count', 0)}")
    print(f"OCR high conf words: {ocr_result.get('high_conf_words', 0)}")
    print(f"Saved to: {ocr_result.get('saved_file', 'N/A')}")
    print(f"OCR text preview:\n{ocr_result['raw_text'][:600]}...")
    
    # Test 2: Hybrid Pipeline
    print("\n--- Hybrid Pipeline ---")
    await llm_service.initialize()
    
    try:
        # Decide extraction method based on OCR confidence
        if ocr_confidence >= OCR_CONFIDENCE_THRESHOLD:
            print(f"✓ Using OCR + LLM (confidence {ocr_confidence:.2f} >= {OCR_CONFIDENCE_THRESHOLD})")
            extraction_method = "ocr_llm"
            
            ocr_results = [{
                "page": 1,
                "text": ocr_result['raw_text'],
                "avg_confidence": ocr_result['avg_confidence'],
                "words": ocr_result['words']
            }]
            extraction = await llm_service.extract_marksheet_from_ocr(ocr_results)
        else:
            print(f"✗ Falling back to Direct LLM Vision (confidence {ocr_confidence:.2f} < {OCR_CONFIDENCE_THRESHOLD})")
            extraction_method = "direct_llm_vision"
            
            # Pass image directly to LLM
            images = [(image_bytes, mime_type)]
            extraction = await llm_service.extract_marksheet(images)
        
        print(f"\n=== EXTRACTED DATA (Method: {extraction_method}) ===")
        print(f"Overall confidence: {extraction.extraction_confidence}")
        
        # Candidate Info
        print(f"\n-- Candidate --")
        print(f"  Name: {extraction.candidate.name}")
        print(f"  Father: {extraction.candidate.father_name}")
        print(f"  Roll No: {extraction.candidate.roll_number}")
        print(f"  Reg No: {extraction.candidate.registration_number}")
        print(f"  DOB: {extraction.candidate.date_of_birth}")
        
        # Examination Info
        print(f"\n-- Examination --")
        print(f"  Exam: {extraction.examination.exam_name}")
        print(f"  Year: {extraction.examination.exam_year}")
        print(f"  Board/University: {extraction.examination.board_university}")
        print(f"  Institution: {extraction.examination.institution_name}")
        print(f"  Course: {extraction.examination.course_name}")
        
        # Subjects
        print(f"\n-- Subjects ({len(extraction.subjects)}) --")
        for i, subj in enumerate(extraction.subjects, 1):
            subj_name = subj.subject_name.value if subj.subject_name else "Unknown"
            marks = f"{subj.obtained_marks.value}/{subj.max_marks.value}" if subj.obtained_marks and subj.max_marks else "N/A"
            grade = subj.grade.value if subj.grade else "N/A"
            subj_group = f" [{subj.subject_group.value}]" if hasattr(subj, 'subject_group') and subj.subject_group else ""
            print(f"  {i}. {subj_name}{subj_group}: {marks} (Grade: {grade})")
            
            # Show theory/practical/oral breakdown if available
            if subj.theory_marks:
                print(f"      Theory: {subj.theory_marks.value}")
            if subj.practical_marks:
                print(f"      Practical: {subj.practical_marks.value}")
            if hasattr(subj, 'oral_marks') and subj.oral_marks:
                print(f"      Oral: {subj.oral_marks.value}")
            if hasattr(subj, 'written_marks') and subj.written_marks:
                print(f"      Written: {subj.written_marks.value}")
            
            # Show detailed components if available
            if hasattr(subj, 'components') and subj.components:
                print(f"      📋 Detailed Breakdown:")
                for comp in subj.components:
                    comp_name = comp.component_name.value if comp.component_name else "Unknown"
                    comp_marks = f"{comp.obtained_marks.value}/{comp.max_marks.value}" if comp.obtained_marks and comp.max_marks else "N/A"
                    print(f"         - {comp_name}: {comp_marks}")
        
        # Result
        print(f"\n-- Result --")
        print(f"  Status: {extraction.result.result_status}")
        print(f"  Total Marks: {extraction.result.total_marks}")
        print(f"  Max Marks: {extraction.result.max_total_marks}")
        print(f"  Percentage: {extraction.result.percentage}")
        print(f"  Division: {extraction.result.division}")
        
    except Exception as e:
        import traceback
        print(f"Extraction Error: {e}")
        traceback.print_exc()


def test_ocr_samples():
    """Quick OCR-only test on available samples"""
    print("=== OCR-Only Test ===")
    print(f"Extract directory exists: {Path('extract').exists()}")
    
    # Find all images
    samples = list(Path('samples').glob('*'))
    testdata = list(Path('testdata').glob('*'))
    all_files = samples + testdata
    
    image_files = [f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    print(f"Found {len(image_files)} image files")
    
    for img_file in image_files[:3]:  # Test first 3
        test_ocr_only(str(img_file))
    
    # Show saved files
    extract_files = list(Path('extract').glob('*.txt'))
    print(f"\n\n=== Saved OCR Files ({len(extract_files)}) ===")
    for f in extract_files[-5:]:
        print(f"  {f.name}")


async def main():
    for i in [1, 2, 3]:
        await test_full_pipeline(f"testdata/marks_sheet_{i}.webp")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ocr":
        # Run OCR-only test
        test_ocr_samples()
    else:
        # Run full pipeline test
        asyncio.run(main())
