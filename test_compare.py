import asyncio
import sys
sys.path.insert(0, '/Users/souradipbiswas/Downloads/python_api_extract')

from app.services.ocr_service import ocr_service
from app.services.extraction import llm_service


async def compare_methods(img_path):
    print(f"\n{'='*70}")
    print(f"Comparing: {img_path}")
    print('='*70)
    
    with open(img_path, 'rb') as f:
        image_bytes = f.read()
    
    ext = img_path.split('.')[-1].lower()
    mime_map = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp'}
    mime_type = mime_map.get(ext, 'image/jpeg')
    
    await llm_service.initialize()
    
    # m 1: OCR + LLM
    print("\n--- Method 1: OCR + LLM ---")
    ocr_result = ocr_service.extract_text(image_bytes)
    print(f"OCR confidence: {ocr_result['avg_confidence']:.3f}")
    
    try:
        ocr_results = [{
            "page": 1,
            "text": ocr_result['raw_text'],
            "avg_confidence": ocr_result['avg_confidence'],
            "words": ocr_result['words']
        }]
        extraction1 = await llm_service.extract_marksheet_from_ocr(ocr_results)
        print(f"Name: {extraction1.candidate.name.value if extraction1.candidate.name else 'N/A'}")
        print(f"Roll: {extraction1.candidate.roll_number.value if extraction1.candidate.roll_number else 'N/A'}")
        print(f"Institution: {extraction1.examination.institution_name.value if extraction1.examination.institution_name else 'N/A'}")
        print(f"Subjects: {len(extraction1.subjects)}")
        for s in extraction1.subjects[:3]:
            name = s.subject_name.value if s.subject_name else "?"
            marks = s.obtained_marks.value if s.obtained_marks else "?"
            print(f"  - {name}: {marks}")
    except Exception as e:
        print(f"Error: {e}")
    
    # m 2: Direct LLM Vision
    print("\n--- Method 2: Direct LLM Vision ---")
    try:
        images = [(image_bytes, mime_type)]
        extraction2 = await llm_service.extract_marksheet(images)
        print(f"Name: {extraction2.candidate.name.value if extraction2.candidate.name else 'N/A'}")
        print(f"Roll: {extraction2.candidate.roll_number.value if extraction2.candidate.roll_number else 'N/A'}")
        print(f"Institution: {extraction2.examination.institution_name.value if extraction2.examination.institution_name else 'N/A'}")
        print(f"Subjects: {len(extraction2.subjects)}")
        for s in extraction2.subjects[:3]:
            name = s.subject_name.value if s.subject_name else "?"
            marks = s.obtained_marks.value if s.obtained_marks else "?"
            print(f"  - {name}: {marks}")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    for i in [1, 2, 3]:
        await compare_methods(f"testdata/marks_sheet_{i}.webp")

if __name__ == "__main__":
    asyncio.run(main())
