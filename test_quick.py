"""Quick test script for OCR + LLM extraction"""
import asyncio
import time
import sys
from app.services.ocr_service import ocr_service
from app.services.extraction import llm_service

async def test(filename='testdata/marksheet.png'):
    start = time.time()
    
    # Test with a sample image
    with open(filename, 'rb') as f:
        image_bytes = f.read()
    
    # Get OCR result
    print(f"Testing: {filename}")
    print("Running OCR...")
    ocr_start = time.time()
    ocr_result = ocr_service.extract_text(image_bytes, save_text=True)
    ocr_time = time.time() - ocr_start
    
    print(f'OCR Time: {ocr_time:.2f}s')
    print(f'OCR Confidence: {ocr_result.get("avg_confidence", 0)}')
    print(f'OCR Words: {ocr_result.get("word_count", 0)}')
    
    # Create OCR results list
    ocr_results = [{
        'page': 1,
        'text': ocr_result['raw_text'],
        'avg_confidence': ocr_result['avg_confidence']
    }]
    
    print('\nCalling LLM extraction (this takes 15-30 seconds)...')
    llm_start = time.time()
    
    try:
        extraction = await llm_service.extract_marksheet_ocr(ocr_results)
        llm_time = time.time() - llm_start
        
        print(f'\nLLM Time: {llm_time:.2f}s')
        print('\n=== EXTRACTION RESULT ===')
        print(f'Candidate Name: {extraction.candidate.name}')
        print(f'Roll Number: {extraction.candidate.roll_number}')
        print(f'Exam: {extraction.examination.exam_name}')
        print(f'Board: {extraction.examination.board_university}')
        print(f'Subjects: {len(extraction.subjects)}')
        for subj in extraction.subjects[:3]:
            print(f'  - {subj.subject_name}: {subj.obtained_marks}/{subj.max_marks}')
        print(f'Result: {extraction.result.result_status}')
        print(f'Confidence: {extraction.extraction_confidence}')
        print(f'\nTotal Time: {time.time() - start:.2f}s')
        
    except Exception as e:
        import traceback
        print(f'Error: {e}')
        traceback.print_exc()

if __name__ == "__main__":
    # Accept filename as argument or use default
    filename = sys.argv[1] if len(sys.argv) > 1 else 'testdata/marksheet.png'
    asyncio.run(test(filename))
