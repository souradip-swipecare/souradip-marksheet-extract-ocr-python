#!/usr/bin/env python
"""Test smart PDF processing - text-based vs image-based detection"""

import asyncio
from pathlib import Path

from app.utils.file_processor import file_processor
from app.services.extraction import llm_service


async def test_smart_pdf():
    """Test smart PDF processing"""
    
    print("=" * 60)
    print("SMART PDF PROCESSING TEST")
    print("=" * 60)
    
    # Find all PDFs
    pdfs = list(Path('testdata').glob('*.pdf')) + list(Path('samples').glob('*.pdf'))
    
    for pdf_path in pdfs:
        print(f"\n{'='*60}")
        print(f"Testing: {pdf_path.name}")
        print("=" * 60)
        
        with open(pdf_path, 'rb') as f:
            content = f.read()
        
        # Smart process
        result = await file_processor.process_file_smart(content, pdf_path.name)
        
        print(f"\nFile Type: {result['type']}")
        
        if result['type'] == 'text_pdf':
            print("✅ TEXT-BASED PDF - No OCR needed! (Most efficient)")
            print(f"   Pages: {len(result['text_content'])}")
            print(f"   Total chars: {result['pdf_info']['total_chars']}")
            
            # Show text preview
            for i, text in enumerate(result['text_content']):
                preview = text[:400].replace('\n', ' ')
                print(f"\n   Page {i+1} preview:\n   {preview}...")
            
            # Test LLM extraction
            print("\n   --- Testing LLM Extraction ---")
            await llm_service.initialize()
            
            ocr_results = []
            for idx, text in enumerate(result['text_content']):
                ocr_results.append({
                    'page': idx + 1,
                    'text': text,
                    'avg_confidence': 0.99,
                    'words': []
                })
            
            try:
                extraction = await llm_service.extract_marksheet_ocr(ocr_results)
                print(f"\n   Extraction Results:")
                print(f"   - Name: {extraction.candidate.name.value}")
                print(f"   - Roll: {extraction.candidate.roll_number.value}")
                print(f"   - Board: {extraction.examination.board_university.value}")
                print(f"   - Subjects: {len(extraction.subjects)}")
                for subj in extraction.subjects[:3]:
                    marks = f"{subj.obtained_marks.value}/{subj.max_marks.value}" if subj.obtained_marks and subj.max_marks else "N/A"
                    print(f"     • {subj.subject_name.value}: {marks}")
                if len(extraction.subjects) > 3:
                    print(f"     ... and {len(extraction.subjects) - 3} more")
                print(f"   - Confidence: {extraction.extraction_confidence:.1%}")
            except Exception as e:
                print(f"   ❌ Extraction failed: {e}")
                
        else:
            print("📷 IMAGE-BASED/SCANNED PDF - Needs OCR")
            print(f"   Pages to OCR: {len(result['images'])}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_smart_pdf())
