#!/usr/bin/env python
"""Test OCR on PDF files"""

import asyncio
from app.utils.file_processor import file_processor
from app.services.ocr_service import ocr_service


async def test_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    print(f'\n{"="*60}')
    print(f'Testing PDF: {pdf_path}')
    print(f'PDF size: {len(pdf_bytes)} bytes')
    
    # Process PDF to images
    images = await file_processor.process_file(pdf_bytes, pdf_path)
    print(f'Pages extracted: {len(images)}')
    
    total_conf = 0
    for idx, (img_bytes, mime) in enumerate(images):
        print(f'\nPage {idx+1}:')
        result = ocr_service.extract_text(img_bytes, save_text=True)
        print(f'  Confidence: {result["avg_confidence"]:.2%}')
        print(f'  Word Count: {result.get("word_count", 0)}')
        print(f'  High Conf Words: {result.get("high_conf_words", 0)}')
        print(f'  Saved to: {result.get("saved_file", "N/A")}')
        print(f'  Text preview: {result["raw_text"][:200]}...')
        total_conf += result['avg_confidence']
    
    avg_conf = total_conf / len(images) if images else 0
    print(f'\n--- Summary ---')
    print(f'Average OCR Confidence: {avg_conf:.2%}')
    print(f'Would use OCR+LLM (threshold 40%): {avg_conf >= 0.40}')
    return avg_conf


async def main():
    import os
    from pathlib import Path
    
    testdata = Path('testdata')
    pdfs = list(testdata.glob('*.pdf'))
    
    print(f'Found {len(pdfs)} PDF files')
    
    for pdf in pdfs[:3]:  # Test first 3
        await test_pdf(str(pdf))


if __name__ == "__main__":
    asyncio.run(main())
