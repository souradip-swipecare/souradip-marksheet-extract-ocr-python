import asyncio
import sys
sys.path.insert(0, '/Users/souradipbiswas/Downloads/python_api_extract')

from app.services.extraction import llm_service

async def test_single():
    img_path = "testdata/marks_sheet_3.webp"
    
    with open(img_path, 'rb') as f:
        image_bytes = f.read()
    
    await llm_service.initialize()
    
    # Direct LLM Vision
    images = [(image_bytes, "image/webp")]
    extraction = await llm_service.extract_marksheet(images)
    
    print("=== MARKSHEET 3 EXTRACTION ===")
    print(f"\nCandidate: {extraction.candidate.name.value if extraction.candidate.name else 'N/A'}")
    print(f"Institution: {extraction.examination.institution_name.value if extraction.examination.institution_name else 'N/A'}")
    
    print(f"\n-- Subjects ({len(extraction.subjects)}) --")
    for i, subj in enumerate(extraction.subjects, 1):
        name = subj.subject_name.value if subj.subject_name else "Unknown"
        marks = f"{subj.obtained_marks.value}/{subj.max_marks.value}" if subj.obtained_marks and subj.max_marks else "N/A"
        group = f" [{subj.subject_group.value}]" if subj.subject_group else ""
        print(f"\n{i}. {name}{group}: {marks}")
        
        # Show components if available
        if subj.components:
            print("   📋 Detailed Breakdown:")
            for comp in subj.components:
                cn = comp.component_name.value if comp.component_name else "?"
                cm = f"{comp.obtained_marks.value}/{comp.max_marks.value}" if comp.obtained_marks and comp.max_marks else "N/A"
                print(f"      - {cn}: {cm}")
    
    print(f"\n-- Result --")
    print(f"Status: {extraction.result.result_status.value if extraction.result.result_status else 'Unknown'}")
    print(f"Total: {extraction.result.total_marks.value if extraction.result.total_marks else 'N/A'}/{extraction.result.max_total_marks.value if extraction.result.max_total_marks else 'N/A'}")
    print(f"Percentage: {extraction.result.percentage.value if extraction.result.percentage else 'N/A'}%")
    print(f"Division: {extraction.result.division.value if extraction.result.division else 'N/A'}")
    
    print(f"\n-- Confidence --")
    print(f"Overall Extraction Confidence: {extraction.extraction_confidence:.1%}" if extraction.extraction_confidence else "N/A")

if __name__ == "__main__":
    asyncio.run(test_single())
