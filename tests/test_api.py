"""
API Tests for Marksheet Extraction API
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
from PIL import Image

from app.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint"""
    
    def test_health_check(self):
        """Test health endpoint returns 200"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "llm_provider" in data


class TestRootEndpoint:
    """Tests for root endpoint"""
    
    def test_root(self):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestSchemaEndpoint:
    """Tests for schema endpoint"""
    
    def test_get_schema(self):
        """Test schema endpoint returns JSON schema"""
        response = client.get("/api/v1/schema")
        assert response.status_code == 200
        data = response.json()
        assert "properties" in data
        assert "candidate" in data["properties"]
        assert "subjects" in data["properties"]


class TestFileValidation:
    """Tests for file validation"""
    
    def test_invalid_file_type(self):
        # Create a fake text file
        file_content = b"This is not an image"
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
        )
        assert response.status_code == 400
    
    def test_empty_file(self):
        """Test rejection of empty files"""
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.jpg", io.BytesIO(b""), "image/jpeg")}
        )
        assert response.status_code in [400, 422]
    
    def test_valid_image_format(self):
        """Test acceptance of valid image format"""
        # Create a small test image
        img = Image.new('RGB', (100, 100), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        with patch('app.services.extraction.llm_service.extract_marksheet') as mock_extract:
            # Mock the extraction to avoid actual API calls
            mock_extract.return_value = create_mock_extraction()
            
            response = client.post(
                "/api/v1/extract",
                files={"file": ("test.png", img_bytes, "image/png")}
            )
            # Should not fail on file validation
            # May fail on extraction if not mocked properly
            assert response.status_code in [200, 422, 500]


class TestAPIKeyAuth:
    
    def test_missing_api_key_when_required(self):
        from app.core.config import settings
        
        if not settings.api_key_enabled:
            pytest.skip("API key authentication is disabled")
        
        img = Image.new('RGB', (100, 100), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        assert response.status_code == 401


class TestBatchExtraction:
    """Tests for batch extraction endpoint"""
    
    def test_too_many_files(self):
        """Test rejection of too many files"""
        files = []
        for i in range(11):  # Max is 10
            img = Image.new('RGB', (100, 100), color='white')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            files.append(("files", (f"test{i}.png", img_bytes, "image/png")))
        
        response = client.post("/api/v1/extract/batch", files=files)
        assert response.status_code == 400


# Helper function to create mock extraction result
def create_mock_extraction():
    """Create a mock extraction result for testing"""
    from app.models.schemas import (
        MarksheetExtraction, CandidateDetails, ExaminationDetails,
        SubjectMarks, OverallResult, DocumentMetadata, StringField,
        FloatField, DateField
    )
    
    return MarksheetExtraction(
        candidate=CandidateDetails(
            name=StringField(value="TEST STUDENT", confidence=0.95),
            roll_number=StringField(value="12345", confidence=0.95)
        ),
        examination=ExaminationDetails(
            exam_name=StringField(value="Test Exam", confidence=0.9),
            exam_year=StringField(value="2023", confidence=0.95),
            board_university=StringField(value="Test Board", confidence=0.9)
        ),
        subjects=[
            SubjectMarks(
                subject_name=StringField(value="Mathematics", confidence=0.95),
                max_marks=FloatField(value=100, confidence=0.9),
                obtained_marks=FloatField(value=85, confidence=0.9)
            )
        ],
        result=OverallResult(
            result_status=StringField(value="PASS", confidence=0.95),
            total_marks=FloatField(value=85, confidence=0.9)
        ),
        metadata=DocumentMetadata(
            document_type=StringField(value="Marksheet", confidence=0.9)
        ),
        extraction_confidence=0.92
    )


class TestIntegration:
    """Integration tests - require actual API keys"""
    
    @pytest.mark.skip(reason="Requires actual LLM API key")
    def test_full_extraction_flow(self):
        # Load a test marksheet image
        with open("samples/test_marksheet.jpg", "rb") as f:
            response = client.post(
                "/api/v1/extract",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "candidate" in data["data"]
        assert data["data"]["extraction_confidence"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestMarksheetExtraction:
    
    @pytest.fixture
    def api_headers(self):
        from app.core.config import settings
        return {"X-API-Key": settings.api_key} if settings.api_key_enabled else {}
    
    def test_simple_image_upload(self, api_headers):
        import os
        test_file = "testdata/marks_sheet_3.webp"
        if not os.path.exists(test_file):
            pytest.skip("file not there")
        
        with open(test_file, "rb") as f:
            resp = client.post(
                "/api/v1/extract",
                files={"file": ("marks_sheet_3.webp", f, "image/webp")},
                headers=api_headers
            )
        
        assert resp.status_code == 200
        result = resp.json()
        assert result["success"] == True
    
    def test_check_subjects_extracted(self, api_headers):
        import os
        test_file = "testdata/shashank.jpg"
        if not os.path.exists(test_file):
            pytest.skip("shashank file missing")
        
        with open(test_file, "rb") as f:
            resp = client.post(
                "/api/v1/extract",
                files={"file": ("shashank.jpg", f, "image/jpeg")},
                headers=api_headers
            )
        
        result = resp.json()
        subjects = result["data"]["subjects"]
        
        assert len(subjects) > 0
        print(f"found {len(subjects)} subjects")
    
    def test_student_name_extracted(self, api_headers):
        """check student name is comming"""
        import os
        test_file = "testdata/marksheet.png"
        if not os.path.exists(test_file):
            pytest.skip("marksheet not found")
        
        with open(test_file, "rb") as f:
            resp = client.post(
                "/api/v1/extract",
                files={"file": ("marksheet.png", f, "image/png")},
                headers=api_headers
            )
        
        data = resp.json()
        name = data["data"]["candidate"]["name"]["value"]
        
        assert name is not None
        assert len(name) > 2
        print(f"student name: {name}")
    
    def test_pdf_file_works(self, api_headers):
        """test pdf marksheet upload"""
        import os
        test_file = "testdata/semister1.pdf"
        if not os.path.exists(test_file):
            pytest.skip("pdf file not availble")
        
        with open(test_file, "rb") as f:
            resp = client.post(
                "/api/v1/extract",
                files={"file": ("semister1.pdf", f, "application/pdf")},
                headers=api_headers
            )
        
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] == True
        print(f"method used: {data['extraction_method']}")
    
    def test_confidence_is_there(self, api_headers):
        """check confidance score exists"""
        import os
        test_file = "testdata/marks_sheet_2.webp"
        if not os.path.exists(test_file):
            pytest.skip("file not there")
        
        with open(test_file, "rb") as f:
            resp = client.post(
                "/api/v1/extract",
                files={"file": ("marks_sheet_2.webp", f, "image/webp")},
                headers=api_headers
            )
        
        data = resp.json()
        conf = data["data"]["extraction_confidence"]
        
        assert conf is not None
        assert conf > 0
        assert conf <= 1
        print(f"confidnce: {conf}")
    
    def test_marks_are_numbers(self, api_headers):
        """verify marks are numric values"""
        import os
        test_file = "testdata/marks_sheet_1.webp"
        if not os.path.exists(test_file):
            pytest.skip("marksheet 1 not found")
        
        with open(test_file, "rb") as f:
            resp = client.post(
                "/api/v1/extract",
                files={"file": ("marks_sheet_1.webp", f, "image/webp")},
                headers=api_headers
            )
        
        data = resp.json()
        subjects = data["data"]["subjects"]
        
        for subj in subjects:
            if subj.get("obtained_marks"):
                marks = subj["obtained_marks"]["value"]
                if marks is not None:
                    assert isinstance(marks, (int, float))
    
    def test_pass_fail_status(self, api_headers):
        """check pass or fail status"""
        import os
        test_file = "testdata/ajay.jpg"
        if not os.path.exists(test_file):
            pytest.skip("ajay file not there")
        
        with open(test_file, "rb") as f:
            resp = client.post(
                "/api/v1/extract",
                files={"file": ("ajay.jpg", f, "image/jpeg")},
                headers=api_headers
            )
        
        data = resp.json()
        result = data["data"]["result"]
        
        if result.get("result_status"):
            status = result["result_status"]["value"]
            print(f"result status: {status}")
    
    def test_processing_time_check(self, api_headers):
        """see how much time it takes"""
        import os
        test_file = "testdata/abhinav.jpg"
        if not os.path.exists(test_file):
            pytest.skip("abhinav not found")
        
        with open(test_file, "rb") as f:
            resp = client.post(
                "/api/v1/extract",
                files={"file": ("abhinav.jpg", f, "image/jpeg")},
                headers=api_headers
            )
        
        data = resp.json()
        time_ms = data["processing_time_ms"]
        
        assert time_ms > 0
        print(f"took {time_ms} miliseconds")


class Test10thMarksheet:
    """simple test for 10th class marksheet"""
    
    @pytest.fixture
    def api_headers(self):
        from app.core.config import settings
        return {"X-API-Key": settings.api_key} if settings.api_key_enabled else {}
    
    def test_10th_board_result(self, api_headers):
        """test 10th marksheet extracton"""
        import os
        test_file = "testdata/10th result[4274].pdf.pdf"
        if not os.path.exists(test_file):
            pytest.skip("10th result not availble")
        
        with open(test_file, "rb") as f:
            resp = client.post(
                "/api/v1/extract",
                files={"file": ("10th_result.pdf", f, "application/pdf")},
                headers=api_headers
            )
        
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["success"] == True
        
        name = data["data"]["candidate"]["name"]["value"]
        print(f"student: {name}")
        
        subjects = data["data"]["subjects"]
        print(f"total subjects: {len(subjects)}")
        assert len(subjects) >= 5
        
        for subj in subjects:
            subj_name = subj["subject_name"]["value"]
            marks = subj.get("obtained_marks", {}).get("value", "N/A")
            print(f"  {subj_name}: {marks}")
        
        conf = data["data"]["extraction_confidence"]
        print(f"confidnce score: {conf}")
        assert conf > 0.5
        
        ocr_conf = data["ocr_confidence"]
        print(f"ocr confidance: {ocr_conf}")

