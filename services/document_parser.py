"""
Document Parser Service
Integrates with LandingAI DPT-2 API for intelligent document extraction.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ExtractedField:
    """Represents an extracted field with its value, confidence, and bounding boxes."""
    value: str
    confidence: float
    field_name: str
    bounding_boxes: List[Dict[str, float]] = field(default_factory=list)  # [{page, l, t, r, b}, ...]


@dataclass  
class ParsedDocument:
    """Represents a fully parsed document with all extracted data."""
    file_hash: str
    original_filename: str
    document_type: str
    markdown_content: str
    extracted_fields: Dict[str, ExtractedField]
    raw_chunks: List[Dict[str, Any]]
    overall_confidence: float
    needs_verification: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "file_hash": self.file_hash,
            "original_filename": self.original_filename,
            "document_type": self.document_type,
            "markdown_content": self.markdown_content,
            "extracted_fields": {
                k: {
                    "value": v.value,
                    "confidence": v.confidence,
                    "field_name": v.field_name,
                    "bounding_boxes": v.bounding_boxes
                }
                for k, v in self.extracted_fields.items()
            },
            "raw_chunks": self.raw_chunks,
            "overall_confidence": self.overall_confidence,
            "needs_verification": self.needs_verification,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedDocument":
        """Reconstruct from dictionary."""
        extracted_fields = {
            k: ExtractedField(
                value=v["value"],
                confidence=v["confidence"],
                field_name=v["field_name"],
                bounding_boxes=v.get("bounding_boxes", [])
            )
            for k, v in data.get("extracted_fields", {}).items()
        }
        return cls(
            file_hash=data["file_hash"],
            original_filename=data["original_filename"],
            document_type=data["document_type"],
            markdown_content=data["markdown_content"],
            extracted_fields=extracted_fields,
            raw_chunks=data.get("raw_chunks", []),
            overall_confidence=data["overall_confidence"],
            needs_verification=data["needs_verification"],
        )


def compute_file_hash(file_path: str) -> str:
    """
    Compute MD5 hash of a file for deduplication.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hex digest string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def detect_document_type(markdown: str, fields: Dict) -> str:
    """
    Detect the type of tax document based on extracted content.
    """
    markdown_lower = markdown.lower()
    
    # Check for common Canadian tax form indicators
    if "notice of assessment" in markdown_lower or "avis de cotisation" in markdown_lower:
        return "Notice of Assessment (NOA)"
    elif "t1 general" in markdown_lower or ("t1" in markdown_lower and "return" in markdown_lower):
        return "T1 General - Income Tax Return"
    elif "t4" in markdown_lower or "statement of remuneration" in markdown_lower:
        return "T4 - Statement of Remuneration Paid"
    elif "t2202" in markdown_lower or "tuition" in markdown_lower:
        return "T2202 - Tuition and Enrolment Certificate"
    elif "rrsp" in markdown_lower or "registered retirement" in markdown_lower:
        return "RRSP Contribution Receipt"
    elif "fhsa" in markdown_lower or "first home savings" in markdown_lower:
        return "FHSA Contribution Receipt"
    elif "rent" in markdown_lower and "receipt" in markdown_lower:
        return "Rent Receipt"
    elif "moving" in markdown_lower and ("expense" in markdown_lower or "receipt" in markdown_lower):
        return "Moving Expense Receipt"
    elif "t5" in markdown_lower or "investment income" in markdown_lower:
        return "T5 - Statement of Investment Income"
    elif "t3" in markdown_lower:
        return "T3 - Statement of Trust Income"
    else:
        return "Tax Document"


def parse_document_sync(file_path: str, original_filename: str) -> ParsedDocument:
    """
    Parse a document using LandingAI DPT-2 API (synchronous version).
    
    Args:
        file_path: Path to the document file
        original_filename: Original name of the uploaded file
        
    Returns:
        ParsedDocument with extracted data and confidence scores
    """
    from agentic_doc.parse import parse
    from pydantic import BaseModel, Field
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Compute file hash for deduplication
    file_hash = compute_file_hash(file_path)
    
    # Define extraction schema for common tax document fields
    class TaxDocumentFields(BaseModel):
        """Schema for extracting tax document fields."""
        document_title: Optional[str] = Field(None, description="The title or type of the document")
        tax_year: Optional[str] = Field(None, description="The tax year this document is for")
        
        # Personal identification
        employee_name: Optional[str] = Field(None, description="Employee's full name (for T4)")
        student_name: Optional[str] = Field(None, description="Student's full name (for T2202)")
        taxpayer_name: Optional[str] = Field(None, description="Taxpayer's or contributor's full name")
        tenant_name: Optional[str] = Field(None, description="Tenant's full name (for rent receipts)")
        mailing_address: Optional[str] = Field(None, description="Taxpayer's mailing address including street, city, province, postal code")
        
        # Income fields
        total_income: Optional[str] = Field(None, description="Total income amount")
        employment_income: Optional[str] = Field(None, description="Employment income from T4")
        other_income: Optional[str] = Field(None, description="Other income amounts")
        
        # Deduction fields  
        income_tax_deducted: Optional[str] = Field(None, description="Income tax already deducted")
        cpp_contributions: Optional[str] = Field(None, description="CPP contributions")
        ei_premiums: Optional[str] = Field(None, description="EI premiums")
        
        # RRSP/FHSA
        rrsp_contribution: Optional[str] = Field(None, description="RRSP contribution amount")
        rrsp_issuer: Optional[str] = Field(None, description="RRSP issuer/financial institution name")
        fhsa_contribution: Optional[str] = Field(None, description="FHSA contribution amount")
        fhsa_issuer: Optional[str] = Field(None, description="FHSA issuer/financial institution name")
        contribution_date: Optional[str] = Field(None, description="Date of RRSP/FHSA contribution")
        
        # Tuition
        tuition_amount: Optional[str] = Field(None, description="Tuition fees paid")
        education_institution: Optional[str] = Field(None, description="Name of educational institution")
        student_number: Optional[str] = Field(None, description="Student ID number")
        
        # Rent
        rent_amount: Optional[str] = Field(None, description="Total rent paid")
        landlord_name: Optional[str] = Field(None, description="Landlord name")
        rental_address: Optional[str] = Field(None, description="Rental property address")
        rent_period_start: Optional[str] = Field(None, description="Rent period start date")
        rent_period_end: Optional[str] = Field(None, description="Rent period end date")
        
        # Donations (T4A, charitable receipts)
        donation_amount: Optional[str] = Field(None, description="Charitable donation amount")
        charity_name: Optional[str] = Field(None, description="Charity or organization name")
        donation_date: Optional[str] = Field(None, description="Date of donation")
        
        # Medical expenses
        medical_expenses: Optional[str] = Field(None, description="Total medical expenses")
        
        # === Notice of Assessment (NOA) / T1 Fields ===
        # Income Summary
        net_income: Optional[str] = Field(None, description="Net income (line 23600)")
        taxable_income: Optional[str] = Field(None, description="Taxable income (line 26000)")
        
        # Tax Results
        total_federal_tax: Optional[str] = Field(None, description="Total federal tax payable")
        total_provincial_tax: Optional[str] = Field(None, description="Total provincial tax payable")
        tax_owing: Optional[str] = Field(None, description="Balance owing or amount to pay")
        refund_amount: Optional[str] = Field(None, description="Refund amount if applicable")
        
        # RRSP Room (from NOA)
        rrsp_deduction_limit: Optional[str] = Field(None, description="RRSP deduction limit for next year")
        rrsp_unused_contributions: Optional[str] = Field(None, description="Unused RRSP contributions available to deduct")
        rrsp_carry_forward: Optional[str] = Field(None, description="RRSP unused room carried forward")
        
        # FHSA Room (from NOA)
        fhsa_contribution_room: Optional[str] = Field(None, description="FHSA contribution room for next year")
        fhsa_unused_room: Optional[str] = Field(None, description="FHSA unused room carried forward")
        
        # Other Carry-Forwards
        capital_loss_carry_forward: Optional[str] = Field(None, description="Net capital losses available to carry forward")
        tuition_carry_forward: Optional[str] = Field(None, description="Unused tuition credits carried forward")
        
        # Assessment Date
        assessment_date: Optional[str] = Field(None, description="Date of assessment or reassessment")
        assessment_year: Optional[str] = Field(None, description="Tax year being assessed")
        
        # Common identifiers
        employer_name: Optional[str] = Field(None, description="Employer or payer name")
        sin_last_three: Optional[str] = Field(None, description="Last 3 digits of SIN if visible")
        
        # Additional observations
        additional_notes: Optional[str] = Field(
            None, 
            description="Any additional observations, anomalies, or important notes about the document that may be useful for tax filing. Examples: handwritten annotations, amendments, unusual formatting, missing sections, or potential discrepancies."
        )
        
    # Call LandingAI API
    # Use DPT-2 mini for lower cost (preview), or full DPT-2 for production
    model_name = os.getenv("LANDINGAI_MODEL", "dpt-2-mini-latest")
    
    try:
        logger.info(f"Parsing document: {file_path} (model: {model_name})")
        results = parse(file_path, extraction_model=TaxDocumentFields)
        
        if not results or len(results) == 0:
            raise ValueError("No results returned from document parsing")
            
        result = results[0]
        logger.info(f"Parse result attributes: {dir(result)}")
        
        # Get markdown content
        markdown_content = result.markdown if hasattr(result, 'markdown') else ""
        logger.info(f"Markdown length: {len(markdown_content)} chars")
        
        # Build chunk lookup by chunk_id for bounding box retrieval
        chunk_lookup = {}
        raw_chunks = []
        if hasattr(result, 'chunks') and result.chunks:
            for chunk in result.chunks:
                chunk_id = getattr(chunk, 'chunk_id', None)
                if chunk_id:
                    chunk_lookup[chunk_id] = chunk
                
                # Extract bounding box info for raw_chunks
                bboxes = []
                if hasattr(chunk, 'grounding') and chunk.grounding:
                    for g in chunk.grounding:
                        if hasattr(g, 'box') and g.box:
                            bboxes.append({
                                "page": g.page,
                                "l": g.box.l,
                                "t": g.box.t,
                                "r": g.box.r,
                                "b": g.box.b
                            })
                
                chunk_data = {
                    "text": chunk.text if hasattr(chunk, 'text') else str(chunk),
                    "chunk_type": chunk.chunk_type if hasattr(chunk, 'chunk_type') else "unknown",
                    "chunk_id": chunk_id,
                    "bounding_boxes": bboxes
                }
                raw_chunks.append(chunk_data)
        logger.info(f"Raw chunks count: {len(raw_chunks)}, chunk lookup size: {len(chunk_lookup)}")
        
        # Process extracted fields with confidence
        extracted_fields = {}
        overall_confidences = []
        
        # Check if extraction exists
        has_extraction = hasattr(result, 'extraction') and result.extraction is not None
        logger.info(f"Has extraction: {has_extraction}")
        
        if has_extraction:
            extraction = result.extraction
            metadata = result.extraction_metadata if hasattr(result, 'extraction_metadata') else None
            logger.info(f"Extraction type: {type(extraction)}")
            logger.info(f"Extraction metadata: {metadata}")
            
            # Try to get fields from extraction
            if hasattr(extraction, 'model_fields'):
                for field_name in extraction.model_fields.keys():
                    value = getattr(extraction, field_name, None)
                    if value is not None and str(value).strip():
                        # Get confidence from metadata if available
                        confidence = 0.95  # Default high confidence
                        bounding_boxes = []  # Collect bounding boxes for this field
                        
                        if metadata and hasattr(metadata, field_name):
                            field_meta = getattr(metadata, field_name, None)
                            if field_meta:
                                # Get confidence
                                if hasattr(field_meta, 'confidence'):
                                    meta_confidence = field_meta.confidence
                                    if meta_confidence is not None:
                                        confidence = meta_confidence
                                
                                # Get bounding boxes from chunk_references
                                if hasattr(field_meta, 'chunk_references') and field_meta.chunk_references:
                                    for ref_id in field_meta.chunk_references:
                                        if ref_id in chunk_lookup:
                                            chunk = chunk_lookup[ref_id]
                                            if hasattr(chunk, 'grounding') and chunk.grounding:
                                                for g in chunk.grounding:
                                                    if hasattr(g, 'box') and g.box:
                                                        bounding_boxes.append({
                                                            "page": g.page,
                                                            "l": g.box.l,
                                                            "t": g.box.t,
                                                            "r": g.box.r,
                                                            "b": g.box.b
                                                        })
                        
                        extracted_fields[field_name] = ExtractedField(
                            value=str(value),
                            confidence=confidence,
                            field_name=field_name,
                            bounding_boxes=bounding_boxes
                        )
                        overall_confidences.append(confidence)
                        logger.info(f"Extracted field: {field_name} = {value} (conf: {confidence}, bboxes: {len(bounding_boxes)})")
            elif hasattr(extraction, '__dict__'):
                # Try direct attribute access
                for field_name, value in extraction.__dict__.items():
                    if value is not None and not field_name.startswith('_'):
                        extracted_fields[field_name] = ExtractedField(
                            value=str(value),
                            confidence=0.90,
                            field_name=field_name,
                            bounding_boxes=[]
                        )
                        overall_confidences.append(0.90)
                        logger.info(f"Extracted field (dict): {field_name} = {value}")
        
        logger.info(f"Total extracted fields: {len(extracted_fields)}")
        
        # Calculate overall confidence - filter out any None values just in case
        valid_confidences = [c for c in overall_confidences if c is not None]
        if valid_confidences:
            overall_confidence = sum(valid_confidences) / len(valid_confidences)
        elif markdown_content:
            # If we got markdown but no fields, still give some confidence
            overall_confidence = 0.75
        else:
            overall_confidence = 0.5
            
        # Detect document type
        document_type = detect_document_type(markdown_content, extracted_fields)
        
        # Determine if verification is needed (any field below 85% confidence)
        needs_verification = any(
            f.confidence < 0.85 for f in extracted_fields.values()
        ) or overall_confidence < 0.85 or len(extracted_fields) == 0
        
        return ParsedDocument(
            file_hash=file_hash,
            original_filename=original_filename,
            document_type=document_type,
            markdown_content=markdown_content,
            extracted_fields=extracted_fields,
            raw_chunks=raw_chunks,
            overall_confidence=overall_confidence,
            needs_verification=needs_verification,
        )
        
    except Exception as e:
        logger.error(f"Error parsing document: {str(e)}", exc_info=True)
        # Return a document with error state
        return ParsedDocument(
            file_hash=file_hash,
            original_filename=original_filename,
            document_type="Unknown (Parse Error)",
            markdown_content=f"Error parsing document: {str(e)}",
            extracted_fields={},
            raw_chunks=[],
            overall_confidence=0.0,
            needs_verification=True,
        )


# Keep async wrapper for compatibility
async def parse_document(file_path: str, original_filename: str) -> ParsedDocument:
    """Async wrapper for parse_document_sync."""
    return parse_document_sync(file_path, original_filename)

