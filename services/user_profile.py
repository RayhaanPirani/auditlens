"""
User Profile Service
Manages user profile storage and verification against extracted document data.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from difflib import SequenceMatcher
import logging

from services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

# Collection name for user profiles
PROFILE_COLLECTION = "user_profiles"

# Default profile ID (for single-user mode)
DEFAULT_PROFILE_ID = "default_user"


def similarity_ratio(str1: str, str2: str) -> float:
    """
    Calculate similarity ratio between two strings using SequenceMatcher.
    Returns a value between 0 and 1.
    """
    if not str1 or not str2:
        return 0.0
    
    # Normalize strings for comparison
    str1_norm = str1.lower().strip()
    str2_norm = str2.lower().strip()
    
    return SequenceMatcher(None, str1_norm, str2_norm).ratio()


class UserProfileService:
    """Service for managing user profiles and document verification."""
    
    def __init__(self):
        self.vector_store = VectorStoreService()
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the profile collection exists."""
        try:
            self.vector_store.get_collection(PROFILE_COLLECTION)
        except Exception:
            # Collection will be created on first insert
            pass
    
    def get_profile(self, profile_id: str = DEFAULT_PROFILE_ID) -> Optional[Dict[str, Any]]:
        """
        Get an existing user profile.
        
        Returns:
            Profile dict or None if not found
        """
        try:
            collection = self.vector_store.get_collection(PROFILE_COLLECTION)
            results = collection.get(
                ids=[profile_id],
                include=["documents", "metadatas"]
            )
            
            if not results["ids"]:
                return None
            
            metadata = results["metadatas"][0]
            
            return {
                "profile_id": profile_id,
                "taxpayer_name": metadata.get("taxpayer_name", ""),
                "mailing_address": metadata.get("mailing_address", ""),
                "sin_last_three": metadata.get("sin_last_three", ""),
                "created_at": metadata.get("created_at", ""),
                "last_updated": metadata.get("last_updated", ""),
                "document_count": metadata.get("document_count", 0),
                "verified_fields": json.loads(metadata.get("verified_fields_json", "{}")),
            }
        except Exception as e:
            logger.warning(f"Error getting profile: {e}")
            return None
    
    def create_profile(
        self,
        extracted_fields: Dict[str, Any],
        source_document: str,
        profile_id: str = DEFAULT_PROFILE_ID
    ) -> Dict[str, Any]:
        """
        Create a new user profile from extracted document fields.
        
        Args:
            extracted_fields: Fields extracted from the first document
            source_document: Filename of the source document
            profile_id: Profile identifier
            
        Returns:
            Created profile dict
        """
        now = datetime.now().isoformat()
        
        # Extract identity fields from the document
        taxpayer_name = self._extract_name(extracted_fields)
        mailing_address = self._extract_address(extracted_fields)
        sin_last_three = self._extract_sin(extracted_fields)
        
        # Build verified fields tracking
        verified_fields = {}
        if taxpayer_name:
            verified_fields["name"] = {
                "value": taxpayer_name,
                "confidence": self._get_confidence(extracted_fields, "name"),
                "source_doc": source_document
            }
        if mailing_address:
            verified_fields["address"] = {
                "value": mailing_address,
                "confidence": self._get_confidence(extracted_fields, "address"),
                "source_doc": source_document
            }
        if sin_last_three:
            verified_fields["sin_last_three"] = {
                "value": sin_last_three,
                "confidence": self._get_confidence(extracted_fields, "sin"),
                "source_doc": source_document
            }
        
        # Store in ChromaDB
        metadata = {
            "taxpayer_name": taxpayer_name,
            "mailing_address": mailing_address,
            "sin_last_three": sin_last_three,
            "created_at": now,
            "last_updated": now,
            "document_count": 1,
            "verified_fields_json": json.dumps(verified_fields),
        }
        
        collection = self.vector_store.get_collection(PROFILE_COLLECTION)
        collection.upsert(
            ids=[profile_id],
            documents=[f"Profile for {taxpayer_name}"],
            metadatas=[metadata]
        )
        
        logger.info(f"Created profile for {taxpayer_name}")
        
        return {
            "profile_id": profile_id,
            **metadata,
            "verified_fields": verified_fields,
        }
    
    def update_profile(
        self,
        extracted_fields: Dict[str, Any],
        source_document: str,
        profile_id: str = DEFAULT_PROFILE_ID
    ) -> Dict[str, Any]:
        """
        Update an existing profile with new document data.
        
        Args:
            extracted_fields: Fields from new document
            source_document: Filename of the new document
            profile_id: Profile identifier
            
        Returns:
            Updated profile dict
        """
        profile = self.get_profile(profile_id)
        if not profile:
            return self.create_profile(extracted_fields, source_document, profile_id)
        
        now = datetime.now().isoformat()
        verified_fields = profile.get("verified_fields", {})
        
        # Update with any new higher-confidence fields
        new_name = self._extract_name(extracted_fields)
        new_address = self._extract_address(extracted_fields)
        new_sin = self._extract_sin(extracted_fields)
        
        # Update name if new one has higher confidence or existing is empty
        if new_name:
            new_conf = self._get_confidence(extracted_fields, "name")
            old_conf = verified_fields.get("name", {}).get("confidence", 0)
            if new_conf > old_conf or not profile.get("taxpayer_name"):
                verified_fields["name"] = {
                    "value": new_name,
                    "confidence": new_conf,
                    "source_doc": source_document
                }
                profile["taxpayer_name"] = new_name
        
        # Update address similarly
        if new_address:
            new_conf = self._get_confidence(extracted_fields, "address")
            old_conf = verified_fields.get("address", {}).get("confidence", 0)
            if new_conf > old_conf or not profile.get("mailing_address"):
                verified_fields["address"] = {
                    "value": new_address,
                    "confidence": new_conf,
                    "source_doc": source_document
                }
                profile["mailing_address"] = new_address
        
        # Update SIN (exact match preferred)
        if new_sin and not profile.get("sin_last_three"):
            verified_fields["sin_last_three"] = {
                "value": new_sin,
                "confidence": self._get_confidence(extracted_fields, "sin"),
                "source_doc": source_document
            }
            profile["sin_last_three"] = new_sin
        
        # Update metadata
        metadata = {
            "taxpayer_name": profile.get("taxpayer_name", ""),
            "mailing_address": profile.get("mailing_address", ""),
            "sin_last_three": profile.get("sin_last_three", ""),
            "created_at": profile.get("created_at", now),
            "last_updated": now,
            "document_count": profile.get("document_count", 0) + 1,
            "verified_fields_json": json.dumps(verified_fields),
        }
        
        collection = self.vector_store.get_collection(PROFILE_COLLECTION)
        collection.upsert(
            ids=[profile_id],
            documents=[f"Profile for {profile.get('taxpayer_name', 'Unknown')}"],
            metadatas=[metadata]
        )
        
        logger.info(f"Updated profile, document count: {metadata['document_count']}")
        
        return {
            "profile_id": profile_id,
            **metadata,
            "verified_fields": verified_fields,
        }
    
    def verify_document(
        self,
        extracted_fields: Dict[str, Any],
        source_document: str,
        profile_id: str = DEFAULT_PROFILE_ID
    ) -> Dict[str, Any]:
        """
        Verify a new document against the existing profile.
        
        Args:
            extracted_fields: Fields from the new document
            source_document: Filename of the document
            profile_id: Profile identifier
            
        Returns:
            Verification result dict with status and discrepancies
        """
        profile = self.get_profile(profile_id)
        
        # If no profile exists, create one
        if not profile:
            new_profile = self.create_profile(extracted_fields, source_document, profile_id)
            return {
                "status": "profile_created",
                "message": "🆕 User profile created from first document",
                "profile": new_profile,
                "discrepancies": [],
                "is_first_document": True
            }
        
        # Compare fields
        discrepancies = []
        
        # Extract fields from new document
        new_name = self._extract_name(extracted_fields)
        new_address = self._extract_address(extracted_fields)
        new_sin = self._extract_sin(extracted_fields)
        
        # Check name match
        if new_name and profile.get("taxpayer_name"):
            name_similarity = similarity_ratio(new_name, profile["taxpayer_name"])
            if name_similarity < 0.8:
                discrepancies.append({
                    "field": "name",
                    "expected": profile["taxpayer_name"],
                    "found": new_name,
                    "similarity": name_similarity,
                    "threshold": 0.8
                })
        
        # Check address match
        if new_address and profile.get("mailing_address"):
            addr_similarity = similarity_ratio(new_address, profile["mailing_address"])
            if addr_similarity < 0.7:
                discrepancies.append({
                    "field": "address",
                    "expected": profile["mailing_address"],
                    "found": new_address,
                    "similarity": addr_similarity,
                    "threshold": 0.7
                })
        
        # Check SIN match (exact)
        if new_sin and profile.get("sin_last_three"):
            if new_sin != profile["sin_last_three"]:
                discrepancies.append({
                    "field": "sin_last_three",
                    "expected": profile["sin_last_three"],
                    "found": new_sin,
                    "similarity": 0.0,
                    "threshold": 1.0
                })
        
        # Update profile with new document
        updated_profile = self.update_profile(extracted_fields, source_document, profile_id)
        
        # Determine status
        if discrepancies:
            return {
                "status": "discrepancy",
                "message": f"⚠️ {len(discrepancies)} discrepancy(ies) detected",
                "profile": updated_profile,
                "discrepancies": discrepancies,
                "is_first_document": False
            }
        else:
            return {
                "status": "verified",
                "message": "✅ Document verified - matches user profile",
                "profile": updated_profile,
                "discrepancies": [],
                "is_first_document": False
            }
    
    def _extract_name(self, fields: Dict[str, Any]) -> str:
        """Extract name from various possible field names."""
        name_fields = ["employee_name", "student_name", "taxpayer_name", "tenant_name"]
        for field_name in name_fields:
            if field_name in fields:
                val = fields[field_name]
                if isinstance(val, dict):
                    return val.get("value", "")
                return val or ""
        return ""
    
    def _extract_address(self, fields: Dict[str, Any]) -> str:
        """Extract address from various possible field names."""
        address_fields = ["mailing_address", "rental_address"]
        for field_name in address_fields:
            if field_name in fields:
                val = fields[field_name]
                if isinstance(val, dict):
                    return val.get("value", "")
                return val or ""
        return ""
    
    def _extract_sin(self, fields: Dict[str, Any]) -> str:
        """Extract SIN last 3 digits."""
        if "sin_last_three" in fields:
            val = fields["sin_last_three"]
            if isinstance(val, dict):
                return val.get("value", "")
            return val or ""
        return ""
    
    def _get_confidence(self, fields: Dict[str, Any], field_type: str) -> float:
        """Get confidence for a field type."""
        field_map = {
            "name": ["employee_name", "student_name", "taxpayer_name", "tenant_name"],
            "address": ["mailing_address", "rental_address"],
            "sin": ["sin_last_three"]
        }
        
        for field_name in field_map.get(field_type, []):
            if field_name in fields:
                val = fields[field_name]
                if isinstance(val, dict):
                    return val.get("confidence", 0.5)
        return 0.5


# Singleton instance
_profile_service = None

def get_profile_service() -> UserProfileService:
    """Get the singleton profile service instance."""
    global _profile_service
    if _profile_service is None:
        _profile_service = UserProfileService()
    return _profile_service
