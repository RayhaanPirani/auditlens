"""
Tax Filing Agent Service
Comprehensive agent for guiding users through Canadian tax filing.
Includes CRA rule search, document verification, user Q&A, and report generation.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from services.vector_store import get_vector_store
from services.user_profile import get_profile_service

load_dotenv()
logger = logging.getLogger(__name__)

# Collection name for user responses
USER_RESPONSES_COLLECTION = "user_responses"

# Tax year being filed
CURRENT_TAX_YEAR = "2025"


# ============== USER RESPONSE STORAGE ==============

def store_user_response(
    question: str,
    answer: str,
    category: str,
    tax_year: str = CURRENT_TAX_YEAR
) -> Dict[str, Any]:
    """
    Store a user-provided response in the vector store.
    All responses are marked with 'user_provided' for audit trail.
    """
    vs = get_vector_store()
    collection = vs.get_collection(USER_RESPONSES_COLLECTION)
    
    response_id = f"user_response_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now().isoformat()
    
    metadata = {
        "question": question,
        "answer": answer,
        "category": category,
        "source": "user_provided",
        "timestamp": timestamp,
        "tax_year": tax_year,
    }
    
    # Store with question as document for searchability
    collection.upsert(
        ids=[response_id],
        documents=[f"Q: {question}\nA: {answer}"],
        metadatas=[metadata]
    )
    
    logger.info(f"Stored user response: {category} - {response_id}")
    
    return {
        "response_id": response_id,
        "stored": True,
        **metadata
    }


def get_user_responses(category: str = None, tax_year: str = CURRENT_TAX_YEAR) -> List[Dict]:
    """
    Retrieve user-provided responses, optionally filtered by category.
    """
    vs = get_vector_store()
    
    try:
        collection = vs.get_collection(USER_RESPONSES_COLLECTION)
        
        # Get all responses
        results = collection.get(include=["documents", "metadatas"])
        
        responses = []
        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            
            # Filter by tax year
            if metadata.get("tax_year") != tax_year:
                continue
                
            # Filter by category if specified
            if category and metadata.get("category") != category:
                continue
                
            responses.append({
                "response_id": doc_id,
                "question": metadata.get("question", ""),
                "answer": metadata.get("answer", ""),
                "category": metadata.get("category", ""),
                "timestamp": metadata.get("timestamp", ""),
            })
        
        return responses
        
    except Exception as e:
        logger.warning(f"Error getting user responses: {e}")
        return []


# ============== AGENT TOOLS ==============

def tool_search_cra_rules(query: str) -> str:
    """
    Search official CRA (Canada Revenue Agency) tax rules and guidelines.
    Use this to find specific tax rules, deductions, credits, and filing requirements.
    
    Args:
        query: The tax topic to search for (e.g., "RRSP contribution limits", "Ontario Trillium Benefit")
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchResults
        
        # Create search tool targeting CRA sources
        search = DuckDuckGoSearchResults(
            max_results=5,
            backend="text"
        )
        
        # Add CRA-specific keywords to improve relevance
        enhanced_query = f"site:canada.ca OR site:cra-arc.gc.ca {query} tax {CURRENT_TAX_YEAR}"
        
        results = search.run(enhanced_query)
        
        if not results:
            return f"No CRA information found for: {query}. Please try a different search term."
        
        return f"""**CRA Search Results for "{query}":**

{results}

_Source: Official Government of Canada websites_"""
        
    except Exception as e:
        logger.error(f"CRA search error: {e}")
        return f"Error searching CRA rules: {str(e)}. Try again or ask a more specific question."


def tool_check_missing_documents(query: str = "") -> str:
    """
    Analyze uploaded documents and identify potentially missing tax documents.
    Checks based on user profile and common tax situations.
    """
    vs = get_vector_store()
    profile_service = get_profile_service()
    
    all_docs = vs.get_all_documents()
    profile = profile_service.get_profile()
    
    # Categorize existing documents
    doc_types = {}
    for doc in all_docs:
        doc_type = doc.get("document_type", "Unknown")
        if doc_type not in doc_types:
            doc_types[doc_type] = []
        doc_types[doc_type].append(doc)
    
    # Check for missing documents
    missing = []
    suggestions = []
    
    # Income documents
    if "T4 - Statement of Remuneration Paid" not in doc_types:
        missing.append("📄 **T4 (Employment Income)** - Required if you had employment income")
    
    if "T5 - Statement of Investment Income" not in doc_types:
        suggestions.append("💰 **T5 (Investment Income)** - Did you receive any interest, dividends, or investment income?")
    
    # RRSP
    if "RRSP Contribution Receipt" not in doc_types:
        suggestions.append("🏦 **RRSP Contribution Receipt** - Did you contribute to an RRSP?")
    
    # FHSA
    if "FHSA Contribution Receipt" not in doc_types:
        suggestions.append("🏠 **FHSA Contribution Receipt** - Did you contribute to a First Home Savings Account?")
    
    # Students
    if "T2202 - Tuition and Enrolment Certificate" not in doc_types:
        suggestions.append("🎓 **T2202 (Tuition)** - Were you a student this year?")
    
    # Previous year's NOA (important for RRSP room)
    if "Notice of Assessment (NOA)" not in doc_types:
        suggestions.append("📋 **Notice of Assessment (NOA)** - Upload your previous year's NOA to see your RRSP/FHSA room")
    
    # Common deductions
    common_deductions = [
        ("medical_expenses", "🏥 **Medical Expenses** - Prescription drugs, dental, vision, etc."),
        ("donations", "❤️ **Charitable Donations** - Receipts for donations over $20"),
        ("rent", "🏢 **Rent Receipts** - For Ontario Trillium Benefit (if applicable)"),
        ("childcare", "👶 **Childcare Expenses** - Daycare, nanny, camps"),
        ("moving_expenses", "📦 **Moving Expenses** - If you moved for work or school"),
    ]
    
    # Check user responses to see what's already been asked
    user_responses = get_user_responses()
    asked_categories = {r.get("category") for r in user_responses}
    
    for category, description in common_deductions:
        if category not in asked_categories:
            suggestions.append(description)
    
    # Build response
    response_parts = [f"## Document Analysis for Tax Year {CURRENT_TAX_YEAR}\n"]
    response_parts.append(f"**Documents Found:** {len(all_docs)}\n")
    
    # List existing documents
    if all_docs:
        response_parts.append("### ✅ Uploaded Documents:")
        for doc_type, docs in doc_types.items():
            response_parts.append(f"- {doc_type} ({len(docs)} document(s))")
    
    # Missing documents
    if missing:
        response_parts.append("\n### ❌ Potentially Missing (Required):")
        for item in missing:
            response_parts.append(f"- {item}")
    
    # Suggestions
    if suggestions:
        response_parts.append("\n### ❓ Please Confirm:")
        for item in suggestions:
            response_parts.append(f"- {item}")
    
    response_parts.append("\n_Please upload any missing documents or answer questions about applicable deductions._")
    
    return "\n".join(response_parts)


def tool_ask_verification_question(question_data: str) -> str:
    """
    Ask a verification question and provide context for why it's needed.
    Format: "category|question"
    Example: "medical_expenses|Did you have any medical expenses this year, such as prescriptions, dental work, or vision care?"
    
    The question will be presented to the user for their response.
    """
    try:
        parts = question_data.split("|", 1)
        if len(parts) != 2:
            return "Error: Use format 'category|question'"
        
        category, question = parts
        category = category.strip()
        question = question.strip()
        
        # Format the question professionally
        formatted = f"""## ❓ Verification Question

**Category:** {category.replace("_", " ").title()}

{question}

---
_Please provide your answer. This information will be stored as "user provided" for audit purposes._
"""
        return formatted
        
    except Exception as e:
        return f"Error formatting question: {str(e)}"


def tool_store_user_response(response_data: str) -> str:
    """
    Store a user's response to a verification question.
    Format: "category|question|answer"
    Example: "medical_expenses|Did you have medical expenses?|Yes, I spent about $500 on dental work"
    
    All responses are marked 'user_provided' for audit trail.
    """
    try:
        parts = response_data.split("|", 2)
        if len(parts) != 3:
            return "Error: Use format 'category|question|answer'"
        
        category, question, answer = parts
        
        result = store_user_response(
            question=question.strip(),
            answer=answer.strip(),
            category=category.strip()
        )
        
        if result["stored"]:
            return f"""✅ **Response Recorded**

**Category:** {category.replace("_", " ").title()}
**Your Answer:** {answer}
**Reference ID:** {result['response_id']}
**Source:** User Provided (for audit trail)

_This information has been saved and will be used in your tax filing._"""
        else:
            return "❌ Error storing response. Please try again."
            
    except Exception as e:
        return f"Error storing response: {str(e)}"


def tool_get_filing_steps(query: str = "") -> str:
    """
    Generate a step-by-step filing checklist based on the user's documents and situation.
    """
    vs = get_vector_store()
    profile_service = get_profile_service()
    
    all_docs = vs.get_all_documents()
    profile = profile_service.get_profile()
    user_responses = get_user_responses()
    
    # Determine what the user has
    has_t4 = any("T4" in d.get("document_type", "") for d in all_docs)
    has_t2202 = any("T2202" in d.get("document_type", "") for d in all_docs)
    has_rrsp = any("RRSP" in d.get("document_type", "") for d in all_docs)
    has_fhsa = any("FHSA" in d.get("document_type", "") for d in all_docs)
    has_noa = any("NOA" in d.get("document_type", "") for d in all_docs)
    has_rent = any("Rent" in d.get("document_type", "") for d in all_docs)
    
    # Build checklist
    steps = [f"""# 📋 Tax Filing Checklist for {CURRENT_TAX_YEAR}

## Personal Information (Step 1)
"""]
    
    if profile:
        steps.append(f"- ✅ Name: {profile.get('taxpayer_name', 'Not set')}")
        steps.append(f"- ✅ Address: {profile.get('mailing_address', 'Not set')}")
    else:
        steps.append("- ⬜ Confirm your personal information")
    
    # Income
    steps.append("\n## Income (Step 2)")
    if has_t4:
        steps.append("- ✅ Employment income (T4) - Uploaded")
    else:
        steps.append("- ⬜ Employment income - Upload T4 or confirm none")
    
    # Deductions
    steps.append("\n## Deductions (Step 3)")
    if has_rrsp:
        steps.append("- ✅ RRSP contributions - Uploaded")
    else:
        steps.append("- ⬜ RRSP contributions - Upload receipt or confirm none")
    
    if has_fhsa:
        steps.append("- ✅ FHSA contributions - Uploaded")
    else:
        steps.append("- ⬜ FHSA contributions - Upload receipt or confirm none")
    
    # Credits
    steps.append("\n## Credits (Step 4)")
    if has_t2202:
        steps.append("- ✅ Tuition credits (T2202) - Uploaded")
    else:
        steps.append("- ⬜ Tuition credits - Upload T2202 if applicable")
    
    # User responses
    if user_responses:
        steps.append("\n## User-Provided Information")
        for resp in user_responses:
            cat = resp.get("category", "").replace("_", " ").title()
            steps.append(f"- ✅ {cat} - Answered")
    
    # Final steps
    steps.append("\n## Final Steps (Step 5)")
    steps.append("- ⬜ Review all entries")
    steps.append("- ⬜ Generate tax report")
    steps.append("- ⬜ File using NETFILE-certified software")
    
    steps.append(f"\n---\n_Checklist generated based on {len(all_docs)} documents and {len(user_responses)} user responses._")
    
    return "\n".join(steps)


def tool_tax_advice(query: str) -> str:
    """
    Provide personalized tax advice based on the user's situation.
    Topics: RRSP optimization, FHSA suggestions, carry-forward strategies, timing advice.
    
    This tool specifically checks for NOA (Notice of Assessment) documents to find
    carry-forward information like RRSP room, FHSA room, tuition credits, and capital losses.
    """
    vs = get_vector_store()
    profile_service = get_profile_service()
    
    all_docs = vs.get_all_documents()
    profile = profile_service.get_profile()
    
    # Initialize data containers
    total_income = 0
    rrsp_contributions = 0
    fhsa_contributions = 0
    tuition_amount = 0
    
    # NOA-specific carry-forward data
    noa_data = {
        "found": False,
        "tax_year": None,
        "rrsp_deduction_limit": 0,
        "rrsp_unused_contributions": 0,
        "rrsp_carry_forward": 0,
        "fhsa_contribution_room": 0,
        "fhsa_unused_room": 0,
        "tuition_carry_forward": 0,
        "capital_loss_carry_forward": 0,
        "net_income": 0,
        "taxable_income": 0,
    }
    
    for doc in all_docs:
        doc_type = doc.get("document_type", "")
        fields = doc.get("extracted_fields", {})
        
        # Check if this is an NOA document - extract carry-forward data
        if "NOA" in doc_type or "Notice of Assessment" in doc_type:
            noa_data["found"] = True
            
            # Helper function to extract numeric value
            def get_value(field_name):
                field = fields.get(field_name, {})
                if isinstance(field, dict):
                    val = field.get("value", "0")
                else:
                    val = field or "0"
                try:
                    return float(str(val).replace("$", "").replace(",", ""))
                except:
                    return 0
            
            # Extract all NOA carry-forward fields
            noa_data["tax_year"] = fields.get("assessment_year", {}).get("value") if isinstance(fields.get("assessment_year"), dict) else fields.get("assessment_year")
            noa_data["rrsp_deduction_limit"] = get_value("rrsp_deduction_limit")
            noa_data["rrsp_unused_contributions"] = get_value("rrsp_unused_contributions")
            noa_data["rrsp_carry_forward"] = get_value("rrsp_carry_forward")
            noa_data["fhsa_contribution_room"] = get_value("fhsa_contribution_room")
            noa_data["fhsa_unused_room"] = get_value("fhsa_unused_room")
            noa_data["tuition_carry_forward"] = get_value("tuition_carry_forward")
            noa_data["capital_loss_carry_forward"] = get_value("capital_loss_carry_forward")
            noa_data["net_income"] = get_value("net_income")
            noa_data["taxable_income"] = get_value("taxable_income")
        
        # Process other document types for current year data
        # Income from T4
        emp_income = fields.get("employment_income", {})
        if isinstance(emp_income, dict):
            val = emp_income.get("value", "0")
        else:
            val = emp_income or "0"
        try:
            total_income += float(str(val).replace("$", "").replace(",", ""))
        except:
            pass
        
        # RRSP contributions from receipts
        rrsp = fields.get("rrsp_contribution", {})
        if isinstance(rrsp, dict):
            val = rrsp.get("value", "0")
        else:
            val = rrsp or "0"
        try:
            rrsp_contributions += float(str(val).replace("$", "").replace(",", ""))
        except:
            pass
        
        # FHSA contributions from receipts
        fhsa = fields.get("fhsa_contribution", {})
        if isinstance(fhsa, dict):
            val = fhsa.get("value", "0")
        else:
            val = fhsa or "0"
        try:
            fhsa_contributions += float(str(val).replace("$", "").replace(",", ""))
        except:
            pass
        
        # Tuition from T2202
        tuition = fields.get("tuition_amount", {})
        if isinstance(tuition, dict):
            val = tuition.get("value", "0")
        else:
            val = tuition or "0"
        try:
            tuition_amount += float(str(val).replace("$", "").replace(",", ""))
        except:
            pass
    
    # Generate advice
    advice_parts = [f"# 💡 Personalized Tax Advice for {CURRENT_TAX_YEAR}\n"]
    
    # === NOA-BASED CARRY-FORWARD INFORMATION ===
    if noa_data["found"]:
        advice_parts.append(f"""## 📋 From Your Previous NOA {"(" + noa_data["tax_year"] + ")" if noa_data["tax_year"] else ""}

Your Notice of Assessment shows the following carry-forward amounts:
""")
        
        # RRSP Room from NOA
        if noa_data["rrsp_deduction_limit"] > 0:
            remaining_rrsp_room = noa_data["rrsp_deduction_limit"] - rrsp_contributions
            advice_parts.append(f"""### 🏦 RRSP Contribution Room
- **Deduction Limit:** ${noa_data["rrsp_deduction_limit"]:,.2f}
- **This Year's Contributions:** ${rrsp_contributions:,.2f}
- **Remaining Room:** ${remaining_rrsp_room:,.2f}
""")
            if noa_data["rrsp_unused_contributions"] > 0:
                advice_parts.append(f"- **Unused Contributions (from prior years):** ${noa_data['rrsp_unused_contributions']:,.2f}")
            if remaining_rrsp_room > 0:
                advice_parts.append(f"""
**Recommendation:** You have **${remaining_rrsp_room:,.2f}** of unused RRSP room.
- Deadline: March 1, {int(CURRENT_TAX_YEAR) + 1}
- If expecting higher income next year, contribute now but defer the deduction
""")
        
        # FHSA Room from NOA
        if noa_data["fhsa_contribution_room"] > 0 or noa_data["fhsa_unused_room"] > 0:
            remaining_fhsa_room = noa_data["fhsa_contribution_room"] - fhsa_contributions
            advice_parts.append(f"""### 🏠 FHSA Contribution Room
- **Contribution Room:** ${noa_data["fhsa_contribution_room"]:,.2f}
- **This Year's Contributions:** ${fhsa_contributions:,.2f}
- **Remaining Room:** ${remaining_fhsa_room:,.2f}
""")
            if noa_data["fhsa_unused_room"] > 0:
                advice_parts.append(f"- **Unused Room Carried Forward:** ${noa_data['fhsa_unused_room']:,.2f}")
        
        # Tuition Carry-Forward from NOA
        if noa_data["tuition_carry_forward"] > 0:
            advice_parts.append(f"""### 🎓 Tuition Credit Carry-Forward
- **Unused Tuition Credits:** ${noa_data["tuition_carry_forward"]:,.2f}

**Tip:** These credits carry forward indefinitely. Use them in years when you have enough income to benefit from the tax reduction.
""")
        
        # Capital Loss Carry-Forward from NOA
        if noa_data["capital_loss_carry_forward"] > 0:
            advice_parts.append(f"""### 📉 Capital Loss Carry-Forward
- **Net Capital Losses:** ${noa_data["capital_loss_carry_forward"]:,.2f}

**Tip:** Capital losses can be used to offset capital gains. They carry forward indefinitely.
""")
        
        # Income comparison
        if noa_data["net_income"] > 0 and total_income > 0:
            income_change = ((total_income - noa_data["net_income"]) / noa_data["net_income"]) * 100
            change_direction = "increased" if income_change > 0 else "decreased"
            advice_parts.append(f"""### 📊 Income Comparison
- **Previous Year Net Income:** ${noa_data["net_income"]:,.2f}
- **This Year's Income:** ${total_income:,.2f}
- **Change:** {change_direction} by **{abs(income_change):.1f}%**
""")
    else:
        advice_parts.append("""## ⚠️ No NOA Found

Upload your previous year's **Notice of Assessment (NOA)** to see:
- Your RRSP contribution room
- FHSA contribution room
- Tuition credits carried forward
- Capital losses carried forward

This helps maximize your tax deductions!
""")
    
    # Current year tuition (from T2202)
    if tuition_amount > 0:
        advice_parts.append(f"""## 🎓 This Year's Tuition Credits

You have **${tuition_amount:,.2f}** in tuition fees from this year.

**Options:**
- Claim now if you have enough income
- Carry forward to a higher-income year
- Transfer up to $5,000 to a parent/grandparent
""")
    
    # FHSA general advice (if no NOA)
    if not noa_data["found"] and total_income > 0:
        advice_parts.append(f"""## 🏠 FHSA (First Home Savings Account)

If you're a first-time home buyer:
- Annual contribution limit: **$8,000**
- Lifetime limit: **$40,000**
- Contributions are tax-deductible AND withdrawals for home purchase are tax-free
- Unused room carries forward (max $8,000 per year)
""")
    
    # General advice
    advice_parts.append(f"""## 📝 General Recommendations

1. **Upload your NOA** if you haven't already - it contains your exact RRSP/FHSA room
2. **RRSP deadline** is March 1, {int(CURRENT_TAX_YEAR) + 1}
3. **Keep all receipts** for 6 years
4. **Consider income timing** - defer deductions if you expect higher income next year
""")
    
    return "\n".join(advice_parts)


def tool_generate_tax_report(include_advice: str = "yes") -> str:
    """
    Generate a comprehensive HTML tax report for filing.
    This report can be used to populate tax software.
    Uses GPT-5.2 for high-quality generation.
    
    Args:
        include_advice: "yes" to include personalized advice, "no" for data only
    """
    # This function generates the report structure
    # The actual HTML generation happens in a separate call with GPT-5.2
    
    vs = get_vector_store()
    profile_service = get_profile_service()
    
    all_docs = vs.get_all_documents()
    profile = profile_service.get_profile()
    user_responses = get_user_responses()
    
    # Gather all data
    report_data = {
        "tax_year": CURRENT_TAX_YEAR,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "personal_info": {
            "name": profile.get("taxpayer_name", "Not provided") if profile else "Not provided",
            "address": profile.get("mailing_address", "Not provided") if profile else "Not provided",
            "sin_last_three": profile.get("sin_last_three", "***") if profile else "***",
        },
        "income": {},
        "deductions": {},
        "credits": {},
        "user_provided": [],
        "documents": []
    }
    
    # Process documents
    for doc in all_docs:
        doc_info = {
            "type": doc.get("document_type", "Unknown"),
            "filename": doc.get("original_filename", "Unknown"),
            "confidence": doc.get("overall_confidence", 0),
        }
        report_data["documents"].append(doc_info)
        
        fields = doc.get("extracted_fields", {})
        
        # Categorize fields
        for field_name, field_data in fields.items():
            if isinstance(field_data, dict):
                value = field_data.get("value", "")
                confidence = field_data.get("confidence", 0)
            else:
                value = field_data
                confidence = 1.0
            
            if not value:
                continue
            
            # Categorize
            if "income" in field_name.lower() or field_name in ["employment_income", "total_income", "other_income"]:
                report_data["income"][field_name] = {"value": value, "confidence": confidence}
            elif field_name in ["rrsp_contribution", "fhsa_contribution", "cpp_contributions", "ei_premiums", "income_tax_deducted"]:
                report_data["deductions"][field_name] = {"value": value, "confidence": confidence}
            elif field_name in ["tuition_amount", "donation_amount", "medical_expenses"]:
                report_data["credits"][field_name] = {"value": value, "confidence": confidence}
    
    # Add user responses
    for resp in user_responses:
        report_data["user_provided"].append({
            "category": resp.get("category", ""),
            "question": resp.get("question", ""),
            "answer": resp.get("answer", ""),
        })
    
    # Generate HTML report
    html_report = generate_html_report(report_data, include_advice=(include_advice.lower() == "yes"))
    
    # Save report
    report_dir = os.path.join(os.path.dirname(__file__), "..", "uploads", "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    report_filename = f"tax_report_{CURRENT_TAX_YEAR}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path = os.path.join(report_dir, report_filename)
    
    with open(report_path, "w") as f:
        f.write(html_report)
    
    return f"""✅ **Tax Report Generated Successfully!**

📄 **Report:** {report_filename}
📁 **Location:** {report_path}

The report includes:
- Personal information (masked SIN)
- Income summary from all T4s and other sources
- Deductions (RRSP, FHSA, CPP, EI)
- Credits (Tuition, donations, medical)
- User-provided information (marked for audit)
- {"Personalized advice" if include_advice.lower() == "yes" else "Data only (no advice)"}

You can open this HTML file in any browser and print it for your records.

REPORT_PATH:{report_path}:END_REPORT_PATH"""


def generate_html_report(data: Dict, include_advice: bool = True) -> str:
    """Generate the actual HTML report content."""
    
    # Calculate totals
    total_income = 0
    for field, info in data.get("income", {}).items():
        try:
            val = info.get("value", "0").replace("$", "").replace(",", "")
            total_income += float(val)
        except:
            pass
    
    total_deductions = 0
    for field, info in data.get("deductions", {}).items():
        if field in ["rrsp_contribution", "fhsa_contribution"]:
            try:
                val = info.get("value", "0").replace("$", "").replace(",", "")
                total_deductions += float(val)
            except:
                pass
    
    # Estimate tax (simplified)
    taxable_income = max(0, total_income - total_deductions - 15705)  # BPA
    estimated_federal_tax = taxable_income * 0.20  # Simplified
    estimated_provincial_tax = taxable_income * 0.05  # Simplified
    
    # Get taxes already paid
    taxes_paid = 0
    for field, info in data.get("deductions", {}).items():
        if field == "income_tax_deducted":
            try:
                val = info.get("value", "0").replace("$", "").replace(",", "")
                taxes_paid += float(val)
            except:
                pass
    
    estimated_refund = taxes_paid - (estimated_federal_tax + estimated_provincial_tax)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tax Filing Report - {data['tax_year']}</title>
    <style>
        :root {{
            --primary: #1a365d;
            --secondary: #2b6cb0;
            --accent: #48bb78;
            --warning: #ed8936;
            --danger: #e53e3e;
            --bg: #f7fafc;
            --card: #ffffff;
            --text: #2d3748;
            --muted: #718096;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        
        header p {{
            opacity: 0.9;
        }}
        
        .card {{
            background: var(--card);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .card h2 {{
            color: var(--primary);
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--bg);
        }}
        
        .card h2 .icon {{
            margin-right: 0.5rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--bg);
        }}
        
        th {{
            color: var(--muted);
            font-weight: 500;
            font-size: 0.875rem;
            text-transform: uppercase;
        }}
        
        .amount {{
            font-family: 'Menlo', monospace;
            font-weight: 600;
        }}
        
        .amount.positive {{
            color: var(--accent);
        }}
        
        .amount.negative {{
            color: var(--danger);
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        
        .summary-item {{
            background: var(--bg);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        
        .summary-item .label {{
            color: var(--muted);
            font-size: 0.875rem;
        }}
        
        .summary-item .value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
        }}
        
        .refund-box {{
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin-top: 1rem;
        }}
        
        .refund-box.owing {{
            background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        }}
        
        .refund-box .amount {{
            font-size: 2.5rem;
            font-weight: 700;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .badge.user-provided {{
            background: #fef3c7;
            color: #92400e;
        }}
        
        .badge.document {{
            background: #dbeafe;
            color: #1e40af;
        }}
        
        .advice-box {{
            background: linear-gradient(135deg, #ebf8ff 0%, #e6fffa 100%);
            border-left: 4px solid var(--secondary);
            padding: 1rem 1.5rem;
            border-radius: 0 8px 8px 0;
            margin-top: 1rem;
        }}
        
        .advice-box h3 {{
            color: var(--secondary);
            margin-bottom: 0.5rem;
        }}
        
        footer {{
            text-align: center;
            color: var(--muted);
            font-size: 0.875rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--bg);
        }}
        
        @media print {{
            body {{
                background: white;
            }}
            .container {{
                padding: 0;
            }}
            .card {{
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📋 Tax Filing Report</h1>
            <p>Tax Year {data['tax_year']} • Generated {data['generated_at']}</p>
        </header>
        
        <!-- Personal Information -->
        <div class="card">
            <h2><span class="icon">👤</span>Personal Information</h2>
            <table>
                <tr>
                    <td><strong>Name</strong></td>
                    <td>{data['personal_info']['name']}</td>
                </tr>
                <tr>
                    <td><strong>Address</strong></td>
                    <td>{data['personal_info']['address']}</td>
                </tr>
                <tr>
                    <td><strong>SIN</strong></td>
                    <td>***-***-{data['personal_info']['sin_last_three']}</td>
                </tr>
            </table>
        </div>
        
        <!-- Summary -->
        <div class="card">
            <h2><span class="icon">📊</span>Tax Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="label">Total Income</div>
                    <div class="value">${total_income:,.2f}</div>
                </div>
                <div class="summary-item">
                    <div class="label">Total Deductions</div>
                    <div class="value">${total_deductions:,.2f}</div>
                </div>
                <div class="summary-item">
                    <div class="label">Taxable Income</div>
                    <div class="value">${taxable_income:,.2f}</div>
                </div>
                <div class="summary-item">
                    <div class="label">Tax Paid at Source</div>
                    <div class="value">${taxes_paid:,.2f}</div>
                </div>
            </div>
            
            <div class="refund-box {'owing' if estimated_refund < 0 else ''}">
                <div class="label">{'Estimated Balance Owing' if estimated_refund < 0 else 'Estimated Refund'}</div>
                <div class="amount">${abs(estimated_refund):,.2f}</div>
            </div>
        </div>
        
        <!-- Income Details -->
        <div class="card">
            <h2><span class="icon">💰</span>Income</h2>
            <table>
                <thead>
                    <tr>
                        <th>Source</th>
                        <th>Amount</th>
                        <th>Source</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add income rows
    for field, info in data.get("income", {}).items():
        label = field.replace("_", " ").title()
        html += f"""                    <tr>
                        <td>{label}</td>
                        <td class="amount">{info.get('value', 'N/A')}</td>
                        <td><span class="badge document">Document</span></td>
                    </tr>
"""
    
    html += """                </tbody>
            </table>
        </div>
        
        <!-- Deductions -->
        <div class="card">
            <h2><span class="icon">📉</span>Deductions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Type</th>
                        <th>Amount</th>
                        <th>Source</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add deduction rows
    for field, info in data.get("deductions", {}).items():
        label = field.replace("_", " ").title()
        html += f"""                    <tr>
                        <td>{label}</td>
                        <td class="amount">{info.get('value', 'N/A')}</td>
                        <td><span class="badge document">Document</span></td>
                    </tr>
"""
    
    html += """                </tbody>
            </table>
        </div>
"""
    
    # User provided information
    if data.get("user_provided"):
        html += """        <!-- User Provided Information -->
        <div class="card">
            <h2><span class="icon">✍️</span>User-Provided Information</h2>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Details</th>
                        <th>Source</th>
                    </tr>
                </thead>
                <tbody>
"""
        for resp in data["user_provided"]:
            html += f"""                    <tr>
                        <td>{resp['category'].replace('_', ' ').title()}</td>
                        <td>{resp['answer']}</td>
                        <td><span class="badge user-provided">User Provided</span></td>
                    </tr>
"""
        html += """                </tbody>
            </table>
        </div>
"""
    
    # Advice section
    if include_advice:
        html += """        <!-- Advice -->
        <div class="card">
            <h2><span class="icon">💡</span>Personalized Advice</h2>
            
            <div class="advice-box">
                <h3>RRSP Contribution</h3>
                <p>Consider maximizing your RRSP contribution before March 1 to reduce your taxable income. If you expect higher income next year, you can contribute now but defer the deduction.</p>
            </div>
            
            <div class="advice-box">
                <h3>FHSA Opportunity</h3>
                <p>If you're a first-time home buyer, consider opening a First Home Savings Account. You can contribute up to $8,000/year with tax deductions similar to RRSPs, and withdrawals for home purchase are tax-free.</p>
            </div>
            
            <div class="advice-box">
                <h3>Keep Records</h3>
                <p>Retain all tax documents for at least 6 years. CRA may request documentation to verify your claims.</p>
            </div>
        </div>
"""
    
    # Documents list
    html += """        <!-- Source Documents -->
        <div class="card">
            <h2><span class="icon">📁</span>Source Documents</h2>
            <table>
                <thead>
                    <tr>
                        <th>Document</th>
                        <th>Type</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for doc in data.get("documents", []):
        conf = doc.get("confidence", 0) * 100
        conf_color = "positive" if conf >= 85 else "warning" if conf >= 70 else "negative"
        html += f"""                    <tr>
                        <td>{doc.get('filename', 'Unknown')}</td>
                        <td>{doc.get('type', 'Unknown')}</td>
                        <td class="amount {conf_color}">{conf:.0f}%</td>
                    </tr>
"""
    
    html += f"""                </tbody>
            </table>
        </div>
        
        <footer>
            <p>This report is for informational purposes only. Please verify all information before filing.</p>
            <p>Generated by AuditLens • {data['generated_at']}</p>
        </footer>
    </div>
</body>
</html>
"""
    
    return html


# ============== AGENT CREATION ==============

TAX_FILING_SYSTEM_PROMPT = """You are an expert Canadian tax accountant helping users file their taxes for {tax_year}.

Your role is to:
1. **Guide users** through the tax filing process step by step
2. **Search CRA rules** when you need to verify specific tax regulations
3. **Check for missing documents** based on the user's situation
4. **Ask verification questions** like an accountant would to ensure completeness
5. **Store user responses** with clear "user provided" attribution for audit trail
6. **Provide personalized advice** on tax optimization (RRSP, FHSA, carry-forwards)
7. **Generate a professional tax report** when all information is gathered

Province awareness: Ask the user's province if not known, as it affects:
- Provincial tax rates
- Provincial credits (e.g., Ontario Trillium Benefit, BC Climate Action Credit)
- Rent deduction eligibility

**CRITICAL - Vector Store Queries:**
When searching documents or tax laws, use SHORT KEYWORD-BASED queries, NOT verbose prompts.
The search uses embeddings - concise queries work best.

✅ GOOD query examples:
- "noa_2024 tuition carryforward"
- "T4 employment income"
- "RRSP contribution room"
- "medical expenses receipts"

❌ BAD query examples (too verbose):
- "Find the unused tuition amount carried forward from prior years for the taxpayer"
- "What is the total employment income shown on the T4 statement"

Keep queries to 3-5 keywords maximum. Include the document filename if looking for specific data.

Available tools:
- search_cra_rules: Search official CRA websites for tax rules
- check_missing_documents: Identify potentially missing documents
- ask_verification_question: Ask accountant-style verification questions
- store_user_response: Store user answers with audit trail
- get_filing_steps: Generate a filing checklist
- tax_advice: Provide personalized tax advice (checks NOA for carry-forwards automatically)
- generate_tax_report: Create the final HTML tax report

Always be helpful, professional, and accurate. If unsure, search CRA rules or ask the user."""

def create_tax_filing_agent():
    """Create and return the tax filing agent executor."""
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    try:
        # Get model from env or use default
        model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        
        llm = ChatOpenAI(
            model=model_name,
            temperature=1,
            streaming=True,
            api_key=openai_api_key
        )
        
        # Define tools
        tools = [
            Tool(
                name="search_cra_rules",
                func=tool_search_cra_rules,
                description="Search official CRA (Canada Revenue Agency) websites for tax rules, deductions, credits, and filing requirements. Use for specific tax questions."
            ),
            Tool(
                name="check_missing_documents",
                func=tool_check_missing_documents,
                description="Analyze uploaded documents and identify potentially missing tax documents based on the user's situation."
            ),
            Tool(
                name="ask_verification_question",
                func=tool_ask_verification_question,
                description="Ask a verification question. Format: 'category|question'. Example: 'medical_expenses|Did you have any medical expenses this year?'"
            ),
            Tool(
                name="store_user_response",
                func=tool_store_user_response,
                description="Store a user's response. Format: 'category|question|answer'. All responses marked 'user_provided' for audit."
            ),
            Tool(
                name="get_filing_steps",
                func=tool_get_filing_steps,
                description="Generate a step-by-step tax filing checklist based on the user's documents and situation."
            ),
            Tool(
                name="tax_advice",
                func=tool_tax_advice,
                description="Provide personalized tax advice on RRSP, FHSA, deductions, credits, and optimization strategies."
            ),
            Tool(
                name="generate_tax_report",
                func=tool_generate_tax_report,
                description="Generate a comprehensive HTML tax report. Use 'yes' to include advice or 'no' for data only."
            ),
        ]
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", TAX_FILING_SYSTEM_PROMPT.format(tax_year=CURRENT_TAX_YEAR)),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
        )
        
        logger.info("Tax filing agent created successfully")
        return agent_executor
        
    except Exception as e:
        logger.error(f"Error creating tax filing agent: {e}")
        raise


# Singleton instance
_tax_filing_agent: Optional[AgentExecutor] = None


def get_tax_filing_agent() -> AgentExecutor:
    """Get the singleton tax filing agent instance."""
    global _tax_filing_agent
    if _tax_filing_agent is None:
        _tax_filing_agent = create_tax_filing_agent()
    return _tax_filing_agent


async def run_tax_filing_agent(
    query: str,
    chat_history: list = None,
    callbacks: Optional[List[Any]] = None
) -> str:
    """Run the tax filing agent with a query and return the response."""
    agent = get_tax_filing_agent()
    
    try:
        result = await agent.ainvoke(
            {
                "input": query,
                "chat_history": chat_history or [],
            },
            config={"callbacks": callbacks} if callbacks else None
        )
        return result.get("output", "I processed your tax filing request.")
    except Exception as e:
        logger.error(f"Error running tax filing agent: {e}")
        return f"Error processing tax filing request: {str(e)}"
