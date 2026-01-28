"""
AuditLens - Tax Document Processing and Auditing Assistant
Main Chainlit application for document upload and processing.
"""

import os
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import chainlit as cl
from chainlit.element import Element
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import services
from services.document_parser import parse_document, compute_file_hash, ParsedDocument
from services.vector_store import get_vector_store

# Configuration
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp"}


def get_file_type(filename: str) -> str:
    """Determine file type from extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return "PDF Document"
    elif ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "Image"
    return "Unknown"


async def save_uploaded_file(file: Element) -> Dict[str, Any]:
    """
    Save an uploaded file to the uploads directory.
    Returns metadata about the saved file.
    """
    # Generate unique filename to avoid collisions
    original_name = file.name
    ext = Path(original_name).suffix.lower()
    unique_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{unique_id}{ext}"
    
    # Save file
    save_path = UPLOAD_DIR / safe_name
    
    # Read the file content and save it
    with open(file.path, "rb") as src:
        content = src.read()
    
    with open(save_path, "wb") as dst:
        dst.write(content)
    
    # Compute file hash for deduplication
    file_hash = compute_file_hash(str(save_path))
    
    return {
        "id": unique_id,
        "original_name": original_name,
        "saved_name": safe_name,
        "path": str(save_path.absolute()),
        "size": len(content),
        "type": get_file_type(original_name),
        "uploaded_at": datetime.now().isoformat(),
        "processed": False,
        "confidence_score": None,
        "file_hash": file_hash,
    }


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session with welcome message and instructions."""
    
    # Initialize session storage for documents and chat history
    cl.user_session.set("documents", [])
    cl.user_session.set("chat_history", [])  # For conversation memory
    
    # Check for API keys
    has_landingai = bool(os.getenv("VISION_AGENT_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    # Initialize vector store
    try:
        vector_store = get_vector_store()
        doc_count = vector_store.get_document_count()
        store_status = f"📊 **{doc_count}** documents in knowledge base"
    except Exception as e:
        store_status = "⚠️ Vector store not initialized"
    
    # Build API status
    api_status = ""
    if not has_landingai or not has_openai:
        api_status = "\n\n> ⚠️ **API Keys Missing:** "
        missing = []
        if not has_landingai:
            missing.append("VISION_AGENT_API_KEY")
        if not has_openai:
            missing.append("OPENAI_API_KEY")
        api_status += ", ".join(missing) + " not set in `.env`"
    
    # Welcome message with rich formatting
    welcome_message = f"""# 🧾 Welcome to AuditLens

**Your AI-Powered Tax Document Assistant**

I help you process and audit your tax documents with ease. Here's what I can do:

---

### 📤 **Upload Your Documents**
Simply attach your tax documents (PDFs or images) to any message. Supported documents include:
- **T4 Forms** – Employment income
- **T2202** – Tuition tax certificates  
- **RRSP Contribution Receipts**
- **Rent Receipts**
- **Moving Expense Receipts**
- And more...

### 🔍 **Smart Processing**
Each document is analyzed using **LandingAI DPT-2** to extract key information. If any data has low confidence, I'll ask you to verify it.

### 💡 **Tax Recommendations**  
Once your documents are processed, I'll help calculate your taxes and provide personalized recommendations to reduce your tax burden.

---

{store_status}{api_status}

**Ready to get started?** Upload your first tax document by clicking the 📎 attachment icon below!
"""
    
    # Add persistent "View Documents" button to the welcome message
    actions = [
        cl.Action(
            name="view_documents",
            payload={"action": "view"},
            label="📂 View All Documents",
            description="Access your knowledge base"
        )
    ]
    
    await cl.Message(content=welcome_message, actions=actions).send()


@cl.action_callback("delete_document_start")
async def delete_document_start(action: cl.Action):
    """Show options to delete documents."""
    try:
        vector_store = get_vector_store()
        # Only allow deleting from personal documents
        stored_docs = vector_store.get_all_documents(collection_name="tax_documents")
    except Exception as e:
        await cl.Message(content=f"❌ Error accessing document store: {e}").send()
        return

    if not stored_docs:
        await cl.Message(content="🗑️ **No documents to delete.**").send()
        return
        
    actions = []
    for doc in stored_docs:
        fname = doc.get("original_filename", "Unknown")
        file_hash = doc.get("file_hash")
        actions.append(
            cl.Action(
                name="delete_document_confirm",
                payload={"file_hash": file_hash, "filename": fname},
                label=f"🗑️ Delete {fname}",
                description=f"Permanently delete {fname}"
            )
        )
    
    # Add cancel option
    actions.append(
        cl.Action(
            name="view_documents",
            payload={"action": "view"},
            label="❌ Cancel",
            description="Go back to document list"
        )
    )
    
    await cl.Message(
        content="⚠️ **Select a document to PERMANENTLY delete:**\nThis action cannot be undone.",
        actions=actions
    ).send()


@cl.action_callback("delete_document_confirm")
async def delete_document_confirm(action: cl.Action):
    """Execute document deletion."""
    file_hash = action.payload.get("file_hash")
    filename = action.payload.get("filename")
    
    if not file_hash:
        return
        
    try:
        vector_store = get_vector_store()
        success = vector_store.delete_document(file_hash, collection_name="tax_documents")
        
        if success:
            # Also remove from session if present
            session_docs = cl.user_session.get("documents") or []
            session_docs = [d for d in session_docs if d.get("file_hash") != file_hash]
            cl.user_session.set("documents", session_docs)
            
            await cl.Message(content=f"✅ **Deleted:** {filename}").send()
            
            # Show updated list
            await view_documents_action(None)
        else:
            await cl.Message(content=f"❌ Failed to delete {filename}").send()
            
    except Exception as e:
        await cl.Message(content=f"❌ Error during deletion: {e}").send()


@cl.action_callback("view_documents")
async def view_documents_action(action: cl.Action):
    """Handle the View All Documents action."""
    documents = cl.user_session.get("documents") or []
    
    # Also get documents from vector store
    try:
        vector_store = get_vector_store()
        stored_docs = vector_store.get_all_documents(collection_name="tax_documents")
        print(f"DEBUG: Retrieved {len(stored_docs)} docs from tax_documents")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"DEBUG: Error retrieving docs: {e}")
        stored_docs = []
    
    if not documents and not stored_docs:
        await cl.Message(content="📂 **No documents uploaded yet.**\n\nUpload your tax documents by attaching them to a message.").send()
        return
    
    # Build document list
    doc_list = "# 📂 Your Uploaded Documents\n\n"
    
    # Session documents (current session)
    if documents:
        doc_list += f"### Current Session: {len(documents)} document(s)\n\n"
        for i, doc in enumerate(documents, 1):
            status = "✅ Processed" if doc.get("processed") else "⏳ Pending"
            confidence = doc.get("confidence_score")
            conf_display = f" ({confidence * 100:.0f}% confidence)" if confidence else ""
            cached = " 📦 *cached*" if doc.get("from_cache") else ""
            
            doc_list += f"**{i}. {doc['original_name']}**{cached}\n"
            doc_list += f"- Type: {doc.get('document_type', doc['type'])} | {status}{conf_display}\n\n"
    
    # Stored documents (from ChromaDB)
    if stored_docs:
        doc_list += f"\n### 💾 Knowledge Base: {len(stored_docs)} document(s)\n\n"
        for doc in stored_docs[:10]:  # Show first 10
            conf = doc.get("overall_confidence", 0) * 100
            doc_list += f"- **{doc['original_filename']}** – {doc['document_type']} ({conf:.0f}%)\n"
    
    # Create elements for image previews (session docs only)
    elements = []
    for doc in documents[:5]:  # Limit previews
        if doc["type"] == "Image" and os.path.exists(doc["path"]):
            elements.append(
                cl.Image(
                    name=doc["original_name"],
                    path=doc["path"],
                    display="inline",
                    size="medium"
                )
            )
        elif doc["type"] == "PDF Document" and os.path.exists(doc["path"]):
            elements.append(
                cl.Pdf(
                    name=doc["original_name"],
                    path=doc["path"],
                    display="inline"
                )
            )
            
    # Add actions (Delete button)
    actions = [
        cl.Action(
            name="delete_document_start",
            payload={},
            label="🗑️ Delete a Document",
            description="Remove a document from the database"
        )
    ]
    
    await cl.Message(content=doc_list, elements=elements if elements else None, actions=actions).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages and file uploads."""
    
    # Check if there are file attachments
    if message.elements:
        await handle_file_uploads(message.elements)
    else:
        # Regular text message - route to Tax Agent
        text = message.content.lower()
        
        # Handle simple UI commands locally
        # Handle simple UI commands locally
        if text in ["view", "list", "show", "documents", "files", "show documents"]:
            # Trigger view documents
            await view_documents_action(None)
            
        elif text == "help":
            response = """I'm your agentic tax assistant! Here's what you can do:

1. **📤 Upload Documents** – Attach tax forms to this chat
2. **💰 Calculate Taxes** – Ask "How much tax do I owe?"
3. **📊 Summarize** – Ask "Summarize my documents"
4. **💡 Get Tips** – Ask "How can I reduce my tax?"
5. **🔍 Search** – Ask "Find my tuition receipt"
6. **📋 File Taxes** – Ask "Help me file my taxes" or "Generate tax report"

How can I help you today?"""
            
            # Add view documents action button
            actions = [
                cl.Action(
                    name="view_documents",
                    payload={"action": "view"},
                    label="📂 View All Documents",
                    description="See all your uploaded documents"
                )
            ]
            await cl.Message(content=response, actions=actions).send()
            
        else:
            # Check if this is a tax filing query
            filing_keywords = ["file my taxes", "file taxes", "filing", "generate report", "tax report", 
                              "missing documents", "what documents", "cra rules", "verification", 
                              "am i ready to file", "help me file", "filing checklist", "start filing"]
            is_filing_query = any(kw in text for kw in filing_keywords)
            
            # Show a thinking indicator
            msg = cl.Message(content="")
            await msg.send()
            
            if is_filing_query:
                # Use Tax Filing Agent for filing-related queries
                try:
                    from services.tax_filing_agent import get_tax_filing_agent
                    from langchain_core.messages import HumanMessage, AIMessage
                    
                    agent = get_tax_filing_agent()
                    chat_history = cl.user_session.get("chat_history", [])
                    
                    # Tool descriptions for filing agent
                    tool_descriptions = {
                        "search_cra_rules": "🔍 Searching CRA rules",
                        "check_missing_documents": "📋 Checking documents",
                        "ask_verification_question": "❓ Preparing question",
                        "store_user_response": "💾 Storing response",
                        "get_filing_steps": "📝 Generating checklist",
                        "tax_advice": "💡 Preparing advice",
                        "generate_tax_report": "📄 Generating report"
                    }
                    
                    final_answer = ""
                    current_tool_step = None
                    
                    async for event in agent.astream_events(
                        {"input": message.content, "chat_history": chat_history},
                        version="v2"
                    ):
                        event_type = event.get("event")
                        
                        if event_type == "on_tool_start":
                            tool_name = event.get("name", "tool")
                            step_name = tool_descriptions.get(tool_name, f"🔧 {tool_name}")
                            current_tool_step = cl.Step(name=step_name)
                            await current_tool_step.__aenter__()
                        
                        elif event_type == "on_tool_end":
                            if current_tool_step:
                                tool_output = event.get("data", {}).get("output", "")
                                current_tool_step.output = tool_output[:200] + "..." if len(tool_output) > 200 else tool_output
                                await current_tool_step.__aexit__(None, None, None)
                                current_tool_step = None
                        
                        elif event_type == "on_chat_model_stream":
                            chunk = event.get("data", {}).get("chunk")
                            if chunk and hasattr(chunk, "content") and chunk.content:
                                final_answer += chunk.content
                                await msg.stream_token(chunk.content)
                    
                    if not final_answer:
                        final_answer = "I processed your tax filing request."
                    
                    # Handle report generation path
                    import re
                    report_match = re.search(r'REPORT_PATH:(.+?):END_REPORT_PATH', final_answer)
                    if report_match:
                        report_path = report_match.group(1).strip()
                        final_answer = re.sub(r'REPORT_PATH:.+?:END_REPORT_PATH', '', final_answer).strip()
                        
                        # Add action to open report
                        actions = [
                            cl.Action(
                                name="open_report",
                                payload={"path": report_path},
                                label="📄 Open Tax Report"
                            )
                        ]
                        msg.actions = actions
                    
                    msg.content = final_answer
                    await msg.update()
                    
                    # Update chat history
                    chat_history.append(HumanMessage(content=message.content))
                    chat_history.append(AIMessage(content=final_answer))
                    if len(chat_history) > 20:
                        chat_history = chat_history[-20:]
                    cl.user_session.set("chat_history", chat_history)
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    msg.content = f"❌ Error with tax filing agent: {str(e)}"
                    await msg.update()
            
            else:
                # Route complex queries to the regular Tax Agent
                try:
                    from services.tax_agent import get_tax_agent
                    from langchain_core.messages import HumanMessage, AIMessage
                    
                    agent = get_tax_agent()
                    if agent is None:
                        msg.content = "❌ Tax agent is not available. Please check your OpenAI API key."
                        await msg.update()
                        return
                    
                    # Get chat history from session
                    chat_history = cl.user_session.get("chat_history", [])
                    
                    # Tool descriptions for step display
                    tool_descriptions = {
                        "calculate_totals": "📊 Calculating totals",
                        "estimate_tax": "💰 Estimating taxes",
                        "research_tax_query": "🔍 Searching documents and tax laws",
                        "get_summary": "📋 Summarizing documents",
                        "update_field": "✏️ Updating field",
                        "tax_tips": "💡 Looking up tax tips"
                    }
                    
                    # Stream tokens using astream_events
                    final_answer = ""
                    current_tool_step = None
                    
                    async for event in agent.astream_events(
                        {
                            "input": message.content,
                            "chat_history": chat_history
                        },
                        version="v2"
                    ):
                        event_type = event.get("event")
                        
                        # Handle tool start - show as step
                        if event_type == "on_tool_start":
                            tool_name = event.get("name", "tool")
                            tool_input = event.get("data", {}).get("input", "")
                            description = tool_descriptions.get(tool_name, f"🔧 {tool_name}")
                            
                            # Create a step for this tool
                            current_tool_step = cl.Step(name=description, type="tool")
                            current_tool_step.input = str(tool_input) if tool_input else ""
                            await current_tool_step.send()
                        
                        # Handle tool end
                        elif event_type == "on_tool_end":
                            if current_tool_step:
                                tool_output = event.get("data", {}).get("output", "")
                                obs_str = str(tool_output)
                                current_tool_step.output = obs_str[:500] + "..." if len(obs_str) > 500 else obs_str
                                await current_tool_step.update()
                                current_tool_step = None
                        
                        # Stream tokens from the final LLM response
                        elif event_type == "on_chat_model_stream":
                            chunk = event.get("data", {}).get("chunk")
                            if chunk and hasattr(chunk, "content") and chunk.content:
                                # Only stream text content (not tool calls)
                                if isinstance(chunk.content, str):
                                    final_answer += chunk.content
                                    await msg.stream_token(chunk.content)
                    
                    # Finalize the message
                    if not final_answer:
                        final_answer = "I processed your request."
                    
                    # Strip broken markdown images that the LLM generates (e.g., ![text](sandbox:/...))
                    import re
                    final_answer = re.sub(r'!\[.*?\]\(sandbox:[^)]+\)', '', final_answer)
                    # Also clean up any double newlines left behind
                    final_answer = re.sub(r'\n{3,}', '\n\n', final_answer).strip()
                    
                    # Update message with text
                    msg.content = final_answer
                    await msg.update()
                    
                    # Check if a proof image was generated (via global variable, not LLM output)
                    from services.tax_agent import get_last_proof_image_path
                    proof_image_path = get_last_proof_image_path()
                    
                    # Send proof image as a separate message if one was generated
                    if proof_image_path:
                        from pathlib import Path
                        proof_path_obj = Path(proof_image_path)
                        if proof_path_obj.exists():
                            # Read image as bytes (more reliable than path)
                            with open(proof_path_obj, "rb") as f:
                                image_bytes = f.read()
                            
                            # Create image element with content (bytes)
                            proof_image = cl.Image(
                                content=image_bytes,
                                name=proof_path_obj.stem,
                                display="inline",
                                size="large",
                                mime="image/png"
                            )
                            
                            # Send as separate message with the image
                            img_msg = cl.Message(
                                content="📷 **Visual Evidence from Document:**",
                                elements=[proof_image]
                            )
                            await img_msg.send()
                    
                    # Update chat history with this exchange
                    chat_history.append(HumanMessage(content=message.content))
                    chat_history.append(AIMessage(content=final_answer))
                    
                    # Keep only last 20 messages (10 exchanges) to manage context window
                    if len(chat_history) > 20:
                        chat_history = chat_history[-20:]
                    
                    cl.user_session.set("chat_history", chat_history)
                
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    msg.content = f"❌ Error running tax agent: {str(e)}"
                    await msg.update()


async def handle_file_uploads(elements: List[Element]):
    """Process uploaded files with LandingAI and store in ChromaDB."""
    
    documents = cl.user_session.get("documents") or []
    uploaded_count = 0
    
    # Get vector store
    try:
        vector_store = get_vector_store()
    except Exception as e:
        await cl.Message(content=f"⚠️ **Vector store error:** {str(e)}").send()
        vector_store = None
    
    for file in elements:
        # Check if file type is supported
        ext = Path(file.name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            await cl.Message(
                content=f"⚠️ **Unsupported file type:** `{file.name}`\n\nPlease upload PDF or image files (PNG, JPG, GIF, WebP)."
            ).send()
            continue
        
        # Step 1: Save the file
        async with cl.Step(name="Saving Document", type="tool") as step:
            step.input = f"Uploading: {file.name}"
            file_metadata = await save_uploaded_file(file)
            step.output = f"✅ Saved: {file.name} ({format_file_size(file_metadata['size'])})"
        
        # Step 2: Check if already processed (deduplication)
        from_cache = False
        cached_result = None
        
        async with cl.Step(name="Checking Knowledge Base", type="tool") as step:
            step.input = f"Hash: {file_metadata['file_hash'][:12]}..."
            
            if vector_store and vector_store.document_exists(file_metadata['file_hash']):
                cached_result = vector_store.get_document(file_metadata['file_hash'])
                
                # Check if cached result has the new bounding_box data
                has_bboxes = False
                if cached_result:
                    extracted = cached_result.get("extracted_fields", {})
                    if extracted and isinstance(extracted, dict):
                        # Check the first value to see if it has bounding_boxes
                        try:
                            first_val = next(iter(extracted.values()), {})
                            if isinstance(first_val, dict) and "bounding_boxes" in first_val:
                                has_bboxes = True
                        except StopIteration:
                            # Empty fields, treat as having no boxes (or valid empty doc)
                            pass
                
                if has_bboxes:
                    from_cache = True
                    step.output = "📦 Found in cache! Skipping API call."
                else:
                    from_cache = False
                    step.output = "🔄 Found in cache but missing visualization data. Re-processing..."
            else:
                step.output = "🆕 New document, will process with AI."
        
        # Step 3: Process document
        if from_cache and cached_result:
            # Use cached result
            processing_result = {
                "document_type": cached_result["document_type"],
                "confidence_score": cached_result["overall_confidence"],
                "extracted_fields": cached_result["extracted_fields"],
                "needs_verification": cached_result["needs_verification"],
                "markdown_content": cached_result["markdown_content"],
            }
            file_metadata["processed"] = True
            file_metadata["confidence_score"] = cached_result["overall_confidence"]
            file_metadata["document_type"] = cached_result["document_type"]
            file_metadata["from_cache"] = True
            
            # Update vector store with the new saved_path (for existing cached docs)
            if vector_store and not cached_result.get("saved_path"):
                vector_store.store_document(
                    file_hash=file_metadata["file_hash"],
                    original_filename=file.name,
                    document_type=cached_result["document_type"],
                    markdown_content=cached_result["markdown_content"],
                    extracted_fields=cached_result["extracted_fields"],
                    raw_chunks=cached_result.get("raw_chunks", []),
                    overall_confidence=cached_result["overall_confidence"],
                    needs_verification=cached_result["needs_verification"],
                    saved_path=file_metadata["path"],  # Update with new saved path
                )
            
        else:
            # Parse with LandingAI
            async with cl.Step(name="AI Document Analysis (LandingAI DPT-2)", type="llm") as step:
                step.input = f"Analyzing: {file.name}"
                
                try:
                    # Import sync version for executor
                    from services.document_parser import parse_document_sync
                    
                    # Run parsing in executor to avoid blocking
                    parsed_doc = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: parse_document_sync(file_metadata["path"], file.name)
                    )
                    
                    processing_result = {
                        "document_type": parsed_doc.document_type,
                        "confidence_score": parsed_doc.overall_confidence,
                        "extracted_fields": {
                            k: {
                                "value": v.value, 
                                "confidence": v.confidence,
                                "field_name": k,
                                "bounding_boxes": v.bounding_boxes
                            }
                            for k, v in parsed_doc.extracted_fields.items()
                        },
                        "needs_verification": parsed_doc.needs_verification,
                        "markdown_content": parsed_doc.markdown_content,
                    }
                    
                    file_metadata["processed"] = True
                    file_metadata["confidence_score"] = parsed_doc.overall_confidence
                    file_metadata["document_type"] = parsed_doc.document_type
                    file_metadata["from_cache"] = False
                    
                    # Store in vector store
                    if vector_store:
                        vector_store.store_document(
                            file_hash=parsed_doc.file_hash,
                            original_filename=file.name,
                            document_type=parsed_doc.document_type,
                            markdown_content=parsed_doc.markdown_content,
                            extracted_fields=processing_result["extracted_fields"],
                            raw_chunks=parsed_doc.raw_chunks,
                            overall_confidence=parsed_doc.overall_confidence,
                            needs_verification=parsed_doc.needs_verification,
                            saved_path=file_metadata["path"],  # Actual file path in uploads/
                        )
                    
                    step.output = f"✅ Analysis complete ({parsed_doc.overall_confidence * 100:.0f}% confidence)"
                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"PARSING ERROR: {error_details}")
                    step.output = f"❌ Error: {str(e)}\n\nDetails: {error_details[:500]}"
                    processing_result = {
                        "document_type": f"Unknown (Parse Error: {str(e)[:50]})",
                        "confidence_score": 0.0,
                        "extracted_fields": {},
                        "needs_verification": True,
                        "markdown_content": f"Error: {str(e)}\n\n{error_details}",
                    }
                    file_metadata["processed"] = False
                    file_metadata["confidence_score"] = 0.0
        
        # Store document metadata in session
        file_metadata["processing_result"] = processing_result
        documents.append(file_metadata)
        uploaded_count += 1
        
        # Create preview elements (with annotations for low confidence fields)
        preview_elements = []
        if file_metadata["type"] == "Image":
            # Check for low confidence fields that need highlighting
            display_path = file_metadata["path"]
            
            if processing_result.get("needs_verification") or any(
                f.get("confidence", 1.0) < 0.85 
                for f in processing_result.get("extracted_fields", {}).values()
            ):
                # Generate annotated image with bounding boxes
                try:
                    from services.image_annotator import annotate_document_image, get_low_confidence_fields
                    
                    low_conf_fields = get_low_confidence_fields(processing_result.get("extracted_fields", {}))
                    
                    if low_conf_fields and any(f.get("bounding_boxes") for f in low_conf_fields):
                        annotated_path = annotate_document_image(
                            file_metadata["path"],
                            low_conf_fields,
                            show_labels=True
                        )
                        display_path = annotated_path
                except Exception as e:
                    print(f"Warning: Could not annotate image: {e}")
            
            preview_elements.append(
                cl.Image(
                    name=file_metadata["original_name"],
                    path=display_path,
                    display="inline",
                    size="large"
                )
            )
        elif file_metadata["type"] == "PDF Document":
            preview_elements.append(
                cl.Pdf(
                    name=file_metadata["original_name"],
                    path=file_metadata["path"],
                    display="inline"
                )
            )
        
        # Build result message
        confidence_pct = processing_result["confidence_score"] * 100
        
        if processing_result["needs_verification"]:
            status_icon = "⚠️"
            status_text = "**Low Confidence - Verification Needed**"
        else:
            status_icon = "✅"
            status_text = "**High Confidence**"
        
        cache_indicator = " 📦 *from cache*" if from_cache else ""
        
        result_message = f"""## {status_icon} Document Processed: {file_metadata['original_name']}{cache_indicator}

{status_text}

---

### 📊 Extracted Information

| Field | Value | Confidence |
|-------|-------|------------|
| **Document Type** | {processing_result['document_type']} | - |
| **Overall Confidence** | {confidence_pct:.0f}% | - |
"""
        
        # Add extracted fields to table
        for field_name, field_data in processing_result.get("extracted_fields", {}).items():
            field_label = field_name.replace("_", " ").title()
            value = field_data.get("value", "N/A")
            conf = field_data.get("confidence", 0) * 100
            conf_indicator = "🟢" if conf >= 85 else "🟡" if conf >= 70 else "🔴"
            result_message += f"| **{field_label}** | {value} | {conf_indicator} {conf:.0f}% |\n"
        
        # === VERIFICATION STEP ===
        # Verify document against user profile (from 2nd document onwards)
        verification_message = ""
        try:
            from services.user_profile import get_profile_service
            profile_service = get_profile_service()
            
            verification_result = profile_service.verify_document(
                extracted_fields=processing_result.get("extracted_fields", {}),
                source_document=file_metadata["original_name"]
            )
            
            file_metadata["verification_result"] = verification_result
            
            if verification_result["is_first_document"]:
                verification_message = f"""
---

### 🆕 User Profile Created

A new profile has been created based on this document:
- **Name:** {verification_result['profile'].get('taxpayer_name', 'Not extracted')}
- **Address:** {verification_result['profile'].get('mailing_address', 'Not extracted')}

_Future documents will be verified against this profile._
"""
            elif verification_result["status"] == "verified":
                verification_message = """
---

### ✅ Identity Verified

Document details match your existing profile.
"""
            elif verification_result["status"] == "discrepancy":
                discrepancy_details = ""
                for d in verification_result["discrepancies"]:
                    field_label = d["field"].replace("_", " ").title()
                    similarity_pct = d["similarity"] * 100
                    discrepancy_details += f"- **{field_label}**: Expected '{d['expected']}' but found '{d['found']}' ({similarity_pct:.0f}% match)\n"
                
                verification_message = f"""
---

### ⚠️ Identity Discrepancy Detected

The following fields don't match your existing profile:

{discrepancy_details}
> Please verify this document belongs to the same taxpayer. If correct, the profile has been updated.
"""
        except Exception as e:
            print(f"Verification error: {e}")
            # Continue without verification feedback
        
        result_message += verification_message
        
        if processing_result["needs_verification"]:
            result_message += """
---

> ⚠️ **Attention Required:** Some extracted data has low confidence. Please review the values above and let me know if any corrections are needed.
"""
        
        await cl.Message(
            content=result_message,
            elements=preview_elements
        ).send()
    
    # Update session
    cl.user_session.set("documents", documents)
    
    # Summary if multiple files
    if uploaded_count > 1:
        await cl.Message(
            content=f"📁 **{uploaded_count} document(s) processed successfully!**\n\nType \"show documents\" or click the button below to see all your uploads.",
            actions=[
                cl.Action(
                    name="view_documents",
                    payload={"action": "view"},
                    label="📂 View All Documents"
                )
            ]
        ).send()
    elif uploaded_count == 1:
        await cl.Message(
            content="Upload more documents or type \"show documents\" to see your collection.",
            actions=[
                cl.Action(
                    name="view_documents",
                    payload={"action": "view"},
                    label="📂 View All Documents"
                )
            ]
        ).send()


if __name__ == "__main__":
    # This allows running with: python app.py (for debugging)
    # Normal usage: chainlit run app.py
    pass
