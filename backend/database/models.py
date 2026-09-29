"""Database tables (guide section 16) plus the extra fields the pipeline needs."""
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from backend.database.db import Base


def _now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now)
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"
    document_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    stored_path = Column(String(500), nullable=False)
    upload_date = Column(DateTime(timezone=True), default=_now)
    document_type = Column(String(40), default="auto")
    detected_type = Column(String(40))
    status = Column(String(30), default="uploaded")  # uploaded|processing|completed|failed
    current_step = Column(String(80))
    progress = Column(Integer, default=0)
    error = Column(Text)
    extraction_method = Column(String(80))
    page_count = Column(Integer)
    char_count = Column(Integer)
    full_text = Column(Text)
    overall_summary = Column(Text)
    summary_method = Column(String(120))
    recommendations = Column(JSON)
    analysis_meta = Column(JSON)
    processed_at = Column(DateTime(timezone=True))
    user = relationship("User", back_populates="documents")
    clauses = relationship("Clause", back_populates="document", cascade="all, delete-orphan",
                           order_by="Clause.clause_number")
    conversations = relationship("Conversation", back_populates="document", cascade="all, delete-orphan",
                                 order_by="Conversation.timestamp")
    persona_summaries = relationship("PersonaSummary", back_populates="document", cascade="all, delete-orphan")
    comparisons = relationship("ComparisonRow", back_populates="document", cascade="all, delete-orphan",
                               order_by="ComparisonRow.position")


class Clause(Base):
    __tablename__ = "clauses"
    clause_id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.document_id"), nullable=False, index=True)
    clause_number = Column(Integer, nullable=False)
    label = Column(String(40))           # numbering as printed in the document, e.g. "5" or "IV"
    title = Column(String(255))
    clause_type = Column(String(40))     # category
    category_confidence = Column(Float)
    classifier_details = Column(JSON)
    original_text = Column(Text, nullable=False)
    summary = Column(Text)               # simple explanation
    summary_method = Column(String(120))
    risk_level = Column(String(10))
    risk_reason = Column(Text)
    precaution = Column(Text)
    risk_evidence = Column(JSON)
    risk_method = Column(String(120))
    references = Column(JSON)            # RAG chunks used
    standard_match = Column(JSON)
    document = relationship("Document", back_populates="clauses")


class Conversation(Base):
    __tablename__ = "conversations"
    conversation_id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.document_id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources = Column(JSON)
    method = Column(String(120))
    timestamp = Column(DateTime(timezone=True), default=_now)
    document = relationship("Document", back_populates="conversations")


class PersonaSummary(Base):
    __tablename__ = "persona_summaries"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.document_id"), nullable=False, index=True)
    persona = Column(String(30), nullable=False)
    summary = Column(Text, nullable=False)
    method = Column(String(120))
    created_at = Column(DateTime(timezone=True), default=_now)
    document = relationship("Document", back_populates="persona_summaries")


class ComparisonRow(Base):
    __tablename__ = "comparisons"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.document_id"), nullable=False, index=True)
    position = Column(Integer, default=0)
    category = Column(String(40))
    status = Column(String(20))          # equivalent | differs | missing | additional
    standard_title = Column(String(255))
    standard_text = Column(Text)
    user_clause_number = Column(Integer)
    user_clause_label = Column(String(40))
    user_title = Column(String(255))
    user_text = Column(Text)
    similarity = Column(Float)
    standard_value = Column(String(255))
    user_value = Column(String(255))
    difference = Column(Text)
    method = Column(String(120))
    document = relationship("Document", back_populates="comparisons")
