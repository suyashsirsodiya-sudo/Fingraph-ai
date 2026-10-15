"""
FinGraph AI - SQLAlchemy 2.x Declarative Models & Constraints
Enforces relational integrity, check constraints, foreign keys, and indexes.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from database import Base


def utc_now():
    return datetime.now(timezone.utc)


class Article(Base):
    __tablename__ = "articles"

    id = Column(String(64), primary_key=True, index=True)
    title = Column(String(512), nullable=False)
    source_name = Column(String(256), nullable=False)
    source_url = Column(String(1024), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    retrieved_at = Column(DateTime(timezone=True), nullable=True)
    excerpt = Column(Text, nullable=True)
    fingerprint = Column(String(64), unique=True, index=True, nullable=True)
    category = Column(String(64), nullable=True, index=True)
    entities = Column(JSON, nullable=False, default=list)
    event_id = Column(String(64), nullable=True, index=True)
    # retrieval_status: 'Source Retrieved' | 'Source Unavailable' | 'Pending'
    retrieval_status = Column(String(64), nullable=False, default="Source Retrieved")
    # review_status: 'Evidence Reviewed' | 'Evidence Insufficient' | 'Pending'
    review_status = Column(String(64), nullable=False, default="Pending")
    is_demo = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class Entity(Base):
    __tablename__ = "entities"

    id = Column(String(64), primary_key=True, index=True)
    canonical_name = Column(String(256), nullable=False, index=True)
    entity_type = Column(String(64), nullable=False, index=True)
    ticker = Column(String(32), nullable=True, index=True)
    exchange = Column(String(64), nullable=True)
    sector = Column(String(128), nullable=True, index=True)
    aliases = Column(JSON, nullable=False, default=list)
    is_demo = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        CheckConstraint(
            "entity_type IN ('event', 'company', 'sector', 'commodity', 'asset', 'economic_indicator')",
            name="ck_entity_type",
        ),
    )


class Evidence(Base):
    __tablename__ = "evidence"

    id = Column(String(64), primary_key=True, index=True)
    article_id = Column(String(64), ForeignKey("articles.id", ondelete="SET NULL"), nullable=True, index=True)
    quote = Column(Text, nullable=False)
    quote_start = Column(Integer, nullable=True)
    quote_end = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class Relationship(Base):
    __tablename__ = "relationships"

    id = Column(String(64), primary_key=True, index=True)
    source_entity_id = Column(String(64), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    target_entity_id = Column(String(64), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    relationship_type = Column(String(64), nullable=False)
    mechanism = Column(String(64), nullable=False)
    direction = Column(String(32), nullable=False)
    explanation = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    weight = Column(Float, nullable=False)
    direct_or_indirect = Column(String(32), nullable=False)
    observed_or_inferred = Column(String(32), nullable=False)
    valid_from = Column(DateTime(timezone=True), nullable=True)
    valid_to = Column(DateTime(timezone=True), nullable=True)
    evidence_id = Column(String(64), ForeignKey("evidence.id", ondelete="SET NULL"), nullable=True, index=True)
    is_demo = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        # No self-loops constraint
        CheckConstraint("source_entity_id != target_entity_id", name="ck_no_self_loops"),
        # Valid confidence 0..1
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="ck_confidence_range"),
        # Valid weight 0..1
        CheckConstraint("weight >= 0.0 AND weight <= 1.0", name="ck_weight_range"),
        # Observed edge must have evidence_id
        CheckConstraint(
            "observed_or_inferred != 'observed' OR evidence_id IS NOT NULL",
            name="ck_observed_must_have_evidence",
        ),
        # Mechanism allowed values
        CheckConstraint(
            "mechanism IN ('cost', 'demand', 'supply', 'substitute', 'ownership', 'membership', 'other')",
            name="ck_mechanism_valid",
        ),
        # Direction allowed values
        CheckConstraint(
            "direction IN ('pressure', 'benefit', 'mixed', 'unclear')",
            name="ck_direction_valid",
        ),
        # Directness allowed values
        CheckConstraint(
            "direct_or_indirect IN ('direct', 'indirect')",
            name="ck_directness_valid",
        ),
        # Observed or inferred allowed values
        CheckConstraint(
            "observed_or_inferred IN ('observed', 'inferred')",
            name="ck_observed_inferred_valid",
        ),
        Index("ix_rel_source_target", "source_entity_id", "target_entity_id"),
    )


class Event(Base):
    __tablename__ = "events"

    id = Column(String(64), primary_key=True, index=True)
    title = Column(String(512), nullable=False)
    event_type = Column(String(64), nullable=False)
    description = Column(Text, nullable=True)
    event_timestamp = Column(DateTime(timezone=True), nullable=True)
    article_id = Column(String(64), ForeignKey("articles.id", ondelete="SET NULL"), nullable=True, index=True)
    entity_id = Column(String(64), ForeignKey("entities.id", ondelete="SET NULL"), nullable=True, index=True)
    extraction_status = Column(String(64), nullable=False, default="completed")
    analysis_mode = Column(String(64), nullable=False, default="rule_based_demo")
    category = Column(String(64), nullable=True, index=True)
    source_url = Column(String(1024), nullable=True)
    source_name = Column(String(256), nullable=True)
    entities = Column(JSON, nullable=False, default=list)


class Observation(Base):
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String(64), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    metric_name = Column(String(128), nullable=False)
    value = Column(Float, nullable=False)
    source_name = Column(String(128), nullable=False)

    __table_args__ = (
        Index("ix_obs_entity_time", "entity_id", "timestamp"),
    )


class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(String(64), primary_key=True, index=True)
    title = Column(String(256), nullable=False)
    source_entity_id = Column(String(64), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    pct_change = Column(Float, nullable=False)
    sector_filter = Column(String(128), nullable=True)
    horizon = Column(String(64), nullable=True)
    results = Column(JSON, nullable=False)
    is_demo = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class ProviderStatus(Base):
    __tablename__ = "provider_status"

    provider_name = Column(String(64), primary_key=True, index=True)
    is_connected = Column(Boolean, nullable=False, default=False)
    last_tested_at = Column(DateTime(timezone=True), nullable=True)
    last_success_at = Column(DateTime(timezone=True), nullable=True)
    last_attempt_at = Column(DateTime(timezone=True), nullable=True)
    records_count = Column(Integer, nullable=False, default=0)
    status_message = Column(String(256), nullable=True)
    last_error = Column(Text, nullable=True)


class Setting(Base):
    __tablename__ = "settings"

    key = Column(String(64), primary_key=True, index=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)
