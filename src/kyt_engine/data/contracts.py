from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
from typing import Any, Dict, List, Optional
from jsonschema import validate, Draft7Validator
from jsonschema.exceptions import ValidationError
import hashlib
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    TRANSACTION_CREATED = "TransactionCreated"
    RISK_SCORE_COMPUTED = "RiskScoreComputed"
    DECISION_MADE = "DecisionMade"


class SchemaVersion:
    def __init__(self, version: str, schema: Dict[str, Any], description: str, created_by: str):
        self.version = version
        self.schema = schema
        self.description = description
        self.created_by = created_by
        self.created_at = datetime.now(datetime.timezone.utc).isoformat()
        self.schema_hash = self._calculate_hash(schema)

    def _calculate_hash(self, schema: Dict[str, Any]) -> str:
        schema_str = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()

    def is_compatible(self, other_version: 'SchemaVersion') -> bool:
        return self.schema_hash == other_version.schema_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "schema": self.schema,
            "description": self.description,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "schema_hash": self.schema_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaVersion':
        return cls(
            version=data["version"],
            schema=data["schema"],
            description=data["description"],
            created_by=data["created_by"],
        )


class SchemaRegistry:
    def __init__(self):
        self._schemas: Dict[EventType, List[SchemaVersion]] = {}
        self._audit_log: List[Dict[str, Any]] = []

    def register_schema(
        self,
        event_type: EventType,
        version: str,
        schema: Dict[str, Any],
        description: str,
        created_by: str,
    ) -> SchemaVersion:
        new_version = SchemaVersion(version, schema, description, created_by)

        if event_type not in self._schemas:
            self._schemas[event_type] = []

        self._schemas[event_type].append(new_version)
        self._log_audit(
            action="register_schema",
            event_type=event_type,
            version=version,
            schema_hash=new_version.schema_hash,
            created_by=created_by,
        )

        logger.info(f"Schema registered: {event_type.value} v{version}")
        return new_version

    def get_latest_schema(self, event_type: EventType) -> Optional[SchemaVersion]:
        if event_type not in self._schemas:
            return None
        return self._schemas[event_type][-1]

    def get_schema_version(self, event_type: EventType, version: str) -> Optional[SchemaVersion]:
        if event_type not in self._schemas:
            return None
        for schema_version in self._schemas[event_type]:
            if schema_version.version == version:
                return schema_version
        return None

    def get_all_versions(self, event_type: EventType) -> List[SchemaVersion]:
        if event_type not in self._schemas:
            return []
        return self._schemas[event_type].copy()

    def validate_event(self, event_type: EventType, event_data: Dict[str, Any]) -> bool:
        schema_version = self.get_latest_schema(event_type)
        if not schema_version:
            logger.warning(f"Schema not found for {event_type.value}")
            return False

        try:
            Draft7Validator.check_schema(schema_version.schema)
            validate(instance=event_data, schema=schema_version.schema)
            self._log_audit(
                action="validate_event",
                event_type=event_type,
                version=schema_version.version,
                event_hash=self._calculate_event_hash(event_data),
            )
            logger.info(f"Event validated: {event_type.value}")
            return True
        except ValidationError as e:
            logger.error(f"Event validation error {event_type.value}: {e}")
            self._log_audit(
                action="validation_error",
                event_type=event_type,
                version=schema_version.version,
                error=str(e),
            )
            return False

    def _calculate_event_hash(self, event_data: Dict[str, Any]) -> str:
        event_str = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()

    def _log_audit(self, action: str, event_type: EventType, version: str, **kwargs) -> None:
        audit_entry = {
            "timestamp": datetime.now(datetime.timezone.utc).isoformat(),
            "action": action,
            "event_type": event_type.value,
            "version": version,
            **kwargs,
        }
        self._audit_log.append(audit_entry)
        logger.debug(f"Audit log entry: {audit_entry}")

    def get_audit_log(self) -> List[Dict[str, Any]]:
        return self._audit_log.copy()


_registry = SchemaRegistry()


TRANSACTION_CREATED_SCHEMA_V1 = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["event_type", "transaction_id", "timestamp", "amount", "currency", "status"],
    "properties": {
        "event_type": {"type": "string", "const": "TransactionCreated"},
        "transaction_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]{36}$"},
        "timestamp": {"type": "string", "format": "date-time"},
        "amount": {"type": "number", "minimum": 0},
        "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
        "status": {"type": "string", "enum": ["pending", "completed", "failed"]},
        "metadata": {"type": "object", "additionalProperties": True},
    },
}

RISK_SCORE_COMPUTED_SCHEMA_V1 = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["event_type", "transaction_id", "timestamp", "risk_score", "risk_level"],
    "properties": {
        "event_type": {"type": "string", "const": "RiskScoreComputed"},
        "transaction_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]{36}$"},
        "timestamp": {"type": "string", "format": "date-time"},
        "risk_score": {"type": "number", "minimum": 0, "maximum": 100},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "factors": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

DECISION_MADE_SCHEMA_V1 = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["event_type", "transaction_id", "timestamp", "decision", "status"],
    "properties": {
        "event_type": {"type": "string", "const": "DecisionMade"},
        "transaction_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]{36}$"},
        "timestamp": {"type": "string", "format": "date-time"},
        "decision": {"type": "string", "enum": ["approve", "decline", "review"]},
        "status": {"type": "string", "enum": ["pending", "completed", "failed"]},
        "reason": {"type": "string"},
        "approver": {"type": "string"},
    },
}


def _make_event_class(event_type: EventType, schema: Dict, required_fields: List[str], optional_fields: Dict[str, Any]):
    et = event_type.value
    @dataclass
    class Event:
        event_type: str = field(default=et)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Event':
            return cls(**data)

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)

        def to_json(self) -> str:
            return json.dumps(self.to_dict(), ensure_ascii=False)

        @classmethod
        def from_json(cls, json_str: str) -> 'Event':
            return cls.from_dict(json.loads(json_str))

        def validate(self) -> bool:
            return _registry.validate_event(event_type, self.to_dict())

    for field_name in required_fields:
        setattr(Event, field_name, field())
    for field_name, default in optional_fields.items():
        setattr(Event, field_name, field(default=default))

    Event.__name__ = et
    Event.__qualname__ = et
    return Event


TransactionCreatedEvent = _make_event_class(
    EventType.TRANSACTION_CREATED,
    TRANSACTION_CREATED_SCHEMA_V1,
    ["transaction_id", "timestamp", "amount", "currency", "status"],
    {"metadata": field(default_factory=dict)},
)

RiskScoreComputedEvent = _make_event_class(
    EventType.RISK_SCORE_COMPUTED,
    RISK_SCORE_COMPUTED_SCHEMA_V1,
    ["transaction_id", "timestamp", "risk_score", "risk_level"],
    {"factors": field(default_factory=list), "confidence": field(default=None)},
)

DecisionMadeEvent = _make_event_class(
    EventType.DECISION_MADE,
    DECISION_MADE_SCHEMA_V1,
    ["transaction_id", "timestamp", "decision", "status"],
    {"reason": field(default=None), "approver": field(default=None)},
)


_registry.register_schema(
    EventType.TRANSACTION_CREATED,
    "v1.0",
    TRANSACTION_CREATED_SCHEMA_V1,
    "Transaction creation event schema",
    "system",
)

_registry.register_schema(
    EventType.RISK_SCORE_COMPUTED,
    "v1.0",
    RISK_SCORE_COMPUTED_SCHEMA_V1,
    "Risk score computation event schema",
    "system",
)

_registry.register_schema(
    EventType.DECISION_MADE,
    "v1.0",
    DECISION_MADE_SCHEMA_V1,
    "Decision event schema",
    "system",
)


class EventSerializer:
    _EVENT_MAP = {
        EventType.TRANSACTION_CREATED: TransactionCreatedEvent,
        EventType.RISK_SCORE_COMPUTED: RiskScoreComputedEvent,
        EventType.DECISION_MADE: DecisionMadeEvent,
    }

    @staticmethod
    def serialize(event: Any) -> str:
        if type(event) not in EventSerializer._EVENT_MAP.values():
            raise ValueError(f"Unsupported event type: {type(event)}")
        if not event.validate():
            raise ValueError(f"Event {type(event).__name__} failed validation")
        return event.to_json()

    @staticmethod
    def deserialize(event_type: str, json_str: str) -> Any:
        event_type_enum = EventType(event_type)
        event_class = EventSerializer._EVENT_MAP.get(event_type_enum)
        if not event_class:
            raise ValueError(f"Unknown event type: {event_type}")
        return event_class.from_json(json_str)

    @staticmethod
    def deserialize_by_event(event: Dict[str, Any]) -> Any:
        event_type_str = event.get("event_type")
        if not event_type_str:
            raise ValueError("Field 'event_type' missing in event")
        return EventSerializer.deserialize(event_type_str, json.dumps(event))


__all__ = [
    "EventType",
    "SchemaVersion",
    "SchemaRegistry",
    "EventSerializer",
    "TransactionCreatedEvent",
    "RiskScoreComputedEvent",
    "DecisionMadeEvent",
    "TRANSACTION_CREATED_SCHEMA_V1",
    "RISK_SCORE_COMPUTED_SCHEMA_V1",
    "DECISION_MADE_SCHEMA_V1",
    "_registry",
]