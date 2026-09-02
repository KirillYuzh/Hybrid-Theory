from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from typing import Any, Dict, List, Optional
from jsonschema import validate, Draft7Validator
from jsonschema.exceptions import ValidationError
import hashlib
import time
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    TRANSACTION_CREATED = "TransactionCreated"
    RISK_SCORE_COMPUTED = "RiskScoreComputed"
    DECISION_MADE = "DecisionMade"


class SchemaVersion:
    """Класс для управления версиями схем с audit trail"""

    def __init__(self, version: str, schema: Dict[str, Any], description: str, created_by: str):
        self.version = version
        self.schema = schema
        self.description = description
        self.created_by = created_by
        self.created_at = datetime.utcnow().isoformat()
        self.schema_hash = self._calculate_hash(schema)

    def _calculate_hash(self, schema: Dict[str, Any]) -> str:
        """Вычисляет хэш-сумму схемы для контроля изменений"""
        schema_str = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()

    def is_compatible(self, other_version: 'SchemaVersion') -> bool:
        """Проверяет совместимость версий схем"""
        return self.schema_hash == other_version.schema_hash

    def to_dict(self) -> Dict[str, Any]:
        """Возвращает версию в формате словаря"""
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
        """Создает версию схемы из словаря"""
        return cls(
            version=data["version"],
            schema=data["schema"],
            description=data["description"],
            created_by=data["created_by"],
        )


class SchemaRegistry:
    """Реестр версионированных схем с audit trail"""

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
        """Регистрирует новую версию схемы"""
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

        logger.info(f"Схема зарегистрирована: {event_type.value} v{version}")
        return new_version

    def get_latest_schema(self, event_type: EventType) -> Optional[SchemaVersion]:
        """Возвращает последнюю зарегистрированную версию схемы"""
        if event_type not in self._schemas:
            return None
        return self._schemas[event_type][-1]

    def get_schema_version(self, event_type: EventType, version: str) -> Optional[SchemaVersion]:
        """Возвращает схему по конкретной версии"""
        if event_type not in self._schemas:
            return None
        for schema_version in self._schemas[event_type]:
            if schema_version.version == version:
                return schema_version
        return None

    def get_all_versions(self, event_type: EventType) -> List[SchemaVersion]:
        """Возвращает все версии схемы"""
        if event_type not in self._schemas:
            return []
        return self._schemas[event_type].copy()

    def validate_event(self, event_type: EventType, event_data: Dict[str, Any]) -> bool:
        """Валидирует событие по последней версии схемы"""
        schema_version = self.get_latest_schema(event_type)
        if not schema_version:
            logger.warning(f"Схема не найдена для {event_type.value}")
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
            logger.info(f"Событие валидировано: {event_type.value}")
            return True
        except ValidationError as e:
            logger.error(f"Ошибка валидации события {event_type.value}: {e}")
            self._log_audit(
                action="validation_error",
                event_type=event_type,
                version=schema_version.version,
                error=str(e),
            )
            return False

    def _calculate_event_hash(self, event_data: Dict[str, Any]) -> str:
        """Вычисляет хэш-сумму события"""
        event_str = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()

    def _log_audit(self, action: str, event_type: EventType, version: str, **kwargs) -> None:
        """Записывает запись в audit log"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "event_type": event_type.value,
            "version": version,
            **kwargs,
        }
        self._audit_log.append(audit_entry)
        logger.debug(f"Запись в audit log: {audit_entry}")

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Возвращает audit log"""
        return self._audit_log.copy()


# Инициализация реестра схем
_registry = SchemaRegistry()


# JSON Schema контракты для событий Kafka
TRANSACTION_CREATED_SCHEMA_V1 = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["event_type", "transaction_id", "timestamp", "amount", "currency", "status"],
    "properties": {
        "event_type": {
            "type": "string",
            "const": "TransactionCreated",
            "description": "Тип события",
        },
        "transaction_id": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]{36}$",
            "description": "Уникальный идентификатор транзакции",
        },
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "Время создания транзакции",
        },
        "amount": {
            "type": "number",
            "minimum": 0,
            "description": "Сумма транзакции",
        },
        "currency": {
            "type": "string",
            "pattern": "^[A-Z]{3}$",
            "description": "Валюта транзакции (ISO 4217)",
        },
        "status": {
            "type": "string",
            "enum": ["pending", "completed", "failed"],
            "description": "Статус транзакции",
        },
        "metadata": {
            "type": "object",
            "additionalProperties": True,
            "description": "Дополнительные метаданные",
        },
    },
    "description": "Схема события создания транзакции",
}

RISK_SCORE_COMPUTED_SCHEMA_V1 = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["event_type", "transaction_id", "timestamp", "risk_score", "risk_level"],
    "properties": {
        "event_type": {
            "type": "string",
            "const": "RiskScoreComputed",
            "description": "Тип события",
        },
        "transaction_id": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]{36}$",
            "description": "Уникальный идентификатор транзакции",
        },
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "Время вычисления риск-скора",
        },
        "risk_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": "Вычисленный риск-скор",
        },
        "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
            "description": "Уровень риска",
        },
        "factors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Факторы, повлиявшие на риск-скор",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Уверенность в расчете",
        },
    },
    "description": "Схема события вычисления риск-скора",
}

DECISION_MADE_SCHEMA_V1 = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["event_type", "transaction_id", "timestamp", "decision", "status"],
    "properties": {
        "event_type": {
            "type": "string",
            "const": "DecisionMade",
            "description": "Тип события",
        },
        "transaction_id": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]{36}$",
            "description": "Уникальный идентификатор транзакции",
        },
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "Время принятия решения",
        },
        "decision": {
            "type": "string",
            "enum": ["approve", "decline", "review"],
            "description": "Принятое решение",
        },
        "status": {
            "type": "string",
            "enum": ["pending", "completed", "failed"],
            "description": "Статус решения",
        },
        "reason": {
            "type": "string",
            "description": "Причина принятия решения",
        },
        "approver": {
            "type": "string",
            "description": "Идентификатор менеджера, принявшего решение",
        },
    },
    "description": "Схема события принятия решения",
}


# Регистрация базовых версий схем
_registry.register_schema(
    EventType.TRANSACTION_CREATED,
    "v1.0",
    TRANSACTION_CREATED_SCHEMA_V1,
    "Первая версия схемы события создания транзакции",
    "system",
)

_registry.register_schema(
    EventType.RISK_SCORE_COMPUTED,
    "v1.0",
    RISK_SCORE_COMPUTED_SCHEMA_V1,
    "Первая версия схемы события вычисления риск-скора",
    "system",
)

_registry.register_schema(
    EventType.DECISION_MADE,
    "v1.0",
    DECISION_MADE_SCHEMA_V1,
    "Первая версия схемы события принятия решения",
    "system",
)


@dataclass
class TransactionCreatedEvent:
    """Событие создания транзакции"""
    transaction_id: str
    timestamp: str
    amount: float
    currency: str
    status: str
    event_type: str = "TransactionCreated"
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransactionCreatedEvent':
        """Создает событие из словаря"""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Возвращает событие в формате словаря"""
        return {
            "event_type": self.event_type,
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp,
            "amount": self.amount,
            "currency": self.currency,
            "status": self.status,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Сериализует событие в JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'TransactionCreatedEvent':
        """Десериализует событие из JSON"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate(self) -> bool:
        """Валидирует событие"""
        return _registry.validate_event(EventType.TRANSACTION_CREATED, self.to_dict())


@dataclass
class RiskScoreComputedEvent:
    """Событие вычисления риск-скора"""
    transaction_id: str
    timestamp: str
    risk_score: float
    risk_level: str
    event_type: str = "RiskScoreComputed"
    factors: Optional[List[str]] = field(default_factory=list)
    confidence: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskScoreComputedEvent':
        """Создает событие из словаря"""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Возвращает событие в формате словаря"""
        return {
            "event_type": self.event_type,
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "factors": self.factors,
            "confidence": self.confidence,
        }

    def to_json(self) -> str:
        """Сериализует событие в JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'RiskScoreComputedEvent':
        """Десериализует событие из JSON"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate(self) -> bool:
        """Валидирует событие"""
        return _registry.validate_event(EventType.RISK_SCORE_COMPUTED, self.to_dict())


@dataclass
class DecisionMadeEvent:
    """Событие принятия решения"""
    transaction_id: str
    timestamp: str
    decision: str
    status: str
    event_type: str = "DecisionMade"
    reason: Optional[str] = None
    approver: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionMadeEvent':
        """Создает событие из словаря"""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Возвращает событие в формате словаря"""
        return {
            "event_type": self.event_type,
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp,
            "decision": self.decision,
            "status": self.status,
            "reason": self.reason,
            "approver": self.approver,
        }

    def to_json(self) -> str:
        """Сериализует событие в JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'DecisionMadeEvent':
        """Десериализует событие из JSON"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate(self) -> bool:
        """Валидирует событие"""
        return _registry.validate_event(EventType.DECISION_MADE, self.to_dict())


class EventSerializer:
    """Сериализатор/десериализатор событий"""

    @staticmethod
    def serialize(event: Any) -> str:
        """Сериализует событие в строку"""
        if not isinstance(event, (TransactionCreatedEvent, RiskScoreComputedEvent, DecisionMadeEvent)):
            raise ValueError(f"Неподдерживаемый тип события: {type(event)}")

        if not event.validate():
            raise ValueError(f"Событие {type(event).__name__} не прошло валидацию")

        return event.to_json()

    @staticmethod
    def deserialize(event_type: str, json_str: str) -> Any:
        """Десериализует строку в событие"""
        event_type_enum = EventType(event_type)

        if event_type_enum == EventType.TRANSACTION_CREATED:
            return TransactionCreatedEvent.from_json(json_str)
        elif event_type_enum == EventType.RISK_SCORE_COMPUTED:
            return RiskScoreComputedEvent.from_json(json_str)
        elif event_type_enum == EventType.DECISION_MADE:
            return DecisionMadeEvent.from_json(json_str)
        else:
            raise ValueError(f"Неизвестный тип события: {event_type}")

    @staticmethod
    def deserialize_by_event(event: Dict[str, Any]) -> Any:
        """Десериализует событие из словаря по полю event_type"""
        event_type_str = event.get("event_type")
        if not event_type_str:
            raise ValueError("Поле 'event_type' отсутствует в событии")

        json_str = json.dumps(event)
        return EventSerializer.deserialize(event_type_str, json_str)


# Экспорт основных компонентов
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