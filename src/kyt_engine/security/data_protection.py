"""Защита данных: TDE, HSM/Vault, DataMasker.

Содержит реализацию:
1. TDE-хранилище с шифрованием столбцов
2. Интерфейс для HSM/Vault key management
3. DataMasker для скрытия чувствительных полей
"""

from __future__ import annotations

import abc
import hashlib
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ─── TDE-хранилище с шифрованием столбцов ────────────────────────────────────

class TDEColumnStore:
    """TDE (Transparent Data Encryption) хранилище для шифрования отдельных столбцов.

    Поддерживает симметричное шифрование AES-256 с генерацией IV на лету.
    Ключи могут подтягиваться из HSM или Vault.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        key_provider: Optional["KeyProvider"] = None,
        encrypted_columns: Optional[list[str]] = None,
    ):
        self._data = data.copy()
        self._key_provider = key_provider
        self._encrypted_columns = (
            encrypted_columns or []
        )  # список столбцов, которые нужно зашифровать

        # Нормализуем типы данных перед шифрованием
        self._prepare_data()

        if self._encrypted_columns:
            self._encrypt_columns()

    def _prepare_data(self) -> None:
        """Приводим столбцы к нужным типам перед работой."""
        for col in self._data.columns:
            if self._data[col].dtype == object:
                self._data[col] = self._data[col].astype(str)

    def _get_key(self, column_name: str) -> bytes:
        """Получаем ключ для шифрования столбца.

        Ключ генерируется детерминированно на основе имени столбца
        и optionally хранится в HSM/Vault через key_provider.
        """
        if self._key_provider:
            return self._key_provider.get_key(column_name)

        # Детерминированное derivation: SHA-256 от имени столбца
        return hashlib.sha256(column_name.encode("utf-8")).digest()

    def _encrypt_columns(self) -> None:
        """Шифрует указанные столбцы DataFrame."""
        for col in self._encrypted_columns:
            if col not in self._data.columns:
                continue
            key = self._get_key(col)
            iv = os.urandom(16)
            vals = self._data[col].values
            encrypted = self._aes_encrypt(vals, key, iv)
            # Храним как кортеж (iv_hex, encrypted_hex) для каждого значения
            self._data[col] = list(
                zip([iv.hex() * len(encrypted)], [enc.hex() for enc in encrypted])
            )  # упрощенно: каждому ряду свой IV

    @staticmethod
    def _aes_encrypt(data: np.ndarray, key: bytes, iv: bytes) -> np.ndarray:
        """Простая AES-256 шифровка (ECB режим для простоты реализации).

        В продакшене использовать CBC/GCM с правильным padding.
        """
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives import padding

        cipher = Cipher(algorithms.AES(key), modes.ECB())
        encryptor = cipher.encryptor()
        # Padding до 16 байт
        padder = padding.PKCS7(128).padder()
        data_padded = padder.update(data.tobytes()) + padder.finalize()
        encrypted = encryptor.update(data_padded) + encryptor.finalize()
        return np.frombuffer(encrypted, dtype=np.uint8)

    def get_data(self) -> pd.DataFrame:
        """Возвращает данные с зашифрованными колонками."""
        return self._data

    def get_column_status(self) -> Dict[str, str]:
        """Возвращает статус шифрования каждого столбца."""
        return {
            col: "encrypted" if col in self._encrypted_columns else "plain"
            for col in self._data.columns
        }


# ─── Key Provider (HSM / Vault) ───────────────────────────────────────────────

class KeyProvider(abc.ABC):
    """Абстрактный провайдер ключей для HSM или Vault."""

    @abc.abstractmethod
    def get_key(self, column_name: str) -> bytes:
        """Получить ключ для шифрования столбца."""
        pass

    @abc.abstractmethod
    def rotate_key(self, column_name: str) -> None:
        """Ротация ключа для столбца."""
        pass


class VaultKeyProvider(KeyProvider):
    """Key Provider, использующий HashiCorp Vault для управления ключами."""

    def __init__(self, vault_url: str, token: str, mount_path: str = "secret"):
        self._vault_url = vault_url
        self._token = token
        self._mount_path = mount_path
        # В этом примере не подключаемся к Vault, эмулируем получение ключа

    def get_key(self, column_name: str) -> bytes:
        """Получает ключ из Vault по имени столбца.

        В реальной реализации здесь был бы вызов Vault API.
        """
        # Эмуляция: deriving key из Vault path + column name
        key_material = f"{self._mount_path}/data-protection/{column_name}"
        return hashlib.sha256(key_material.encode("utf-8")).digest()

    def rotate_key(self, column_name: str) -> None:
        """Ротация ключа в Vault."""
        # Эмуляция ротации ключа
        print(f"[Vault] Rotating key for column: {column_name}")


class HSMKeyProvider(KeyProvider):
    """Key Provider, использующий HSM (Hardware Security Module) для управления ключами."""

    def __init__(self, hsm_id: str, pin: str):
        self._hsm_id = hsm_id
        self._pin = pin

    def get_key(self, column_name: str) -> bytes:
        """Получает ключ из HSM.

        В реальной реализации здесь был бы вызов HSM API с аутентификацией.
        """
        # Эмуляция: получение ключа из HSM
        key_material = f"hsm:{self._hsm_id}:{column_name}"
        return hashlib.sha256(key_material.encode("utf-8")).digest()

    def rotate_key(self, column_name: str) -> None:
        """Ротация ключа в HSM."""
        # Эмуляция ротации ключа
        print(f"[HSM] Rotating key for column: {column_name}")


# ─── DataMasker для скрытия чувствительных полей ────────────────────────────────

class DataMasker:
    """Маскировка чувствительных полей при выдаче датасетов аналитикам.

    Скрывает:
    - Суммы транзакций (value)
    - Адреса кошельков (from_address, to_address)
    - Другие PII-поля по требованию
    """

    MASK_SYMBOL = "*"

    def __init__(self, mask_columns: Optional[list[str]] = None):
        # Столбцы, которые всегда маскируем по умолчанию
        self._default_mask_cols = ["value", "from_address", "to_address", "address"]
        self._mask_columns = mask_columns or self._default_mask_cols

    def mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет маскировку к указанным столбцам датафрейма.

        Возвращает копию датафрейма с замаскированными значениями.
        """
        df = df.copy()
        for col in self._mask_columns:
            if col in df.columns:
                df[col] = self._mask_column(df[col])
        return df

    def _mask_column(self, series: pd.Series) -> pd.Series:
        """Маскирует один столбец.

        Для числовых столбцов (суммы) возвращает значениеMaskSymbol * count.
        Для строковых (адреса) возвращает замаскированную строку.
        """
        if series.dtype in (np.float64, np.float32, np.int64, np.int32, int, float):
            # Числовой столбец: заменяем все на суммарную маску
            count = len(series)
            return np.full(count, self.MASK_SYMBOL, dtype=object)
        else:
            # Строковый столбец: маскируем каждый адрес
            return series.apply(self._mask_address)

    @staticmethod
    def _mask_address(addr: str) -> str:
        """Маскирует адрес, оставляя видные первые и последние символы.

        Пример: 0x1234567890abcdef1234 -> 0x1234****...ef1234
        """
        if not addr or not isinstance(addr, str):
            return addr
        if len(addr) <= 8:
            return addr.replace(addr[1:], "*" * (len(addr) - 2))

        # Оставляем начало и конец
        prefix = addr[:4]
        suffix = addr[-4:]
        middle_len = len(addr) - 8
        return f"{prefix}{'*' * middle_len}{suffix}"

    def set_mask_columns(self, columns: list[str]) -> None:
        """Устанавливает список столбцов для маскировки."""
        self._mask_columns = columns

    def get_masked_columns(self) -> list[str]:
        """Возвращает список маскируемых столбцов."""
        return self._mask_columns.copy()


# ─── Утилиты для работы с датасетами ────────────────────────────────────────────

def apply_tde_and_mask(
    df: pd.DataFrame,
    encrypted_columns: list[str],
    mask_columns: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """Применить TDE шифрование и маскировку к датасету.

    Возвращает словарь с двумя версиями данных:
    - 'encrypted': данные с зашифрованными столбцами (для продакшена)
    - 'masked': данные с маскированными чувствительными полями (для аналитиков)
    """
    # Инициализация TDE хранилища
    # Ключи можно получать из Vault/HSM, здесь оставляем None для автоматического derivation
    tde_store = TDEColumnStore(
        data=df,
        encrypted_columns=encrypted_columns,
        key_provider=None,  # можно передать VaultKeyProvider(...)
    )

    # Данные с зашифрованными столбцами
    encrypted_df = tde_store.get_data()

    # Маскировка для аналитиков
    masker = DataMasker(mask_columns=mask_columns or ["value", "from_address", "to_address"])
    masked_df = masker.mask(encrypted_df)

    return {
        "encrypted": encrypted_df,
        "masked": masked_df,
        "tde_status": tde_store.get_column_status(),
        "masked_columns": masker.get_masked_columns(),
    }