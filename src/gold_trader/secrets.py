from __future__ import annotations

from contextlib import suppress

import keyring

SERVICE_NAME = "gold-trader"


def set_secret(name: str, value: str) -> None:
    if not name or not value:
        raise ValueError("secret name and value are required")
    keyring.set_password(SERVICE_NAME, name, value)


def get_secret(name: str) -> str | None:
    return keyring.get_password(SERVICE_NAME, name)


def delete_secret(name: str) -> None:
    with suppress(keyring.errors.PasswordDeleteError):
        keyring.delete_password(SERVICE_NAME, name)
