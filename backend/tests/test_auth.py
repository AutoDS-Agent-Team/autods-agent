import asyncio
from typing import Any
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from app.services import auth_service

def request(app: FastAPI, method: str, path: str, **kwargs: Any) -> Response:
    async def send() -> Response:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(send())

def test_register_login_and_no_hash_in_response(test_app: FastAPI) -> None:
    registered = request(test_app, "POST", "/api/v1/auth/register", json={"email":"auth@example.com","password":"safe-password-123"})
    assert registered.status_code == 201
    assert "password_hash" not in registered.text and "safe-password-123" not in registered.text
    assert request(test_app, "POST", "/api/v1/auth/login", json={"email":"auth@example.com","password":"wrong-password-123"}).status_code == 400
    assert request(test_app, "POST", "/api/v1/auth/login", json={"email":"auth@example.com","password":"safe-password-123"}).status_code == 200

def test_google_identity_is_verified_and_links_existing_email(test_app: FastAPI, monkeypatch) -> None:
    monkeypatch.setattr(auth_service.id_token, "verify_oauth2_token", lambda *_args: {"sub":"google-subject", "email":"linked@example.com", "email_verified":True})
    response = request(test_app, "POST", "/api/v1/auth/register", json={"email":"linked@example.com","password":"safe-password-123"})
    assert response.status_code == 201
    google = request(test_app, "POST", "/api/v1/auth/google", json={"credential":"mock-google-credential"})
    assert google.status_code == 200, google.text
    assert google.json()["user"]["email"] == "linked@example.com"
    assert "password_hash" not in google.text and "mock-google-credential" not in google.text
    again = request(test_app, "POST", "/api/v1/auth/google", json={"credential":"mock-google-credential"})
    assert again.status_code == 200 and again.json()["user"]["id"] == google.json()["user"]["id"]
