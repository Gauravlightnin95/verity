"""FastAPI endpoint tests via TestClient. Fully mocked end-to-end - no
network calls, no real API keys required."""


def test_health(api_client):
    resp = api_client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_verify_with_text(api_client):
    resp = api_client.post("/verify", data={"text": "The city approved a new metro line."})

    assert resp.status_code == 200
    body = resp.json()
    assert body["label"]
    assert "confidence" in body
    assert "checks_performed" in body


def test_verify_with_url(api_client):
    resp = api_client.post("/verify", data={"url": "https://example.com/article"})

    assert resp.status_code == 200
    assert resp.json()["label"]


def test_verify_with_image_upload(api_client):
    resp = api_client.post(
        "/verify",
        files={"image": ("clipping.png", b"\x89PNG\r\n\x1a\n fake bytes", "image/png")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["label"]
    signal_names = {s["name"] for s in body["signals"]}
    assert {"halftone", "masthead_match"} <= signal_names


def test_verify_with_no_input_is_a_400(api_client):
    resp = api_client.post("/verify", data={})

    assert resp.status_code == 400
