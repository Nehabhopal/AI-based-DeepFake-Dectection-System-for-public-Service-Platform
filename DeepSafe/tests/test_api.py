def test_health_reports_missing_model(client):
    r=client.get("/api/health")
    assert r.status_code==200
    assert r.json["model"]=="missing"

def test_analyze_requires_login(client):
    r=client.post("/api/analyze/image")
    assert r.status_code==401
