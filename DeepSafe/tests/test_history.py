def test_history_is_private(client):
    client.post("/api/auth/signup",json={"name":"A","email":"a@example.com","password":"securepass123"})
    assert client.get("/api/history").status_code==200
    client.post("/api/auth/logout")
    assert client.get("/api/history").status_code==401
