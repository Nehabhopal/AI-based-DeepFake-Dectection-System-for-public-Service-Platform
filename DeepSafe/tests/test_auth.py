def test_signup_login_and_me(client):
    r=client.post("/api/auth/signup",json={"name":"Test User","email":"test@example.com","password":"securepass123"})
    assert r.status_code==201
    assert r.json["user"]["email"]=="test@example.com"
    assert client.get("/api/auth/me").json["user"]["name"]=="Test User"
    client.post("/api/auth/logout")
    r=client.post("/api/auth/login",json={"email":"test@example.com","password":"securepass123"})
    assert r.status_code==200
    assert r.json["success"] is True

def test_bad_login_is_rejected(client):
    r=client.post("/api/auth/login",json={"email":"none@example.com","password":"wrong-password"})
    assert r.status_code==401
