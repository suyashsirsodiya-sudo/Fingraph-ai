import json
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"

def get(path):
    url = BASE + path
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

def post(path, body):
    url = BASE + path
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

print("Testing /api endpoints:")
for ep in [
    "/api/status",
    "/api/overview",
    "/api/events",
    "/api/graph",
    "/api/graph/quality",
    "/api/graph/conflicts",
    "/api/nodes/ent_crude_oil",
    "/api/edges/rel_01",
    "/api/paths?event_id=evt_demo_hormuz",
]:
    status, data = get(ep)
    print(f"GET {ep} -> {status}")
    assert status == 200, f"Expected 200 for {ep}, got {status}"

# Test POST /api/scenarios
sc_body = {
    "title": "Crude Oil +15% Shock Scenario",
    "source_entity_id": "ent_event_hormuz",
    "pct_change": 15.0,
    "horizon": "1-3 months"
}
status, sc_data = post("/api/scenarios", sc_body)
print(f"POST /api/scenarios -> {status}, label: {sc_data.get('label')}")
assert status == 200
assert sc_data.get("label") == "Scenario Analysis - Not a Price Forecast"

# Test GET /api/scenarios
status, sc_list = get("/api/scenarios")
print(f"GET /api/scenarios -> {status}, count: {len(sc_list)}")
assert status == 200

# Test 404 on unknown node
status, err_data = get("/api/nodes/unknown_node_xyz")
print(f"GET /api/nodes/unknown_node_xyz -> {status}")
assert status == 404

print("All /api endpoints tested and verified successfully!")
