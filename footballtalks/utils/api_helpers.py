# Helper functions for FootballTalks API calls
import requests

def get_json(url, headers=None):
    """Safely fetch JSON from any API endpoint."""
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"⚠️ API error {r.status_code}: {url}")
            return None
    except Exception as e:
        print("Network error:", e)
        return None
