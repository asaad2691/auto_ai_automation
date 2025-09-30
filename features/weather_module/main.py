import requests
from datetime import datetime

def get_weather(latitude: float, longitude: float):
    """
    Get current weather using Open-Meteo API (no API key required).
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}"
            f"&current_weather=true"
        )
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            current = data.get("current_weather", {})
            return {
                "temperature": current.get("temperature"),
                "windspeed": current.get("windspeed"),
                "winddirection": current.get("winddirection"),
                "weathercode": current.get("weathercode"),
                "time": current.get("time"),
            }
        else:
            return {"error": f"API request failed with status {response.status_code}"}

    except Exception as e:
        return {"error": str(e)}


def get_current_location():
    """
    Get current location (latitude, longitude) using IP geolocation.
    Free service: ipinfo.io (no API key needed for basic info).
    """
    try:
        res = requests.get("https://ipinfo.io/json", timeout=10)
        if res.status_code == 200:
            loc = res.json().get("loc", "")
            if loc:
                lat, lon = loc.split(",")
                return float(lat), float(lon)
        return None, None
    except Exception as e:
        print("[Error] Failed to fetch location:", e)
        return None, None


def run(query: str = "") -> str:
    lat, lon = get_current_location()
    if lat is None or lon is None:
        return "❌ Failed to get current location."

    weather = get_weather(lat, lon)
    if "error" in weather:
        return f"❌ Failed to fetch weather data: {weather['error']}"

    return (
        f"🌤️ Current Weather at ({lat}, {lon}):\n"
        f"- Temperature: {weather['temperature']} °C\n"
        f"- Wind Speed: {weather['windspeed']} km/h\n"
        f"- Wind Direction: {weather['winddirection']}°\n"
        f"- Weather Code: {weather['weathercode']}\n"
        f"- Time: {weather['time']}"
    )


if __name__ == "__main__":
    print(run())
