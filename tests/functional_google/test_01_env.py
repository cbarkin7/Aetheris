from dotenv import load_dotenv
import os

load_dotenv()

print("GOOGLE_REFRESH_TOKEN existe:", bool(os.getenv("GOOGLE_REFRESH_TOKEN")))
print("Primeros 40 chars:", os.getenv("GOOGLE_REFRESH_TOKEN", "")[:40])
print("GOOGLE_CLIENT_ID existe:", bool(os.getenv("GOOGLE_CLIENT_ID")))
print("GOOGLE_CLIENT_SECRET existe:", bool(os.getenv("GOOGLE_CLIENT_SECRET")))