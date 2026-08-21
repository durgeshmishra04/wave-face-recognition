import firebase_admin
from firebase_admin import credentials, messaging

cred = credentials.Certificate("./pythonai.json")
firebase_admin.initialize_app(cred)

message = messaging.Message(
    notification=messaging.Notification(
        title="Notification",
        body="Hello everyone!"
    ),
    topic="all_devices"
)

response = messaging.send(message)

print(response)