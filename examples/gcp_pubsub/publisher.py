import os
from google.cloud import pubsub_v1
import json
from google.auth import jwt

PROJECT_ID="<YOUR_PROJECT_ID>"
CREDENTIALS = json.load(open("./credentials.json"))

audience = "https://pubsub.googleapis.com/google.pubsub.v1.Subscriber"
credentials = jwt.Credentials.from_service_account_info(CREDENTIALS, audience=audience)
subscriber = pubsub_v1.SubscriberClient(credentials=credentials)

publisher_audience = "https://pubsub.googleapis.com/google.pubsub.v1.Publisher"
credentials_pub = credentials.with_claims(audience=publisher_audience)
publisher = pubsub_v1.PublisherClient(credentials=credentials_pub)

TOPIC="generate"

topic_name = 'projects/{project_id}/topics/{topic}'.format(
    project_id=PROJECT_ID,
    topic=TOPIC
)

on_generated_subscription_name = 'projects/{project_id}/subscriptions/{sub}'.format(
    project_id=PROJECT_ID,
    sub='on_generated-sub',
)

on_error_subscription_name = 'projects/{project_id}/subscriptions/{sub}'.format(
    project_id=PROJECT_ID,
    sub='on_error-sub',
)

payload = {
    'prefix': 'so',
    'suffix': '',
    'jobId': '1234',
    'isCaseSensitive': 'False'
}

encoded_payload = json.dumps(payload).encode('utf-8')

publisher.publish(topic_name, encoded_payload)

print(f"Published {encoded_payload} to {topic_name}")

def on_generated(message):
    data = message.data.decode("utf-8")
    print(f"Received message: {data}")
    message.ack()

def on_error(message):
    data = message.data.decode("utf-8")
    print(f"Received error: {data}")
    message.ack()

on_generated_future = subscriber.subscribe(on_generated_subscription_name, on_generated)
on_error_future = subscriber.subscribe(on_error_subscription_name, on_error)


try:
    on_generated_future.result()
    on_error_future.result()
except KeyboardInterrupt:
    on_generated_future.cancel()
    on_error_future.result()