import json
import logging
from google.cloud import pubsub_v1
from google.auth import jwt
import asyncio
import multiprocessing
from multiprocessing.pool import Pool
import numpy as np
from typing import List
from base58 import b58decode, b58encode
from nacl.signing import SigningKey

import sys
sys.path.append('../../')
from core.searcher import multi_gpu_init, save_result
from core.utils.helpers import check_character, load_kernel_source
from core.config import HostSetting
from core.opencl.manager import (
    get_all_gpu_devices,
    get_chosen_devices,
)

logging.basicConfig(level="INFO", format="[%(levelname)s %(asctime)s] %(message)s")

PROJECT_ID="<YOUR_PROJECT_ID>"
CREDENTIALS = json.load(open("./credentials.json"))

# subscriber authentication
audience = "https://pubsub.googleapis.com/google.pubsub.v1.Subscriber"
credentials = jwt.Credentials.from_service_account_info(CREDENTIALS, audience=audience)
subscriber = pubsub_v1.SubscriberClient(credentials=credentials)

# publisher authentication
publisher_audience = "https://pubsub.googleapis.com/google.pubsub.v1.Publisher"
credentials_pub = credentials.with_claims(audience=publisher_audience)
publisher = pubsub_v1.PublisherClient(credentials=credentials_pub)


generate_topic = 'projects/{project_id}/topics/{topic}'.format(
    project_id=PROJECT_ID,
    topic='on_generated',
)

error_topic = 'projects/{project_id}/topics/{topic}'.format(
    project_id=PROJECT_ID,
    topic='on_error', 
)

topic_name = 'projects/{project_id}/topics/{topic}'.format(
    project_id=PROJECT_ID,
    topic='generate',
)

subscription_name = 'projects/{project_id}/subscriptions/{sub}'.format(
    project_id=PROJECT_ID,
    sub='generate-sub',
)

def get_result(outputs: List):
    private_key, pub_key = "", ""
    result_count = 0
    for output in outputs:
        if not output[0]:
            continue
        result_count += 1
        pv_bytes = bytes(output[1:])
        pv = SigningKey(pv_bytes)
        pb_bytes = bytes(pv.verify_key)
        pub_key = b58encode(pb_bytes).decode()
        decoded_private_key = b58encode(pv_bytes).decode()
        private_hex_string = ''.join(format(x, '02x') for x in list(pv_bytes + pb_bytes))
        private_key = b58encode(bytes.fromhex(private_hex_string)).decode()

    logging.info(f"Found pub key: {pub_key}")
    logging.info(f"Private key: {private_key}")
    return private_key, pub_key


def generate_address(prefix, suffix, jobId, is_case_sensitive):
    starts_with = prefix
    ends_with = suffix
    count = 1
    select_device = False
    iteration_bits = 24
    if not starts_with and not ends_with:
        click.echo("Please provide at least one of --starts-with or --ends-with.")
    check_character("starts_with", starts_with)
    check_character("ends_with", ends_with)

    chosen_devices: Optional[Tuple[int, List[int]]] = None
    if select_device:
        chosen_devices = get_chosen_devices()
        gpu_counts = len(chosen_devices[1])
    else:
        gpu_counts = len(get_all_gpu_devices())

    logging.info(
        f"Searching Solana pubkey with starts_with='{starts_with}', ends_with='{ends_with}', case_sensitive={'on' if is_case_sensitive else 'off'}"
    )
    logging.info(f"Using {gpu_counts} OpenCL device(s)")

    result_count = 0
    results = []

    with multiprocessing.Manager() as manager:
        with Pool(processes=gpu_counts) as pool:
            kernel_source = load_kernel_source(
                starts_with, ends_with, is_case_sensitive
            )
            lock = manager.Lock()
            while result_count < count:
                stop_flag = manager.Value("i", 0)
                partial_results = pool.starmap(
                    multi_gpu_init,
                    [
                        (
                            x,
                            HostSetting(kernel_source, iteration_bits),
                            gpu_counts,
                            stop_flag,
                            lock,
                            chosen_devices,
                        )
                        for x in range(gpu_counts)
                    ],
                )
                result_count += 1
                for partial_result in partial_results:
                    if isinstance(partial_result, dict) and 'error' in partial_result:
                        logging.error(f"Error in worker process: {partial_result['error']}")
                        continue 
                    if isinstance(partial_result, np.ndarray) and partial_result.size > 0 and partial_result[0]:
                        results.append(partial_result)
                    elif isinstance(partial_result, list) and len(partial_result) > 0 and partial_result[0]:
                        results.append(partial_result)



    logging.info("Generation finished, making private and public key")
    privateKey, pubKey = get_result(results)
    logging.info("Pub/secret key made, emitting result")
    json_result = json.dumps({"privateKey": privateKey, "pubKey": pubKey, "jobId": jobId})

    future = publisher.publish(generate_topic, bytes(json_result, "utf-8"))
    future.result()
    logging.info("Result emitted")
    return 'Generated'


def generate(message):
    data = message.data.decode("utf-8")
    body = json.loads(data)
    prefix = body["prefix"]
    suffix = body["suffix"]
    jobId = body["jobId"]
    isCaseSensitive = body["isCaseSensitive"]
    message.ack()
    try:
        logging.info("Generation started")
        generate_address(prefix, suffix, jobId, bool(isCaseSensitive == 'True'))
    except Exception as e:
        print(e)
        logging.info("Error while generation, sending error event")
        json_result = json.dumps({"error": str(e), "jobId":jobId})
        future = publisher.publish(error_topic, bytes(json_result, "utf-8"))
        future.result()

future = subscriber.subscribe(subscription_name, generate)


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = subscriber.subscribe(subscription_name, generate)
    logging.info("Listening for generate event, generator is live")
    loop.run_forever()

if __name__ == "__main__":
    # important because we are using multiprocessing and pyopencl context runs in isolation
    # According to the Python documentation, spawn is the default on Windows and macOS. So sub/pub works fine on macOS but not on linux.
    multiprocessing.set_start_method("spawn")
    main()
