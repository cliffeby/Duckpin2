#!/usr/bin/env python

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import credentials
import random
from random import randint
import time
import sys
# import iothub_client
# from iothub_client import IoTHubClient, IoTHubClientError, IoTHubTransportProvider, IoTHubClientResult
# from iothub_client import IoTHubMessage, IoTHubMessageDispositionResult, IoTHubError, DeviceMethodReturnValue
# from iothub_client import IoTHubClientRetryPolicy, GetRetryPolicyReturnValue
# from iothub_client_args import get_iothub_opt, OptionError
# From GitHub Migration v1 to v2
from azure.iot.device.aio import IoTHubDeviceClient
from azure.iot.device import Message
from azure.core.exceptions import AzureError
from azure.storage.blob import BlobClient

TIMEOUT = 241000
MINIMUM_POLLING_TIME = 9

# messageTimeout - the maximum time in milliseconds until a message times out.
# The timeout period starts at IoTHubClient.send_event_async.
# By default, messages do not expire.
MESSAGE_TIMEOUT = 10000
RECEIVE_CONTEXT = 0
MESSAGE_COUNT = 5
RECEIVED_COUNT = 0
CONNECTION_STATUS_CONTEXT = 0
TWIN_CONTEXT = 0
SEND_REPORTED_STATE_CONTEXT = 0
METHOD_CONTEXT = 0
PINS = 0

# global counters
RECEIVE_CALLBACKS = 0
SEND_CALLBACKS = 0
BLOB_CALLBACKS = 0
CONNECTION_STATUS_CALLBACKS = 0
TWIN_CALLBACKS = 0
SEND_REPORTED_STATE_CALLBACKS = 0
METHOD_CALLBACKS = 0

# chose HTTP, AMQP, AMQP_WS or MQTT as transport protocol
# v1 PROTOCOL = IoTHubTransportProvider.MQTT
CONNECTION_STRING = credentials.loginFree["ConnectionString"]

# MSGLane4_TXT = "{\"deviceId\": \"myPythonDevice\",\"PinsCCC\": \"% i,\"}"

def store_blob(blob_info, file_name):
    try:
        sas_url = "https://{}/{}/{}{}".format(
            credentials.blob_info["hostName"],
            credentials.blob_info["containerName"],
            credentials.blob_info["blobName"],
            credentials.blob_info["sasToken"]
        )

        print("\nUploading file: {} to Azure Storage as blob: {} in container {}\n".format(file_name, blob_info["blobName"], blob_info["containerName"]))

        # Upload the specified file
        with BlobClient.from_blob_url(sas_url) as blob_client:
            with open(file_name, "rb") as f:
                result = blob_client.upload_blob(f, overwrite=True)
                return (True, result)

    except FileNotFoundError as ex:
        # catch file not found and add an HTTP status code to return in notification to IoT Hub
        ex.status_code = 404
        return (False, ex)

    except AzureError as ex:
        # catch Azure errors that might result from the upload operation
        return (False, ex)

def receive_message_callback(message, counter):
    global RECEIVE_CALLBACKS
    message_buffer = message.get_bytearray()
    size = len(message_buffer)
    print ( "Received Message [%d]:" % counter )
    print ( "    Data: <<<%s>>> & Size=%d" % (message_buffer[:size].decode('utf-8'), size) )
    map_properties = message.properties()
    key_value_pair = map_properties.get_internals()
    print ( "    Properties: %s" % key_value_pair )
    counter += 1
    RECEIVE_CALLBACKS += 1
    print ( "    Total calls received: %d" % RECEIVE_CALLBACKS )
    return IoTHubMessageDispositionResult.ACCEPTED


def send_confirmation_callback(message, result, user_context):
    global SEND_CALLBACKS
    print ( "Confirmation[%d] received for message with result = %s" % (user_context, result) )
    map_properties = message.properties()
    print ( "    message_id: %s" % message.message_id )
    print ( "    correlation_id: %s" % message.correlation_id )
    key_value_pair = map_properties.get_internals()
    print ( "    Properties: %s" % key_value_pair )
    SEND_CALLBACKS += 1
    print ( "    Total calls confirmed: %d" % SEND_CALLBACKS )


def connection_status_callback(result, reason, user_context):
    global CONNECTION_STATUS_CALLBACKS
    print ( "Connection status changed[%d] with:" % (user_context) )
    print ( "    reason: %d" % reason )
    print ( "    result: %s" % result )
    CONNECTION_STATUS_CALLBACKS += 1
    print ( "    Total calls confirmed: %d" % CONNECTION_STATUS_CALLBACKS )

def send_reported_state_callback(status_code, user_context):
    global SEND_REPORTED_STATE_CALLBACKS
    print ( "Confirmation[%d] for reported state received with:" % (user_context) )
    print ( "    status_code: %d" % status_code )
    SEND_REPORTED_STATE_CALLBACKS += 1
    print ( "    Total calls confirmed: %d" % SEND_REPORTED_STATE_CALLBACKS )

def device_method_callback(method_name, payload, user_context):
    global METHOD_CALLBACKS
    print ( "\nMethod callback called with:\nmethodName = %s\npayload = %s\ncontext = %s" % (method_name, payload, user_context) )
    METHOD_CALLBACKS += 1
    print ( "Total calls confirmed: %d\n" % METHOD_CALLBACKS )
    device_method_return_value = DeviceMethodReturnValue()
    device_method_return_value.response = "{ \"Response\": \"This is the response from the device\" }"
    device_method_return_value.status = 200
    return device_method_return_value

def blob_upload_conf_callback(result, user_context):
    global BLOB_CALLBACKS
    print ( "Blob upload confirmation[%d] received for message with result = %s" % (user_context, result) )
    BLOB_CALLBACKS += 1
    print ( "    Total calls confirmed: %d" % BLOB_CALLBACKS )

def iothub_client_init():
    # prepare iothub client
    # v1 client = IoTHubClient(CONNECTION_STRING, PROTOCOL)
    client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
#     await client.connect()
#     client.set_option("messageTimeout", MESSAGE_TIMEOUT)
#     client.set_option("logtrace", 0)
#     client.set_message_callback(receive_message_callback, RECEIVE_CONTEXT)
#     retryPolicy = IoTHubClientRetryPolicy.RETRY_INTERVAL
#     retryInterval = 100
#     client.set_retry_policy(retryPolicy, retryInterval)
#     print ( "SetRetryPolicy to: retryPolicy = %d" %  retryPolicy)
#     print ( "SetRetryPolicy to: retryTimeoutLimitInSeconds = %d" %  retryInterval)
#     retryPolicyReturn = client.get_retry_policy()
#     print ( "GetRetryPolicy returned: retryPolicy = %d" %  retryPolicyReturn.retryPolicy)
#     print ( "GetRetryPolicy returned: retryTimeoutLimitInSeconds = %d" %  retryPolicyReturn.retryTimeoutLimitInSeconds)
    return client

def print_last_message_time(client):
    try:
        last_message = client.get_last_message_receive_time()
        print ( "Last Message: %s" % time.asctime(time.localtime(last_message)) )
        print ( "Actual time : %s" % time.asctime() )
    except IoTHubClientError as iothub_client_error:
        if iothub_client_error.args[0].result == IoTHubClientResult.INDEFINITE_TIME:
            print ( "No message received" )
        else:
            print ( iothub_client_error )