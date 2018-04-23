import io
import random
import picamera

def write_now():
    # Randomly return True (like a fake motion detection routine)
    return random.randint(0, 10) == 0

def write_video(stream):
    global counter
    print('Writing video!'+ str(counter))
    with stream.lock:
        # Find the first header frame in the video
        for frame in stream.frames:
            print('FRAME TYPE', frame.frame_type)
            if frame.frame_type == picamera.PiVideoFrameType.sps_header:
                stream.seek(frame.position)
                print("FRAME POSITION", frame.position)
                break
        # Write the rest of the stream to disk
        with io.open('../videos/motion62sec'+ str(counter)+ '.h264', 'wb') as output:
            output.write(stream.read())
counter = 0
with picamera.PiCamera() as camera:
    stream = picamera.PiCameraCircularIO(camera, seconds=6)
    camera.start_recording(stream, format='h264')
    try:
        while True:
            camera.wait_recording(1)
            if write_now():
                # Keep recording for 10 seconds and only then write the
                # stream to disk
                camera.wait_recording(2)
                counter +=1
                write_video(stream)
                stream.clear()
    finally:
        camera.stop_recording()