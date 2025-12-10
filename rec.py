import cv2 as cv

def capture_video(output_file='benchmark_video.mp4', width=960, height=540, fps=20):
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_file, fourcc, fps, (width, height))

    print("Press 'q' to stop recording.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv.imshow('Recording Preview', frame)
        out.write(frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()
    print(f"Video saved as {output_file}")

if __name__ == '__main__':
    capture_video()