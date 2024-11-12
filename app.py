import os
from flask import Flask, request, jsonify, render_template, send_file
from faster_whisper import WhisperModel
from fpdf import FPDF
import google.generativeai as genai
import cv2
import dlib
import numpy as np
import base64


app = Flask(__name__)


face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')


# Define 3D model points of a human head
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
])
# Define the camera matrix and distortion coefficients
CAMERA_MATRIX = np.array([[1000, 0, 320],
                          [0, 1000, 240],
                          [0, 0, 1]], dtype="double")
DISTORTION_COEFFS = np.zeros((4, 1))  # Assuming no lens distortion


# Initialize Kalman Filter for tracking
def initialize_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kalman


left_kalman = initialize_kalman_filter()
right_kalman = initialize_kalman_filter()
left_pupil_history = []
right_pupil_history = []


model_size = "distil-large-v3"
model = WhisperModel(model_size, device="cpu", compute_type="int8")


transcriptions = []
gemini_results = []


@app.route('/')
def index():
    questions = [
        "Tell me about yourself.",
        "What are your strengths?",
        "What is one of your weaknesses?",
        "Why do you want to work here?"]
    return render_template('hom.html', questions=questions)


@app.route('/questions')
def questions():
    # List of questions to render
    questions = [
        "What is your name?",
        "How old are you?",
        "Where are you from?",
        "What are your hobbies?"
    ]
    return render_template('questions.html', questions=questions)


@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json.get('image')
    if data:
        try:
        # Decode base64 image
            image_data = base64.b64decode(data.split(",")[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert to grayscale and detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)

            for face in faces:
                (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                landmarks = predictor(gray, face)
                left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
                right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

                # Draw circles around eye landmarks
                for (x, y) in left_eye + right_eye:
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

                # Calculate pupil and gaze direction
                left_pupil = get_eye_gaze_direction(left_eye, frame, left_kalman, left_pupil_history)
                right_pupil = get_eye_gaze_direction(right_eye, frame, right_kalman, right_pupil_history)

                # Head pose estimation points
                image_points = np.array([
                    (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
                    (landmarks.part(8).x, landmarks.part(8).y),    # Chin
                    (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
                    (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
                    (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
                    (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
                ], dtype="double")

                # Estimate head pose
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    MODEL_POINTS, image_points, CAMERA_MATRIX, DISTORTION_COEFFS)
                if success:
                    rmat, _ = cv2.Rodrigues(rotation_vector)
                    pose_mat = np.hstack((rmat, translation_vector))
                    proj_matrix = CAMERA_MATRIX @ pose_mat

                    # Calculate angles
                    pitch = np.arctan2(proj_matrix[1, 2], proj_matrix[2, 2])
                    yaw = np.arctan2(-proj_matrix[0, 2], np.sqrt(proj_matrix[1, 2]**2 + proj_matrix[2, 2]**2))
                    roll = np.arctan2(proj_matrix[0, 1], proj_matrix[0, 0])

                    head_pose_direction = determine_head_pose_direction(np.degrees(pitch), np.degrees(yaw))
                    gaze_direction = determine_eye_gaze_direction(left_pupil, left_eye) or determine_eye_gaze_direction(right_pupil, right_eye)

                    # Display head pose and gaze direction
                    cv2.putText(frame, f"Head Pose: {head_pose_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Encode processed frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            processed_frame = base64.b64encode(buffer).decode('utf-8')
            if not ret:
                return jsonify({'status': 'Error encoding image'}), 500
            
            # Return success response
            print('Frame received and processed')
            return jsonify({'processed_frame': f'data:image/jpeg;base64,{processed_frame}'})

        except Exception as e:
            print(f"Error processing frame: {e}")
            return jsonify({'status': 'Error processing frame', 'error': str(e)}), 500
    print('No frame received')
    return jsonify({'status': 'No frame received'}), 400


def get_eye_gaze_direction(eye_points, frame, kalman, history):
    eye_region = frame[eye_points[1][1]:eye_points[5]
                       [1], eye_points[0][0]:eye_points[3][0]]

    pupil_center = detect_pupil(eye_region)
    if pupil_center:
        pupil_center = (pupil_center[0] + eye_points[0]
                        [0], pupil_center[1] + eye_points[1][1])

        kalman.correct(
            np.array([[np.float32(pupil_center[0])], [np.float32(pupil_center[1])]]))
        prediction = kalman.predict()
        kalman_pupil = (int(prediction[0]), int(prediction[1]))

        smoothed_pupil = apply_gaussian_smoothing(history, kalman_pupil)

        # Draw smoothed pupil
        cv2.circle(frame, smoothed_pupil, 5, (255, 0, 0), -1)

        return smoothed_pupil
    return None


def determine_head_pose_direction(pitch, yaw):
    if abs(pitch) < 10:
        if abs(yaw) < 10:
            return "Straight"
        elif yaw > 10:
            return "Right"
        else:
            return "Left"
    elif pitch > 10:
        return "Up"
    else:
        return "Down"


def determine_eye_gaze_direction(pupil_position, eye_region):
    if pupil_position:
        eye_center_x = (eye_region[0][0] + eye_region[3][0]) // 2
        eye_center_y = (eye_region[1][1] + eye_region[5][1]) // 2

        if pupil_position[1] < eye_center_y - 5:
            return "Up"
        elif pupil_position[1] > eye_center_y + 5:
            return "Down"
        elif pupil_position[0] < eye_center_x - 5:
            return "Left"
        elif pupil_position[0] > eye_center_x + 5:
            return "Right"
        else:
            return "Straight"
    return None


def detect_pupil(eye_frame):
    gray_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        return int(x), int(y)
    return None


def apply_gaussian_smoothing(history, new_value, max_length=5, sigma=1.0):
    if len(history) >= max_length:
        history.pop(0)
    history.append(new_value)
    smoothed_value = tuple(np.round(cv2.GaussianBlur(
        np.array(history).astype(float), (1, max_length), sigma)).astype(int)[-1])
    return smoothed_value


@app.route('/transcribe', methods=['POST'])
def transcribe():
    global transcriptions
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400

        audio_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(audio_path)

        # Step 4: Perform transcription with the preloaded Whisper model
        try:
            segments, info = model.transcribe(
                audio_path, beam_size=1, language="en", condition_on_previous_text=False)
            transcription = " ".join([segment.text for segment in segments])
            print("Transcription successful:", transcription)
            transcriptions.append(transcription)
            # Get predefined answer
            predefined_response = get_predefined_answer(transcription)

            # # Send transcription and response to Gemini
            send_to_gemini(transcription, predefined_response)
        except Exception as e:
            transcription = f"Transcription Failed: {e}"
            print(transcription)

        os.remove(audio_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

    return jsonify({"transcription": transcription})


def get_predefined_answer(transcription):
    # Simple matching based on keywords
    predefined_answers = [
        "Tell me about yourself.",
        "What are your strengths?",
        "What is one of your weaknesses?",
        "Why do you want to work here?"]

    for phrase in predefined_answers:
        if phrase.lower() in transcription.lower():
            return phrase

    return "Sorry, I didn't understand that."


def send_to_gemini(transcription, response):
    global gemini_results
    genai.configure(api_key="AIzaSyCs1wMw4RjOrsKcZWPASdx0nNXj2kurGCE")
    mode = genai.GenerativeModel("gemini-pro")

    # Define the prompt for Gemini
    prompt = f"User's answer: {transcription}\nGiven question: {response}\nPlease rate the user's answer to the given question on a scale from 1 to 10. Then, summarize the user's answer in 5 to 6 lines."
    response = mode.generate_content(prompt)
    print(prompt)
    gemini_results.append(response.text)
    print(response)
    return 0


@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_right_margin(20)
    pdf.set_left_margin(20)
    pdf.set_auto_page_break(auto=True, margin=10)
    # Assuming transcriptions and gemini_results are lists of transcription text and JSON objects, respectively
    for i, (transcription, gemini_result) in enumerate(zip(transcriptions, gemini_results), start=1):
        # Add transcription to PDF
        pdf.multi_cell(
            0, 10, txt=f"Transcription {i}: {transcription}", border=1)

        pdf.multi_cell(
            0, 10, txt=f"Gemini Result {i}: {gemini_result}", border=1)
        pdf.cell(0, 10, txt="", ln=True)  # Blank line for spacing

    pdf_output_path = "transcriptions.pdf"
    pdf.output(pdf_output_path)

    return send_file(pdf_output_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
