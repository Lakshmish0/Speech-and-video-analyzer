function toggleDropdown(element) {
  const dropdown = element.parentElement;
  dropdown.classList.toggle('open');
}

let mediaRecorder;
let recordedChunks = [];
let stream;
let captureInterval;

// Request permissions for camera and microphone
async function requestPermissions() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  } catch (error) {
    alert("Camera and microphone access is required to use this feature.");
    console.error("Permission denied or error accessing media devices:", error);
  }
}

// Handle stream and media recorder for a specific dropdown
function handleStream() {
  const preview = document.getElementById("camera-preview");
  preview.srcObject = stream;
  preview.style.display = "block";

  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) recordedChunks.push(event.data);
  };
}


function captureAndSendFrame() {
const video = document.getElementById('camera-preview');
const canvas = document.getElementById('captureCanvas');
const context = canvas.getContext('2d');

if (video && video.videoWidth && video.videoHeight) {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const frame = canvas.toDataURL('image/jpeg'); // Capture frame as a base64 image

  // Send frame to the server
  fetch('/process_frame', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: frame })
  })
  .then(response => response.json())
  .then(data => {
    if (data.processed_frame) {
      // Update the processed frame display
      const processedImage = document.getElementById('processed-output');
      processedImage.src = data.processed_frame;
    }
  })
  .catch(error => console.error("Error fetching processed frame:", error));
} else {
  console.warn("Video element or video dimensions are not ready.");
}
}

// Event delegation to manage button functionality for each dropdown individually
document.addEventListener("click", async (event) => {
  const dropdownContent = event.target.closest(".dropdown-content");

  if (event.target.classList.contains("startBtn")) {
    document.getElementById("camera-preview").style.display = "none";
    document.getElementById("processed-output").style.display = "block"; 
    if (!stream) await requestPermissions();
    if (!mediaRecorder) handleStream();
    // Request permissions if not already done
    recordedChunks = []; // Clear any previous data
    mediaRecorder.start();
    
    captureInterval = setInterval(captureAndSendFrame, 50);

    dropdownContent.querySelector(".stopBtnandTrans").disabled = false;
    dropdownContent.querySelector(".startBtn").disabled = true;
    dropdownContent.querySelector(".repeatBtn").disabled = true;
  }

  if (event.target.classList.contains("stopBtnandTrans")) {

    document.getElementById("camera-preview").style.display = "block";
    document.getElementById("processed-output").style.display = "none";

    mediaRecorder.stop();

    clearInterval(captureInterval);

    dropdownContent.querySelector(".stopBtnandTrans").disabled = true;
    dropdownContent.querySelector(".repeatBtn").disabled = false;
    document.getElementById("camera-preview").style.display = "none";

    mediaRecorder.onstop = async () => {
        const blob = new Blob(recordedChunks, { type: "audio/wav" });
        const formData = new FormData();
        formData.append("file", blob, "recording.wav");

        try {
            const response = await fetch('/transcribe', { method: 'POST', body: formData });
            const result = await response.json();

            dropdownContent.querySelector(".transcription-result").innerText = response.ok
                ? result.transcription || "Transcription failed."
                : result.error || "Transcription failed.";
        } catch (error) {
            console.error("Error during transcription request:", error);
            dropdownContent.querySelector(".transcription-result").innerText = "An error occurred during transcription.";
        }
    };
}


  if (event.target.classList.contains("repeatBtn")) {
    dropdownContent.querySelector(".startBtn").disabled = false;
    dropdownContent.querySelector(".stopBtnandTrans").disabled = true;
    dropdownContent.querySelector(".repeatBtn").disabled = true;

    const playback = document.querySelector(".playback");
    playback.style.display = "none";
    playback.src = "";
    dropdownContent.querySelector(".transcription-result").innerText = "";
  }
});

requestPermissions();