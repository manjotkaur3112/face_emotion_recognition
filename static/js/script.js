document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("imageInput");
  const preview = document.getElementById("preview");
  const previewCard = document.getElementById("previewCard");
  const uploadForm = document.getElementById("uploadForm");
  const loader = document.getElementById("loader");

  if (fileInput) {
    fileInput.addEventListener("change", function () {
      const file = this.files[0];
      if (file) {
        preview.src = URL.createObjectURL(file);
        previewCard.style.display = "block";
      }
    });
  }

  if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
      loader.style.display = "block";
    });
  }

  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const overlay = document.getElementById("overlay");
  const ctx = overlay ? overlay.getContext("2d") : null;
  const status = document.getElementById("status");

  if (video && canvas && overlay && status) {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user",
          },
          audio: false,
        });

        video.srcObject = stream;

        video.onloadedmetadata = function () {
          if (loader) loader.style.display = "none";
          overlay.width = video.videoWidth;
          overlay.height = video.videoHeight;
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          status.textContent = "Camera started. Detecting emotion...";

          captureFrame();
          setInterval(captureFrame, 4000);
        };
      } catch (error) {
        console.error("Camera error:", error);

        if (loader) {
          loader.innerHTML = `
            <p class="text-danger">Unable to access camera.</p>
            <p>Please allow camera permission in your browser.</p>
          `;
        }

        status.textContent = "Camera access failed.";
      }
    }

    async function captureFrame() {
      if (!video.videoWidth || !video.videoHeight) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        async function (blob) {
          if (!blob) return;

          const formData = new FormData();
          formData.append("image", blob, "camera.jpg");
          status.textContent = "Analyzing emotion...";

          try {
            const response = await fetch("/predict_frame", {
              method: "POST",
              body: formData,
            });

            const result = await response.json();

            if (!response.ok) {
              console.error(result);
              status.textContent = "Prediction failed.";
              return;
            }

            drawResults(result);
          } catch (error) {
            console.error("Prediction error:", error);
            status.textContent = "Unable to connect to server.";
          }
        },
        "image/jpeg",
        0.85,
      );
    }

    function drawResults(result) {
      if (!ctx) return;
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      if (!result.faces || result.faces.length === 0) {
        status.textContent = "No face detected.";
        return;
      }

      result.faces.forEach(function (face) {
        const x = face.x;
        const y = face.y;
        const width = face.width;
        const height = face.height;

        ctx.strokeStyle = "#00ff00";
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, width, height);

        const text = face.emotion + " " + face.confidence + "%";
        ctx.font = "bold 28px Arial";
        const textWidth = ctx.measureText(text).width;
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(x, Math.max(0, y - 45), textWidth + 20, 42);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(text, x + 10, Math.max(30, y - 15));
      });

      status.textContent = "Next prediction in 5 seconds.";
    }

    startCamera();
  }
});
