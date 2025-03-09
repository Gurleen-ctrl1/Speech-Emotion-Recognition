function uploadAudio() {
    let fileInput = document.getElementById('audioFile');
    if (fileInput.files.length === 0) {
        alert('Please select a file first!');
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = "Predicted Emotion: " + data.emotion;
    })
    .catch(error => {
        console.error("Error:", error);
        alert("Error predicting emotion.");
    });
}
