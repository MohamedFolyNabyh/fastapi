async function predict() {
    const input = document.getElementById('imageInput');
    const file = input.files[0];

    if (!file) {
        alert("Please select an image!");
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('imagePreview').innerHTML = `<img src="${e.target.result}" alt="Preview">`;
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        document.getElementById("result").innerText = "Prediction: " + result.result;
    } catch (error) {
        document.getElementById("result").innerText = "Error during prediction!";
        console.error(error);
    }
}
