let uploadedFile = null; // Save the file globally
let myChart = null;

function previewImage() {
    const input = document.getElementById('imageUpload');
    const file = input.files[0];

    if (!file) {
        return;
    }

    uploadedFile = file; // Save for later

    // handle the image on load
    const reader = new FileReader();
    reader.onload = function(event) {
        document.getElementById('uploaded-img').src = event.target.result;
        document.getElementById('uploaded-img').style.display = "";

        // stuff to make it show up in the User interface
        const img = new Image();
        img.onload = function() {
            const canvas = document.getElementById('imageCanvas');
            const ctx = canvas.getContext('2d');

            const targetWidth = 224;
            const targetHeight = 224;
            canvas.width = targetWidth;
            canvas.height = targetHeight;
            ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

function processImage() {
    /*
        To process the image,
        we must save the file to the server
        In our case, the server is our local machine
        We then store the path to this file for future processing
    */
    if (!uploadedFile) {
        alert('Please upload an image first!');
        return;
    }

    if (myChart !== null) {
        myChart.destroy();
    }


    const formData = new FormData();
    formData.append('file', uploadedFile);

    fetch('http://localhost:5000/run_script', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        console.log('Server response:', data);
        document.getElementById('result').innerText = "";//"Classification Complete";
        plotChart(data);
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'An error occurred.';
    });

    document.getElementById('result').innerText = 'Processing image...';
}

function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function plotChart(data){

    data = JSON.parse(data);

    console.log("Here: ", data);

    const rec = data.rec;
    console.log("rec: ", rec)

    delete data.rec;

    console.log("rec: ", rec)
    
    
    const classes = Object.keys(data);
    const values = Object.values(data);
    // const yValues = [55, 49, 44, 24, 15];
    const barColors = ["red", "green","blue","orange"];

    myChart = new Chart("myChart", {
    type: "bar",
    data: {
        labels: classes,
        datasets: [{
        backgroundColor: barColors,
        data: values
        }]
    },options: {
        title: {
          display: true,
          text: "Model Predictions"
        }
      }
    });

    prediction = classes[argMax(values)]

    document.getElementById("myChart").style.display="";
    document.getElementById("final-classification").innerText = "Final Classification: " + prediction;
    document.getElementById("recommendation").innerText = "Generated Recommendation: " + rec;
}