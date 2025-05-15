const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const axios = require('axios'); // Used to call ML API

const app = express();
const PORT = process.env.PORT || 3000;

// Set up middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('public'));
app.set('view engine', 'ejs');

// Home page route
app.get('/', (req, res) => {
    res.render('index', { prediction: null });
});

// Handle form submission
app.post('/predict', async (req, res) => {
    const inputData = req.body;

    try {
        const response = await axios.post('http://localhost:5000/predict', inputData); // Flask backend URL
        const prediction = response.data.prediction;
        res.render('index', { prediction: `Customer is likely to: ${prediction}` });
    } catch (error) {
        console.error(error.message);
        res.render('index', { prediction: 'Error occurred during prediction.' });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
