// API Configuration
const API_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : 'https://your-api-domain.com'; // Update this for production

// DOM Elements
const fighter1Input = document.getElementById('fighter1');
const fighter2Input = document.getElementById('fighter2');
const fightersList = document.getElementById('fighters-list');
const predictBtn = document.getElementById('predict-btn');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const resultsDiv = document.getElementById('results');

// Store fighters list
let allFighters = [];

// Load fighters on page load
async function loadFighters() {
    try {
        const response = await fetch(`${API_URL}/fighters`);
        const data = await response.json();
        allFighters = data.fighters;
        
        // Populate datalist
        fightersList.innerHTML = '';
        allFighters.forEach(fighter => {
            const option = document.createElement('option');
            option.value = fighter;
            fightersList.appendChild(option);
        });
    } catch (error) {
        showError('Failed to load fighters. Please refresh the page.');
        console.error('Error loading fighters:', error);
    }
}

// Validate inputs and enable/disable button
function validateInputs() {
    const fighter1 = fighter1Input.value.trim();
    const fighter2 = fighter2Input.value.trim();
    
    // Check if both fighters are selected and different
    if (fighter1 && fighter2 && fighter1 !== fighter2) {
        predictBtn.disabled = false;
        hideError();
    } else {
        predictBtn.disabled = true;
    }
}

// Show error message
function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

// Hide error message
function hideError() {
    errorDiv.classList.add('hidden');
}

// Hide results
function hideResults() {
    resultsDiv.classList.add('hidden');
}

// Show loading state
function showLoading() {
    loadingDiv.classList.remove('hidden');
    hideResults();
    hideError();
    predictBtn.disabled = true;
}

// Hide loading state
function hideLoading() {
    loadingDiv.classList.add('hidden');
    predictBtn.disabled = false;
}

// Display prediction results
function displayResults(data) {
    const fighter1Name = data.fighter1;
    const fighter2Name = data.fighter2;
    const prob1 = (data.fighter1_win_probability * 100).toFixed(1);
    const prob2 = (data.fighter2_win_probability * 100).toFixed(1);
    const winner = data.predicted_winner;
    
    // Update card 1
    document.getElementById('name1').textContent = fighter1Name;
    document.getElementById('prob1').textContent = `${prob1}%`;
    const card1 = document.getElementById('card1');
    const winner1 = document.getElementById('winner1');
    
    if (winner === fighter1Name) {
        card1.classList.add('winner');
        winner1.classList.remove('hidden');
    } else {
        card1.classList.remove('winner');
        winner1.classList.add('hidden');
    }
    
    // Update card 2
    document.getElementById('name2').textContent = fighter2Name;
    document.getElementById('prob2').textContent = `${prob2}%`;
    const card2 = document.getElementById('card2');
    const winner2 = document.getElementById('winner2');
    
    if (winner === fighter2Name) {
        card2.classList.add('winner');
        winner2.classList.remove('hidden');
    } else {
        card2.classList.remove('winner');
        winner2.classList.add('hidden');
    }
    
    // Show results
    resultsDiv.classList.remove('hidden');
}

// Make prediction
async function predictFight() {
    const fighter1 = fighter1Input.value.trim();
    const fighter2 = fighter2Input.value.trim();
    
    // Validate
    if (!fighter1 || !fighter2) {
        showError('Please select both fighters.');
        return;
    }
    
    if (fighter1 === fighter2) {
        showError('Please select two different fighters.');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                fighter1: fighter1,
                fighter2: fighter2
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }
        
        const data = await response.json();
        displayResults(data);
        hideError();
    } catch (error) {
        showError(`Error: ${error.message}`);
        hideResults();
    } finally {
        hideLoading();
    }
}

// Event Listeners
fighter1Input.addEventListener('input', validateInputs);
fighter2Input.addEventListener('input', validateInputs);
predictBtn.addEventListener('click', predictFight);

// Load fighters when page loads
loadFighters();

