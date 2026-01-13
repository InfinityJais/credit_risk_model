document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const submitBtn = document.getElementById('predictBtn');
    const originalBtnText = submitBtn.innerText;
    const resultCard = document.getElementById('resultCard');
    const errorMsg = document.getElementById('statusText');

    // 1. Show Loading State
    submitBtn.innerText = "Analyzing...";
    submitBtn.disabled = true;
    resultCard.classList.add('hidden');

    // 2. Collect & Format Data
    const formData = new FormData(e.target);
    const data = {};

    // List of fields that MUST be Float (decimals)
    const floatFields = [
        'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever',
        'pct_tl_open_L6M', 'pct_tl_closed_L6M', 'pct_tl_closed_L12M'
    ];

    // List of fields that act as Strings
    const stringFields = [
        'GENDER', 'MARITALSTATUS', 'EDUCATION', 
        'last_prod_enq2', 'first_prod_enq2'
    ];

    formData.forEach((value, key) => {
        if (stringFields.includes(key)) {
            data[key] = value;
        } else if (floatFields.includes(key)) {
            data[key] = parseFloat(value) || 0.0;
        } else {
            // Everything else is an Integer
            data[key] = parseInt(value) || 0;
        }
    });

    try {
        // 3. Send Request to API
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.statusText}`);
        }

        const result = await response.json();

        // 4. Update UI with Result
        const band = result.risk_band; // e.g., "P1"
        const status = result.status;  // e.g., "Must Give"

        // Update Text
        document.getElementById('riskBand').innerText = band;
        document.getElementById('statusText').innerText = status;

        // Color Logic
        const badge = document.getElementById('riskBand');
        badge.className = 'badge'; // Reset class
        
        if (['P1', 'P2'].includes(band)) {
            badge.classList.add('badge-success'); // Green
            resultCard.style.borderLeft = "5px solid #28a745";
        } else {
            badge.classList.add('badge-danger');  // Red
            resultCard.style.borderLeft = "5px solid #dc3545";
        }

        // Show Card
        resultCard.classList.remove('hidden');
        resultCard.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        alert("Error connecting to API: " + error.message);
        console.error(error);
    } finally {
        // 5. Reset Button
        submitBtn.innerText = originalBtnText;
        submitBtn.disabled = false;
    }
});