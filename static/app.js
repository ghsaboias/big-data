// Event listener for when the DOM content is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, fetching data...');
    fetchDataAndUpdateUI();
});

// Function to fetch data from the API and update the UI
function fetchDataAndUpdateUI() {
    fetch('/api/data')
        .then(response => {
            console.log('Response received:', response);
            if (!response.ok) {
                return response.json().then(err => { throw err; });
            }
            return response.json();
        })
        .then(data => {
            console.log('Data received:', data);
            if (data.error) {
                throw new Error(data.error);
            }
            if (!data.todays_return_comparison || !data.return_data) {
                throw new Error("Invalid data structure received from server");
            }
            displayTodayReturn(data.todays_return_comparison);
            displayReturnTable(data.return_data);
            hideErrorMessage();
        })
        .catch(error => {
            console.error('Error:', error);
            displayErrorMessage(error.message || "An unknown error occurred");
        });
}

// Function to display today's return comparison
function displayTodayReturn(todayComparison) {
    const element = document.getElementById('todays-return');
    const returnPercentage = (Math.exp(todayComparison.today_return) - 1) * 100;
    const direction = todayComparison.direction === 'up' ? '▲' : '▼';
    const color = todayComparison.direction === 'up' ? '#28a745' : '#dc3545';
    
    const sanityCheck = todayComparison.sanity_check;
    
    element.innerHTML = `
        <span style="color: ${color}; font-size: 18px; font-weight: bold;">${direction} ${returnPercentage.toFixed(2)}%</span>
        <span style="font-size: 16px;">BTC: $${todayComparison.latest_price.toFixed(2)}</span>
        <span style="font-size: 14px;">Percentile: ${todayComparison.percentile.toFixed(2)}%</span>
        <span class="info-icon" onclick="toggleSanityCheck(this)">ⓘ</span>
        <div class="sanity-check-dropdown">
            <p>Return Distribution (1 Year)</p>
            <p>Today's log return: ${todayComparison.today_return.toFixed(6)}</p>
            <p>Historical range: ${sanityCheck.min_return.toFixed(6)} to ${sanityCheck.max_return.toFixed(6)}</p>
            <p>Median: ${sanityCheck.median_return.toFixed(6)}</p>
            <p>Positive returns:</p>
            <p>- 10th percentile: ${sanityCheck.positive_returns_10th_percentile.toFixed(6)}</p>
            <p>- 90th percentile: ${sanityCheck.positive_returns_90th_percentile.toFixed(6)}</p>
            <p>Negative returns:</p>
            <p>- 10th percentile: ${sanityCheck.negative_returns_10th_percentile.toFixed(6)}</p>
            <p>- 90th percentile: ${sanityCheck.negative_returns_90th_percentile.toFixed(6)}</p>
            <p>Rank: ${sanityCheck.rank} out of ${sanityCheck.total_days} ${sanityCheck.comparison_type} days</p>
            <p>Largest ${sanityCheck.comparison_type} return in past year: ${sanityCheck.largest_return.toFixed(6)}</p>
            <p>Ratio to largest ${sanityCheck.comparison_type} return: ${sanityCheck.ratio_to_largest.toFixed(2)}</p>
            <p>Larger than 90% of ${sanityCheck.comparison_type} returns: ${sanityCheck.larger_than_90_percent ? 'Yes' : 'No'}</p>
        </div>
    `;
}

// Function to toggle the sanity check dropdown
function toggleSanityCheck(icon) {
    const dropdown = icon.nextElementSibling;
    if (dropdown.style.display === 'none' || dropdown.style.display === '') {
        dropdown.style.display = 'block';
    } else {
        dropdown.style.display = 'none';
    }
}

// Function to hide the error message
function hideErrorMessage() {
    document.getElementById('error-message').style.display = 'none';
}

// Function to display the error message
function displayErrorMessage(message) {
    const errorElement = document.getElementById('error-message');
    errorElement.textContent = 'Error loading data. Please try again later. Details: ' + message;
    errorElement.style.display = 'block';
}

// Function to display return table
function displayReturnTable(returnData) {
    const container = document.getElementById('return-table-container');
    const timeFrameToggle = document.getElementById('time-frame-toggle');
    
    let currentPage = 1;
    const rowsPerPage = 20;
    
    function createTable(data, page) {
        const startIndex = (page - 1) * rowsPerPage;
        const endIndex = startIndex + rowsPerPage;
        const pageData = data.slice(startIndex, endIndex);
        
        const table = document.createElement('table');
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Return</th>
                    <th>Rank</th>
                </tr>
            </thead>
            <tbody>
                ${pageData.map(row => `
                    <tr>
                        <td>${row.time}</td>
                        <td class="${row.return >= 0 ? 'positive' : 'negative'}">${(row.return * 100).toFixed(2)}%</td>
                        <td>${row.rank} / ${row.total}</td>
                    </tr>
                `).join('')}
            </tbody>
        `;
        return table;
    }
    
    function createPagination(data) {
        const totalPages = Math.ceil(data.length / rowsPerPage);
        const pagination = document.createElement('div');
        pagination.className = 'pagination';
        
        // Add "Previous" button
        const prevButton = document.createElement('button');
        prevButton.textContent = '< Prev';
        prevButton.addEventListener('click', () => {
            if (currentPage > 1) {
                currentPage--;
                updateTable();
            }
        });
        pagination.appendChild(prevButton);
        
        // Add page numbers (show 5 pages around the current page)
        for (let i = Math.max(1, currentPage - 2); i <= Math.min(totalPages, currentPage + 2); i++) {
            const pageButton = document.createElement('button');
            pageButton.textContent = i;
            pageButton.classList.toggle('active', i === currentPage);
            pageButton.addEventListener('click', () => {
                currentPage = i;
                updateTable();
            });
            pagination.appendChild(pageButton);
        }
        
        // Add "Next" button
        const nextButton = document.createElement('button');
        nextButton.textContent = 'Next >';
        nextButton.addEventListener('click', () => {
            if (currentPage < totalPages) {
                currentPage++;
                updateTable();
            }
        });
        pagination.appendChild(nextButton);
        
        return pagination;
    }
    
    function updateTable() {
        const timeFrame = timeFrameToggle.value;
        const data = timeFrame === 'hourly' ? returnData.hourly : returnData.daily;
        container.innerHTML = '';
        container.appendChild(createTable(data, currentPage));
        container.appendChild(createPagination(data));
    }
    
    timeFrameToggle.addEventListener('change', () => {
        currentPage = 1;
        updateTable();
    });
    updateTable();
}

// Add this function call at the end of the file
document.addEventListener('DOMContentLoaded', function() {
    // ... existing code ...
    
    // Close sanity check dropdown when clicking outside
    document.addEventListener('click', function(event) {
        if (!event.target.matches('.info-icon')) {
            const dropdowns = document.getElementsByClassName('sanity-check-dropdown');
            for (let i = 0; i < dropdowns.length; i++) {
                if (dropdowns[i].style.display === 'block') {
                    dropdowns[i].style.display = 'none';
                }
            }
        }
    });
});