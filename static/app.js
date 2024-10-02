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
    
    element.innerHTML = `
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px">
        <div style="display: flex; align-items: center;">
            <span style="font-size: 14px;">Previous Close: $${todayComparison.yesterday_close.toFixed(2)}</span>
            <span style="font-size: 14px; margin-left: 10px;">Current Price: $${todayComparison.latest_price.toFixed(2)}</span>
        </div>
        <span style="color: ${color}; font-size: 18px; font-weight: bold;">${direction} ${returnPercentage.toFixed(2)}%</span>
        <span style="font-size: 14px; margin-left: 10px;">Percentile: ${todayComparison.percentile.toFixed(2)}%</span>
    </div>
    `;
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