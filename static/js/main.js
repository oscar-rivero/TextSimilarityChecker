document.addEventListener('DOMContentLoaded', function() {
    // Handle the text area character count
    const textArea = document.getElementById('text-input');
    const charCount = document.getElementById('char-count');
    
    if (textArea && charCount) {
        textArea.addEventListener('input', function() {
            charCount.textContent = textArea.value.length;
        });
    }
    
    // Initialize any tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Set up the copy to clipboard functionality
    const copyBtns = document.querySelectorAll('.copy-btn');
    if (copyBtns) {
        copyBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const textToCopy = this.getAttribute('data-text');
                if (textToCopy) {
                    navigator.clipboard.writeText(textToCopy).then(() => {
                        // Change button text/icon temporarily
                        const originalText = this.innerHTML;
                        this.innerHTML = '<i class="fas fa-check"></i> Copied!';
                        setTimeout(() => {
                            this.innerHTML = originalText;
                        }, 2000);
                    }).catch(err => {
                        console.error('Could not copy text: ', err);
                    });
                }
            });
        });
    }
    
    // Handle the export report button
    const exportReportBtn = document.getElementById('export-report');
    if (exportReportBtn) {
        exportReportBtn.addEventListener('click', function() {
            // Fetch the report data
            fetch('/report')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    // Create a downloadable text file
                    const reportText = createReportText(data);
                    const blob = new Blob([reportText], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    
                    // Create a temporary link and click it to download
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'plagiarism_report.txt';
                    document.body.appendChild(a);
                    a.click();
                    
                    // Clean up
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                })
                .catch(error => {
                    console.error('Error exporting report:', error);
                    alert('Error exporting report: ' + error.message);
                });
        });
    }
    
    // Handle the visualization of results
    const resultsContainer = document.getElementById('results-container');
    if (resultsContainer) {
        // Initialize charts if results are available
        initializeCharts();
    }
});

function createReportText(data) {
    let report = '===== PLAGIARISM CHECK REPORT =====\n\n';
    report += `Date: ${data.timestamp}\n`;
    report += `Words analyzed: ${data.original_length}\n`;
    report += `Sources checked: ${data.sources_checked}\n`;
    report += `Average similarity: ${(data.average_similarity * 100).toFixed(2)}%\n`;
    report += `Total matching phrases: ${data.total_matches}\n\n`;
    
    report += '--- TOP MATCHING SOURCES ---\n\n';
    data.top_sources.forEach((source, index) => {
        report += `${index + 1}. ${source.title}\n`;
        report += `   URL: ${source.url}\n`;
        report += `   Similarity: ${(source.similarity * 100).toFixed(2)}%\n`;
        report += `   Matching phrases: ${source.match_count}\n\n`;
    });
    
    return report;
}

function initializeCharts() {
    // Get the similarity data from the page
    const chartContainer = document.getElementById('similarity-chart');
    
    if (!chartContainer) return;
    
    // Get similarity data from hidden input
    const similarityDataElem = document.getElementById('similarity-data');
    if (!similarityDataElem) return;
    
    try {
        const similarityData = JSON.parse(similarityDataElem.value);
        
        // Create labels and data for the chart
        const labels = similarityData.map(item => {
            // Truncate long titles
            let title = item.source.title;
            return title.length > 30 ? title.substring(0, 27) + '...' : title;
        });
        
        const data = similarityData.map(item => (item.similarity * 100).toFixed(2));
        
        // Create the chart
        const ctx = chartContainer.getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Similarity (%)',
                    data: data,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Similarity (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Source'
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Similarity: ${context.raw}%`;
                            }
                        }
                    }
                }
            }
        });
    } catch (e) {
        console.error('Error initializing chart:', e);
    }
}

// Function to highlight the matched text in the original content
function highlightMatches() {
    const originalTextContainer = document.getElementById('original-text-container');
    if (!originalTextContainer) return;
    
    const matchesDataElem = document.getElementById('matches-data');
    if (!matchesDataElem) return;
    
    try {
        // Safely parse the JSON data, handling potential flattened array structure
        let matchesValue = matchesDataElem.value;
        let matches = [];
        
        try {
            // Try to parse the direct value
            matches = JSON.parse(matchesValue);
            
            // If matches is not an array but contains matches from multiple sources
            if (!Array.isArray(matches)) {
                // Try to extract matches from results structure
                const resultsData = document.getElementById('similarity-data');
                if (resultsData) {
                    const results = JSON.parse(resultsData.value);
                    matches = [];
                    results.forEach(result => {
                        if (result.matches && Array.isArray(result.matches)) {
                            matches = matches.concat(result.matches);
                        }
                    });
                }
            }
        } catch (parseError) {
            console.warn('Could not parse matches data, using empty array:', parseError);
            matches = [];
        }
        
        // If we still don't have an array, return early
        if (!Array.isArray(matches)) {
            console.warn('Matches data is not an array, skipping highlighting');
            return;
        }
        
        let originalText = originalTextContainer.textContent;
        
        // Create a map of matches to avoid overlapping highlights
        let highlightMap = new Map();
        
        // Add each match to the map
        matches.forEach(match => {
            // Make sure match has the expected structure
            if (match && typeof match === 'object' && match.original) {
                const matchText = match.original;
                const startIdx = originalText.indexOf(matchText);
                
                if (startIdx !== -1) {
                    const endIdx = startIdx + matchText.length;
                    
                    // Store this match in our map
                    for (let i = startIdx; i < endIdx; i++) {
                        highlightMap.set(i, true);
                    }
                }
            }
        });
        
        // Now create the highlighted HTML
        let highlightedHtml = '';
        let inHighlight = false;
        
        for (let i = 0; i < originalText.length; i++) {
            const char = originalText[i];
            
            if (highlightMap.has(i) && !inHighlight) {
                // Start of a highlighted section
                highlightedHtml += '<span class="highlight">';
                inHighlight = true;
            } else if (!highlightMap.has(i) && inHighlight) {
                // End of a highlighted section
                highlightedHtml += '</span>';
                inHighlight = false;
            }
            
            // Add the current character
            highlightedHtml += char;
        }
        
        // Close any open highlight span
        if (inHighlight) {
            highlightedHtml += '</span>';
        }
        
        // Set the highlighted HTML
        originalTextContainer.innerHTML = highlightedHtml;
    } catch (e) {
        console.error('Error highlighting matches:', e);
    }
}

// Call highlight function when the page is loaded
document.addEventListener('DOMContentLoaded', function() {
    highlightMatches();
});
