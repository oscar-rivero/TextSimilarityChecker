document.addEventListener('DOMContentLoaded', function() {
    // Character counter for text input
    const textInput = document.getElementById('text-input');
    const charCount = document.getElementById('char-count');
    
    if (textInput && charCount) {
        textInput.addEventListener('input', function() {
            charCount.textContent = this.value.length;
        });
        
        // Initial count
        charCount.textContent = textInput.value.length;
    }
    
    // Export report functionality
    const exportReportBtn = document.getElementById('export-report');
    if (exportReportBtn) {
        exportReportBtn.addEventListener('click', function() {
            // Call the API endpoint to get report data
            fetch(window.location.origin + '/report/')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Error generating report: ' + response.statusText);
                    }
                    return response.json();
                })
                .then(data => {
                    // Generate and download the report
                    createReportText(data);
                })
                .catch(error => {
                    console.error('Error fetching report data:', error);
                    alert('Error generating report: ' + error.message);
                });
        });
    }
    
    // Initialize charts if on results page
    if (document.getElementById('similarity-chart')) {
        initializeCharts();
    }
    
    // Highlight matching text
    highlightMatches();
});

// Function to generate and download a report
function createReportText(data) {
    // Create text report
    let reportText = "PLAGIARISM DETECTION REPORT\n";
    reportText += "============================\n\n";
    reportText += "Generated on: " + new Date().toLocaleString() + "\n\n";
    
    // Original text section
    reportText += "ORIGINAL TEXT:\n";
    reportText += "-------------\n";
    reportText += data.original_text + "\n\n";
    
    // Summary section
    reportText += "SUMMARY:\n";
    reportText += "--------\n";
    reportText += "Sources Found: " + data.sources_count + "\n";
    reportText += "Average Similarity: " + data.average_similarity.toFixed(2) + "%\n";
    reportText += "Total Matching Phrases: " + data.total_matches + "\n\n";
    
    // Sources section
    reportText += "MATCHING SOURCES:\n";
    reportText += "----------------\n";
    
    if (data.sources.length === 0) {
        reportText += "No matching sources found.\n\n";
    } else {
        data.sources.forEach((source, index) => {
            reportText += (index + 1) + ". " + source.title + "\n";
            reportText += "   URL: " + source.url + "\n";
            reportText += "   Similarity: " + (source.similarity * 100).toFixed(2) + "%\n";
            reportText += "   Matching Phrases: " + source.matches.length + "\n\n";
            
            // List matching phrases
            if (source.matches.length > 0) {
                reportText += "   Matching Content:\n";
                source.matches.forEach((match, mIndex) => {
                    reportText += "   - " + match.original + "\n";
                });
                reportText += "\n";
            }
        });
    }
    
    // Create a blob and download
    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'plagiarism_report_' + new Date().toISOString().slice(0, 10) + '.txt';
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 100);
}

// Function to initialize charts on the results page
function initializeCharts() {
    try {
        const similarityDataElem = document.getElementById('similarity-data');
        if (!similarityDataElem) return;
        
        const data = JSON.parse(similarityDataElem.value);
        if (!data || data.length === 0) return;
        
        // Create data for chart
        const labels = data.map(item => {
            // Truncate long titles
            let title = item.source.title || 'Unnamed Source';
            return title.length > 30 ? title.substring(0, 27) + '...' : title;
        });
        
        const values = data.map(item => (item.similarity * 100).toFixed(2));
        const backgroundColors = data.map(item => {
            const similarity = item.similarity;
            if (similarity > 0.5) return '#dc3545'; // high
            if (similarity > 0.3) return '#fd7e14'; // medium
            return '#28a745'; // low
        });
        
        // Create the chart
        const ctx = document.getElementById('similarity-chart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Similarity %',
                    data: values,
                    backgroundColor: backgroundColors,
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
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
                            text: 'Sources'
                        },
                        ticks: {
                            autoSkip: false,
                            maxRotation: 90,
                            minRotation: 45
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            title: function(tooltipItems) {
                                const index = tooltipItems[0].dataIndex;
                                return data[index].source.title || 'Unnamed Source';
                            },
                            afterTitle: function(tooltipItems) {
                                const index = tooltipItems[0].dataIndex;
                                return data[index].source.url || '';
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