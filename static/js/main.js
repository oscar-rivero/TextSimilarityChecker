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
    reportText += "Average Similarity: " + parseFloat(data.average_similarity).toFixed(2) + "%\n";
    reportText += "Total Matching Phrases: " + data.total_matches + "\n";
    
    // Classification information
    if (data.classification) {
        reportText += "\nCLASSIFICATION:\n";
        reportText += "---------------\n";
        reportText += "Primary Category: " + data.classification.primary_category + "\n";
        
        if (data.classification.top_categories && data.classification.top_categories.length > 0) {
            reportText += "Top Categories:\n";
            data.classification.top_categories.forEach(category => {
                if (Array.isArray(category) && category.length >= 2) {
                    reportText += "  - " + category[0] + " (" + parseFloat(category[1]).toFixed(1) + ")\n";
                }
            });
        }
        
        if (data.classification.search_terms && data.classification.search_terms.length > 0) {
            reportText += "\nSearch Terms Used:\n";
            data.classification.search_terms.forEach(term => {
                reportText += "  - " + term + "\n";
            });
        }
    }
    
    reportText += "\n";
    
    // Sources section
    reportText += "MATCHING SOURCES:\n";
    reportText += "----------------\n";
    
    if (data.sources.length === 0) {
        reportText += "No matching sources found.\n\n";
    } else {
        data.sources.forEach((source, index) => {
            reportText += (index + 1) + ". " + source.title + "\n";
            reportText += "   URL: " + source.url + "\n";
            
            // Add category information
            if (source.category_tag) {
                reportText += "   Category: " + source.category_tag + "\n";
            }
            
            // Add relevance score information
            if (source.relevance_score) {
                const relevanceScore = parseInt(source.relevance_score);
                let relevanceText = 'Low';
                if (relevanceScore > 2000) relevanceText = 'Extremely High';
                else if (relevanceScore > 1000) relevanceText = 'Very High';
                else if (relevanceScore > 500) relevanceText = 'High';
                else if (relevanceScore > 250) relevanceText = 'Medium';
                
                reportText += "   Relevance: " + relevanceText + "\n";
            }
            
            reportText += "   Similarity: " + parseFloat(source.similarity).toFixed(2) + "%\n";
            reportText += "   Matching Phrases: " + source.matches.length + "\n\n";
            
            // List matching phrases
            if (source.matches.length > 0) {
                reportText += "   Matching Content:\n";
                source.matches.forEach((match, mIndex) => {
                    reportText += "   - " + match.original + "\n";
                });
                reportText += "\n";
            }
            
            // Add semantic matches if available
            if (source.semantic_matches && source.semantic_matches.length > 0) {
                reportText += "   Semantic Matches (Possible Paraphrasing):\n";
                source.semantic_matches.forEach((match, mIndex) => {
                    if (Array.isArray(match) && match.length >= 2) {
                        reportText += "   - Original: " + match[0] + "\n";
                        reportText += "     Similar to: " + match[1] + "\n";
                    }
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
        
        let data;
        try {
            data = JSON.parse(similarityDataElem.value);
        } catch (e) {
            console.warn("Failed to parse similarity data:", e);
            return;
        }
        
        if (!data || data.length === 0) return;
        
        // Create data for chart
        const labels = data.map(item => {
            // Check if item has required properties
            if (!item || !item.source) {
                return 'Unknown Source';
            }
            // Get category tag if available - safely check if property exists
            const categoryTag = (item.category_tag && typeof item.category_tag === 'string') ? 
                `[${item.category_tag}] ` : '';
            
            // Truncate long titles
            let title = item.source.title || 'Unnamed Source';
            title = title.length > 25 ? title.substring(0, 22) + '...' : title;
            
            // Combine category and title - show category prominently
            return categoryTag + title;
        });
        
        const values = data.map(item => {
            const similarity = parseFloat(item.similarity) || 0;
            return similarity.toFixed(2);
        });
        
        const backgroundColors = data.map(item => {
            // Color based on category - use category-specific colors
            if (item.category_tag) {
                const category = item.category_tag.toLowerCase();
                
                // Match our CSS colors
                if (category === 'wikipedia') return '#28a745'; // Wikipedia - green
                if (category === 'biology') return '#6610f2'; // Biology - purple
                if (category === 'history') return '#fd7e14'; // History - orange
                if (category === 'literature') return '#6f42c1'; // Literature - violet
                if (category === 'technology') return '#20c997'; // Technology - teal
                if (category === 'science') return '#0dcaf0'; // Science - cyan
                if (category === 'medical') return '#dc3545'; // Medical - red
                if (category === 'academic') return '#6c757d'; // Academic - gray
            }
            
            // Fallback to relevance score
            const relevanceScore = item.relevance_score ? parseInt(item.relevance_score) : 0;
            if (relevanceScore > 2000) {
                return '#dc3545'; // Critical match - red
            }
            if (relevanceScore > 1000) {
                return '#9c27b0'; // Very high relevance - purple
            }
            if (relevanceScore > 500) {
                return '#fd7e14'; // High relevance - orange
            }
            
            // Fallback to similarity-based colors
            const similarity = parseFloat(item.similarity) || 0;
            if (similarity > 50) return '#dc3545'; // high - red
            if (similarity > 30) return '#fd7e14'; // medium - orange
            return '#28a745'; // low - green
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
                                try {
                                    const index = tooltipItems[0].dataIndex;
                                    if (data && data[index] && data[index].source) {
                                        return data[index].source.title || 'Unnamed Source';
                                    }
                                    return 'Source';
                                } catch (e) {
                                    console.warn('Error getting tooltip title:', e);
                                    return 'Source';
                                }
                            },
                            afterTitle: function(tooltipItems) {
                                try {
                                    const index = tooltipItems[0].dataIndex;
                                    if (!data || !data[index]) return '';
                                    
                                    // Get URL safely
                                    const url = (data[index].source && data[index].source.url) ? 
                                        data[index].source.url : '';
                                    
                                    // Add category tag if available
                                    let category = '';
                                    if (data[index].category_tag && typeof data[index].category_tag === 'string') {
                                        category = `\nCategory: ${data[index].category_tag}`;
                                    }
                                    
                                    // Add relevance score if available
                                    let relevance = '';
                                    if (data[index].relevance_score) {
                                        const relevanceScore = parseInt(data[index].relevance_score) || 0;
                                        let relevanceText = 'Low';
                                        if (relevanceScore > 2000) relevanceText = 'Extremely High';
                                        else if (relevanceScore > 1000) relevanceText = 'Very High';
                                        else if (relevanceScore > 500) relevanceText = 'High';
                                        else if (relevanceScore > 250) relevanceText = 'Medium';
                                        
                                        relevance = `\nRelevance: ${relevanceText}`;
                                    }
                                    
                                    return url + category + relevance;
                                } catch (e) {
                                    console.warn('Error getting tooltip afterTitle:', e);
                                    return '';
                                }
                            },
                            label: function(context) {
                                try {
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    if (context.parsed.y !== null) {
                                        label += context.parsed.y + '%';
                                    }
                                    
                                    // Add match count if available
                                    const index = context.dataIndex;
                                    if (data && data[index] && data[index].matches && 
                                        Array.isArray(data[index].matches) && data[index].matches.length) {
                                        label += ` (${data[index].matches.length} matches)`;
                                    }
                                    
                                    return label;
                                } catch (e) {
                                    console.warn('Error getting tooltip label:', e);
                                    return 'Similarity';
                                }
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
        let allMatches = [];
        
        try {
            // Try to parse the results data first
            const resultsData = document.getElementById('similarity-data');
            if (resultsData) {
                const results = JSON.parse(resultsData.value);
                
                if (Array.isArray(results)) {
                    // Extract all matches from each result
                    results.forEach(result => {
                        if (result && result.matches && Array.isArray(result.matches)) {
                            allMatches = allMatches.concat(result.matches);
                        }
                    });
                    
                    // If we found matches, use those
                    if (allMatches.length > 0) {
                        matches = allMatches;
                    } else {
                        // Otherwise try to parse the direct matches data
                        try {
                            const directMatches = JSON.parse(matchesValue);
                            if (Array.isArray(directMatches)) {
                                matches = directMatches;
                            }
                        } catch (innerError) {
                            console.warn('Could not parse direct matches data:', innerError);
                        }
                    }
                }
            }
        } catch (parseError) {
            console.warn('Could not parse results data, trying direct matches:', parseError);
            try {
                const directMatches = JSON.parse(matchesValue);
                if (Array.isArray(directMatches)) {
                    matches = directMatches;
                }
            } catch (directError) {
                console.warn('Could not parse matches data at all, using empty array:', directError);
            }
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