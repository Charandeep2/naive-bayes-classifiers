// Main JavaScript for Naive Bayes Classifier Web Application

let modelInfo = null;
let featureNames = [];

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeSmoothScroll();
});

// Initialize event listeners
function initializeEventListeners() {
    // CSV Upload Form
    document.getElementById('uploadForm').addEventListener('submit', handleCSVUpload);
    
    // Manual Training Button
    document.getElementById('trainManualBtn').addEventListener('click', handleManualTraining);
    
    // Sample Dataset Cards
    document.querySelectorAll('.sample-dataset-card').forEach(card => {
        card.addEventListener('click', handleSampleDatasetClick);
    });
}

// Handle Sample Dataset Click
async function handleSampleDatasetClick(e) {
    const card = e.currentTarget;
    const datasetId = card.getAttribute('data-dataset-id');
    
    try {
        // Show training status
        showTrainingStatus('Loading dataset and training model...');
        
        // Disable card during training
        card.style.opacity = '0.6';
        card.style.pointerEvents = 'none';
        
        // Update card to show training
        const originalContent = card.innerHTML;
        card.innerHTML = `
            <div class="text-center">
                <div class="spinner-border spinner-border-sm text-primary mb-2" role="status">
                    <span class="visually-hidden">Training...</span>
                </div>
                <p class="small mb-0">Training Model...</p>
            </div>
        `;
        
        const response = await fetch('/api/load-sample', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ dataset_id: datasetId })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update training status
            updateTrainingStatus('Model trained successfully!', 'success');
            
            modelInfo = data.data;
            featureNames = data.data.feature_names;
            
            // Show model status with training confirmation
            showModelStatus(data.data, data.data.dataset_name, data.data.description);
            displayDatasetTable(data.data.dataset_rows, data.data.feature_names, data.data.target_column);
            
            showAlert(`${data.data.dataset_name} loaded and model trained successfully!`, 'success');
            
            // Ensure visualizations are loaded
            setTimeout(() => {
                loadVisualizations();
            }, 500);
            
            setupPredictionForm();
            
            // Highlight the selected card and restore content
            document.querySelectorAll('.sample-dataset-card').forEach(c => {
                c.style.borderColor = 'rgba(139, 92, 246, 0.3)';
                // Restore other cards to original if needed
                if (c !== card && c.getAttribute('data-dataset-id')) {
                    const cardId = c.getAttribute('data-dataset-id');
                    if (cardId === 'weather') {
                        c.innerHTML = `<div class="sample-icon"><i class="fas fa-cloud-sun"></i></div><h6>Weather Classification</h6><p class="small">Predict tennis play based on weather conditions</p>`;
                    } else if (cardId === 'email') {
                        c.innerHTML = `<div class="sample-icon"><i class="fas fa-envelope"></i></div><h6>Email Spam Detection</h6><p class="small">Classify emails as spam or ham</p>`;
                    } else if (cardId === 'customer') {
                        c.innerHTML = `<div class="sample-icon"><i class="fas fa-shopping-cart"></i></div><h6>Customer Purchase</h6><p class="small">Predict customer purchase behavior</p>`;
                    }
                }
            });
            
            // Restore clicked card content with checkmark
            if (datasetId === 'weather') {
                card.innerHTML = `<div class="sample-icon"><i class="fas fa-cloud-sun"></i></div><h6>Weather Classification <i class="fas fa-check-circle text-success ms-1"></i></h6><p class="small">Model trained and ready</p>`;
            } else if (datasetId === 'email') {
                card.innerHTML = `<div class="sample-icon"><i class="fas fa-envelope"></i></div><h6>Email Spam Detection <i class="fas fa-check-circle text-success ms-1"></i></h6><p class="small">Model trained and ready</p>`;
            } else if (datasetId === 'customer') {
                card.innerHTML = `<div class="sample-icon"><i class="fas fa-shopping-cart"></i></div><h6>Customer Purchase <i class="fas fa-check-circle text-success ms-1"></i></h6><p class="small">Model trained and ready</p>`;
            }
            
            card.style.borderColor = 'rgba(139, 92, 246, 0.8)';
            card.style.boxShadow = '0 0 15px rgba(139, 92, 246, 0.5)';
        } else {
            showAlert(data.error || 'Failed to load and train sample dataset', 'danger');
            updateTrainingStatus('Training failed. Please try again.', 'danger');
            card.innerHTML = originalContent;
        }
    } catch (error) {
        showAlert('Error: ' + error.message, 'danger');
        updateTrainingStatus('Error during training: ' + error.message, 'danger');
        card.innerHTML = originalContent;
    } finally {
        card.style.opacity = '1';
        card.style.pointerEvents = 'auto';
    }
}

// Show Training Status
function showTrainingStatus(message) {
    let statusDiv = document.getElementById('trainingStatus');
    if (!statusDiv) {
        statusDiv = document.createElement('div');
        statusDiv.id = 'trainingStatus';
        statusDiv.className = 'alert alert-info mt-3';
        const trainingSection = document.querySelector('#training .card-body');
        if (trainingSection) {
            trainingSection.insertBefore(statusDiv, trainingSection.querySelector('#modelStatus'));
        }
    }
    
    statusDiv.innerHTML = `
        <div class="d-flex align-items-center">
            <div class="spinner-border spinner-border-sm me-2" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <strong><i class="fas fa-cog fa-spin me-2"></i>Training Model:</strong> ${message}
        </div>
    `;
    statusDiv.style.display = 'block';
}

// Update Training Status
function updateTrainingStatus(message, type = 'success') {
    const statusDiv = document.getElementById('trainingStatus');
    if (statusDiv) {
        const icon = type === 'success' ? 'check-circle' : 'exclamation-circle';
        const alertClass = type === 'success' ? 'alert-success' : 'alert-danger';
        statusDiv.className = `alert ${alertClass} mt-3`;
        statusDiv.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas fa-${icon} me-2"></i>
                <strong>Training Complete:</strong> ${message}
            </div>
        `;
        
        // Auto-hide after 3 seconds if successful
        if (type === 'success') {
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 3000);
        }
    }
}

// Smooth scroll for navigation links
function initializeSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// Handle CSV Upload
async function handleCSVUpload(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('csvFile');
    const targetColumn = document.getElementById('targetColumn').value;
    
    if (!fileInput.files[0]) {
        showAlert('Please select a CSV file', 'danger');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    if (targetColumn) {
        formData.append('target_column', targetColumn);
    }
    
    try {
        showLoading('upload');
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            modelInfo = data.data;
            featureNames = data.data.feature_names;
            showModelStatus(data.data);
            // Hide dataset display for CSV uploads
            document.getElementById('datasetDisplay').style.display = 'none';
            showAlert('Model trained successfully!', 'success');
            // Ensure visualizations are loaded
            setTimeout(() => {
                loadVisualizations();
            }, 500);
            setupPredictionForm();
        } else {
            showAlert(data.error || 'Failed to train model', 'danger');
        }
    } catch (error) {
        showAlert('Error: ' + error.message, 'danger');
    } finally {
        hideLoading('upload');
    }
}

// Handle Manual Training
async function handleManualTraining() {
    const numFeatures = parseInt(document.getElementById('numFeatures').value);
    const featureNamesInput = document.getElementById('featureNames').value;
    const manualData = document.getElementById('manualData').value;
    
    if (!manualData.trim()) {
        showAlert('Please enter training data', 'danger');
        return;
    }
    
    // Parse manual data
    const rows = manualData.trim().split('\n').map(line => {
        return line.split(',').map(x => x.trim());
    });
    
    // Parse feature names
    let featureNamesList = [];
    if (featureNamesInput.trim()) {
        featureNamesList = featureNamesInput.split(',').map(x => x.trim());
    }
    
    if (featureNamesList.length !== numFeatures) {
        featureNamesList = Array(numFeatures).fill(null).map((_, i) => 
            featureNamesList[i] || `Feature_${i+1}`
        );
    }
    
    try {
        showLoading('manual');
        const response = await fetch('/api/manual-data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                rows: rows,
                feature_names: featureNamesList
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            modelInfo = data.data;
            featureNames = data.data.feature_names;
            showModelStatus(data.data);
            // Hide dataset display for manual input
            document.getElementById('datasetDisplay').style.display = 'none';
            showAlert('Model trained successfully!', 'success');
            // Ensure visualizations are loaded
            setTimeout(() => {
                loadVisualizations();
            }, 500);
            setupPredictionForm();
        } else {
            showAlert(data.error || 'Failed to train model', 'danger');
        }
    } catch (error) {
        showAlert('Error: ' + error.message, 'danger');
    } finally {
        hideLoading('manual');
    }
}

// Show Model Status
function showModelStatus(info, datasetName = null, description = null) {
    const statusDiv = document.getElementById('modelStatus');
    const infoDiv = document.getElementById('modelInfo');
    
    let html = '';
    if (datasetName) {
        html += `<div class="mb-3 p-3" style="background: rgba(139, 92, 246, 0.1); border-radius: 10px; border-left: 4px solid #8b5cf6;">
            <h6 class="mb-1"><i class="fas fa-database me-2"></i>${datasetName}</h6>
            <p class="mb-0 small">${description}</p>
        </div>`;
    }
    
    // Show training accuracy if available
    const trainingAccuracy = info.training_accuracy !== undefined ? info.training_accuracy : null;
    
    html += `
        <div class="row">
            <div class="col-md-6">
                <p><strong>Samples:</strong> ${info.n_samples}</p>
                <p><strong>Features:</strong> ${info.n_features}</p>
                <p><strong>Feature Names:</strong> ${info.feature_names.join(', ')}</p>
            </div>
            <div class="col-md-6">
                <p><strong>Classes:</strong> ${info.classes.join(', ')}</p>
                ${trainingAccuracy !== null ? `<p><strong>Training Accuracy:</strong> <span class="badge bg-success">${(trainingAccuracy * 100).toFixed(2)}%</span></p>` : ''}
                <p><strong>Feature Types:</strong> ${info.feature_types.map((t, i) => 
                    `${info.feature_names[i]}: ${t}`
                ).join(', ')}</p>
            </div>
        </div>
        <div class="mt-3 p-2" style="background: rgba(16, 185, 129, 0.1); border-radius: 5px;">
            <small><i class="fas fa-info-circle me-1"></i><strong>Model Status:</strong> Trained and ready for predictions</small>
        </div>
    `;
    
    infoDiv.innerHTML = html;
    statusDiv.style.display = 'block';
    statusDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display Dataset Table
function displayDatasetTable(rows, featureNames, targetColumn) {
    const displayDiv = document.getElementById('datasetDisplay');
    const tableContainer = document.getElementById('datasetTableContainer');
    
    if (!rows || rows.length === 0) {
        displayDiv.style.display = 'none';
        return;
    }
    
    // Create table HTML
    let tableHtml = `
        <div class="table-responsive" style="max-height: 500px; overflow-y: auto;">
            <table class="table table-dark table-striped table-hover" style="margin-bottom: 0;">
                <thead style="position: sticky; top: 0; background: var(--dark-surface-2); z-index: 10;">
                    <tr>
                        <th scope="col" style="color: var(--primary-color);">#</th>
    `;
    
    // Add feature column headers
    featureNames.forEach(name => {
        tableHtml += `<th scope="col" style="color: var(--primary-color);">${name}</th>`;
    });
    
    // Add target column header
    tableHtml += `<th scope="col" style="color: var(--success-color); background: rgba(16, 185, 129, 0.1);">${targetColumn} <i class="fas fa-bullseye ms-1"></i></th>`;
    
    tableHtml += `
                    </tr>
                </thead>
                <tbody>
    `;
    
    // Add data rows
    rows.forEach((row, index) => {
        tableHtml += `<tr>`;
        tableHtml += `<td style="color: var(--text-muted);">${index + 1}</td>`;
        
        // Add feature values
        row.slice(0, -1).forEach((value, idx) => {
            tableHtml += `<td>${value}</td>`;
        });
        
        // Add target value with special styling
        const targetValue = row[row.length - 1];
        tableHtml += `<td style="font-weight: 600; color: var(--success-color); background: rgba(16, 185, 129, 0.1);">${targetValue}</td>`;
        tableHtml += `</tr>`;
    });
    
    tableHtml += `
                </tbody>
            </table>
        </div>
        <div class="mt-3">
            <small class="text-muted">
                <i class="fas fa-info-circle me-1"></i>
                Showing ${rows.length} rows. The last column (${targetColumn}) is the target variable used for classification.
            </small>
        </div>
    `;
    
    tableContainer.innerHTML = tableHtml;
    displayDiv.style.display = 'block';
    
    // Add toggle functionality
    const toggleBtn = document.getElementById('toggleDatasetDisplay');
    let isExpanded = true;
    
    if (toggleBtn) {
        // Remove existing event listeners by cloning
        const newToggleBtn = toggleBtn.cloneNode(true);
        toggleBtn.parentNode.replaceChild(newToggleBtn, toggleBtn);
        
        newToggleBtn.addEventListener('click', function() {
            isExpanded = !isExpanded;
            const tableContainerEl = document.getElementById('datasetTableContainer');
            if (isExpanded) {
                tableContainerEl.style.display = 'block';
                this.innerHTML = '<i class="fas fa-chevron-up"></i>';
            } else {
                tableContainerEl.style.display = 'none';
                this.innerHTML = '<i class="fas fa-chevron-down"></i>';
            }
        });
    }
    
    // Scroll to dataset display smoothly
    setTimeout(() => {
        displayDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// Load Visualizations
async function loadVisualizations() {
    const container = document.getElementById('visualizationContainer');
    const loading = document.getElementById('vizLoading');
    
    // Check if model is trained
    if (!modelInfo || !featureNames || featureNames.length === 0) {
        container.innerHTML = '<div class="text-center text-muted py-5"><i class="fas fa-chart-bar fa-3x mb-3"></i><p>Train a model first to see visualizations</p></div>';
        return;
    }
    
    try {
        loading.style.display = 'block';
        container.innerHTML = '';
        
        const response = await fetch('/api/visualizations');
        const data = await response.json();
        
        if (data.error) {
            container.innerHTML = `<div class="alert alert-danger">
                <i class="fas fa-exclamation-circle me-2"></i>
                ${data.error}
            </div>`;
            loading.style.display = 'none';
            return;
        }
        
        // Check if Plotly is loaded
        if (typeof Plotly === 'undefined') {
            container.innerHTML = '<div class="alert alert-danger">Plotly.js library is not loaded. Please refresh the page.</div>';
            loading.style.display = 'none';
            return;
        }
        
        // Generate visualizations
        generateVisualizations(data);
        
        // Resize all charts after a delay to ensure proper rendering
        setTimeout(() => {
            const chartIds = [
                'classDistributionChart',
                ...Object.keys(data.numerical_features || {}).map(idx => `numericalChart_${idx}`),
                ...Object.keys(data.categorical_features || {}).map(idx => `categoricalChart_${idx}`),
                'confusionMatrixChart'
            ].filter(id => document.getElementById(id));
            
            chartIds.forEach(chartId => {
                try {
                    Plotly.Plots.resize(chartId);
                } catch (e) {
                    console.warn(`Could not resize chart ${chartId}:`, e);
                }
            });
        }, 500);
        
        // Scroll to visualizations section after generation (give time for charts to render)
        setTimeout(() => {
            document.getElementById('visualizations').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 1000);
        
    } catch (error) {
        container.innerHTML = `<div class="alert alert-danger">
            <i class="fas fa-exclamation-circle me-2"></i>
            Error loading visualizations: ${error.message}
        </div>`;
        console.error('Visualization error:', error);
    } finally {
        loading.style.display = 'none';
    }
}

// Handle window resize for Plotly charts
window.addEventListener('resize', function() {
    if (typeof Plotly !== 'undefined') {
        const chartIds = [
            'classDistributionChart',
            ...Array.from(document.querySelectorAll('[id^="numericalChart_"]')).map(el => el.id),
            ...Array.from(document.querySelectorAll('[id^="categoricalChart_"]')).map(el => el.id),
            'confusionMatrixChart'
        ].filter(id => document.getElementById(id));
        
        chartIds.forEach(chartId => {
            try {
                Plotly.Plots.resize(chartId);
            } catch (e) {
                // Silently fail on resize errors
            }
        });
    }
});

// Generate Visualizations using Plotly
function generateVisualizations(data) {
    const container = document.getElementById('visualizationContainer');
    
    if (!container) {
        console.error('Visualization container not found');
        return;
    }
    
    // Clear container and add grid
    container.innerHTML = '<div class="viz-grid"></div>';
    const grid = container.querySelector('.viz-grid');
    
    if (!grid) {
        console.error('Viz grid not found');
        return;
    }
    
    // Count charts to be created
    let chartsCreated = 0;
    const totalCharts = [
        data.class_distribution ? 1 : 0,
        data.numerical_features ? Object.keys(data.numerical_features).length : 0,
        data.categorical_features ? Object.keys(data.categorical_features).length : 0,
        data.performance && data.performance.confusion_matrix ? 1 : 0
    ].reduce((a, b) => a + b, 0);
    
    // Show message if no charts can be created
    if (totalCharts === 0) {
        container.innerHTML = '<div class="alert alert-info"><i class="fas fa-info-circle me-2"></i>No visualization data available. This may occur if the dataset has no valid features.</div>';
        return;
    }
    
    console.log(`Generating ${totalCharts} visualization(s)...`);
    
    // 1. Class Distribution Pie Chart
    if (data.class_distribution) {
        try {
            const classValues = Object.values(data.class_distribution).map(v => Number(v));
            const classLabels = Object.keys(data.class_distribution);
            
            // Filter out invalid values
            const validPairs = classLabels.map((label, idx) => ({
                label,
                value: classValues[idx]
            })).filter(pair => isFinite(pair.value) && pair.value > 0);
            
            if (validPairs.length === 0) {
                console.warn('No valid class distribution data');
                return; // Skip if no valid data
            }
            
            // Ensure we have at least 2 colors for the pie chart
            const colors = ['#667eea', '#764ba2', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#14b8a6'];
            
            const pieTrace = {
                labels: validPairs.map(p => p.label),
                values: validPairs.map(p => p.value),
                type: 'pie',
                hole: 0.35,
                textinfo: 'label+percent',
                textposition: 'auto',
                textfont: {
                    size: 13,
                    color: '#ffffff',
                    family: 'Segoe UI'
                },
                marker: {
                    colors: colors.slice(0, validPairs.length),
                    line: {
                        color: '#1e293b',
                        width: 3
                    }
                },
                hovertemplate: '<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                pull: 0.02  // Slight pull effect for better visibility
            };
            
            const pieLayout = {
                title: {
                    text: 'Class Distribution',
                    font: { size: 18, family: 'Segoe UI', color: '#e0e0e0', weight: 'bold' },
                    x: 0.5,
                    xanchor: 'center',
                    pad: { t: 10, b: 10 }
                },
                showlegend: true,
                legend: {
                    font: { color: '#e0e0e0', size: 13, family: 'Segoe UI' },
                    orientation: 'v',
                    x: 1.15,
                    xanchor: 'left',
                    y: 0.5,
                    yanchor: 'middle',
                    bgcolor: 'rgba(0,0,0,0)',
                    bordercolor: 'rgba(139, 92, 246, 0.3)',
                    borderwidth: 1
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                autosize: true,
                margin: { l: 60, r: 180, t: 60, b: 60 },
                height: 450
            };
            
            const pieDiv = document.createElement('div');
            pieDiv.className = 'viz-card';
            pieDiv.id = 'classDistributionChart';
            pieDiv.style.minHeight = '450px';
            pieDiv.style.width = '100%';
            grid.appendChild(pieDiv);
            
            // Wait a bit for DOM to be ready and ensure Plotly is loaded
            setTimeout(() => {
                if (typeof Plotly === 'undefined') {
                    pieDiv.innerHTML = '<div class="alert alert-danger">Plotly library not loaded. Please refresh the page.</div>';
                    return;
                }
                
                // Ensure div exists before plotting
                if (!document.getElementById('classDistributionChart')) {
                    console.error('Chart div not found');
                    return;
                }
                
                Plotly.newPlot('classDistributionChart', [pieTrace], pieLayout, {
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                    toImageButtonOptions: {
                        format: 'png',
                        filename: 'class_distribution',
                        height: 600,
                        width: 800,
                        scale: 1
                    }
                }).then(() => {
                    console.log('Pie chart rendered successfully');
                    // Force multiple resizes to ensure proper display
                    setTimeout(() => {
                        try {
                            Plotly.Plots.resize('classDistributionChart');
                        } catch (e) {
                            console.warn('Resize error:', e);
                        }
                    }, 200);
                    setTimeout(() => {
                        try {
                            Plotly.Plots.resize('classDistributionChart');
                        } catch (e) {
                            // Ignore
                        }
                    }, 500);
                }).catch(err => {
                    console.error('Error plotting pie chart:', err);
                    const errorDiv = document.getElementById('classDistributionChart');
                    if (errorDiv) {
                        errorDiv.innerHTML = `<div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Could not generate class distribution chart: ${err.message}
                            <br><small>Error details: ${JSON.stringify(err)}</small>
                        </div>`;
                    }
                });
            }, 150);
        } catch (error) {
            console.error('Error processing class distribution:', error);
        }
    }
    
    // 2. Numerical Feature Distributions
    if (data.numerical_features) {
        Object.keys(data.numerical_features).forEach(featIdx => {
            try {
                const feat = data.numerical_features[featIdx];
                const traces = [];
                
                Object.keys(feat.classes).forEach(cls => {
                    const clsData = feat.classes[cls];
                    // Filter out invalid data - more robust filtering
                    const validData = (clsData.data || []).filter(d => {
                        const num = Number(d);
                        return d !== null && d !== undefined && !isNaN(num) && isFinite(num);
                    }).map(d => Number(d));
                    
                    if (validData.length > 0) {
                        traces.push({
                            x: validData,
                            type: 'histogram',
                            name: `Class ${cls}`,
                            opacity: 0.7,
                            nbinsx: Math.min(20, Math.max(5, Math.ceil(Math.sqrt(validData.length))))
                        });
                    }
                });
                
                if (traces.length === 0) {
                    return; // Skip if no valid traces
                }
                
                // Calculate axis ranges with safety checks
                try {
                    const allData = traces.flatMap(t => t.x || []);
                    
                    if (allData.length === 0) {
                        return; // Skip if no data
                    }
                    
                    const minVal = Math.min(...allData);
                    const maxVal = Math.max(...allData);
                    
                    // Handle edge case where all values are the same
                    let xRange = null;
                    if (minVal === maxVal) {
                        // If all values are the same, create a small range around the value
                        const center = minVal;
                        xRange = [center - 1, center + 1];
                    } else {
                        const range = maxVal - minVal;
                        if (range > 0 && isFinite(range)) {
                            xRange = [minVal - range * 0.1, maxVal + range * 0.1];
                        }
                    }
                    
                    const layout = {
                        title: `Distribution: ${feat.name}`,
                        xaxis: { 
                            title: feat.name,
                            color: '#e0e0e0',
                            gridcolor: 'rgba(255,255,255,0.1)',
                            ...(xRange && isFinite(xRange[0]) && isFinite(xRange[1]) ? {
                                range: xRange,
                                autorange: false
                            } : {
                                autorange: true
                            })
                        },
                        yaxis: { 
                            title: 'Frequency',
                            autorange: true,
                            color: '#e0e0e0',
                            gridcolor: 'rgba(255,255,255,0.1)'
                        },
                        barmode: 'overlay',
                        font: { size: 12, family: 'Segoe UI', color: '#e0e0e0' },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    };
                    
                    const div = document.createElement('div');
                    div.className = 'viz-card';
                    div.id = `numericalChart_${featIdx}`;
                    div.style.minHeight = '400px';
                    grid.appendChild(div);
                    
                    // Wait for DOM to be ready
                    setTimeout(() => {
                        Plotly.newPlot(`numericalChart_${featIdx}`, traces, layout, {
                            responsive: true,
                            displayModeBar: true,
                            displaylogo: false,
                            modeBarButtonsToRemove: ['pan2d', 'lasso2d']
                        }).then(() => {
                            console.log(`Numerical chart ${featIdx} rendered successfully`);
                        }).catch(err => {
                            console.error(`Error plotting numerical chart ${featIdx}:`, err);
                            div.innerHTML = `<div class="alert alert-warning">
                                <i class="fas fa-exclamation-triangle me-2"></i>
                                Could not generate chart for ${feat.name}: ${err.message}
                            </div>`;
                        });
                    }, 100);
                } catch (rangeError) {
                    console.error(`Error calculating range for ${feat.name}:`, rangeError);
                    // Use autorange as fallback
                    const layout = {
                        title: `Distribution: ${feat.name}`,
                        xaxis: { 
                            title: feat.name,
                            autorange: true,
                            color: '#e0e0e0',
                            gridcolor: 'rgba(255,255,255,0.1)'
                        },
                        yaxis: { 
                            title: 'Frequency',
                            autorange: true,
                            color: '#e0e0e0',
                            gridcolor: 'rgba(255,255,255,0.1)'
                        },
                        barmode: 'overlay',
                        font: { size: 12, family: 'Segoe UI', color: '#e0e0e0' },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    };
                    
                    const div = document.createElement('div');
                    div.className = 'viz-card';
                    div.id = `numericalChart_${featIdx}`;
                    div.style.minHeight = '400px';
                    grid.appendChild(div);
                    
                    // Wait for DOM to be ready
                    setTimeout(() => {
                        Plotly.newPlot(`numericalChart_${featIdx}`, traces, layout, {
                            responsive: true,
                            displayModeBar: true,
                            displaylogo: false,
                            modeBarButtonsToRemove: ['pan2d', 'lasso2d']
                        }).then(() => {
                            console.log(`Numerical chart ${featIdx} (fallback) rendered successfully`);
                        }).catch(err => {
                            console.error(`Error plotting numerical chart ${featIdx}:`, err);
                            div.innerHTML = `<div class="alert alert-warning">
                                <i class="fas fa-exclamation-triangle me-2"></i>
                                Could not generate chart for ${feat.name}: ${err.message}
                            </div>`;
                        });
                    }, 100);
                }
            } catch (error) {
                console.error(`Error processing numerical feature ${featIdx}:`, error);
            }
        });
    }
    
    // 3. Categorical Features
    if (data.categorical_features) {
        Object.keys(data.categorical_features).forEach(featIdx => {
            try {
                const feat = data.categorical_features[featIdx];
                const traces = [];
                const allValues = new Set();
                
                Object.values(feat.classes || {}).forEach(clsData => {
                    if (clsData && clsData.prob_dist) {
                        Object.keys(clsData.prob_dist).forEach(val => allValues.add(val));
                    }
                });
                
                if (allValues.size === 0) {
                    return; // Skip if no values
                }
                
                const sortedValues = Array.from(allValues).sort();
                
                Object.keys(feat.classes || {}).forEach((cls, idx) => {
                    const clsData = feat.classes[cls];
                    if (clsData && clsData.prob_dist) {
                        const probs = sortedValues.map(val => {
                            const prob = clsData.prob_dist[val];
                            return (prob !== undefined && prob !== null) ? Number(prob) : 0;
                        }).filter(p => isFinite(p) && p >= 0);
                        
                        if (probs.length > 0) {
                            traces.push({
                                x: sortedValues,
                                y: probs,
                                type: 'bar',
                                name: `Class ${cls}`
                            });
                        }
                    }
                });
                
                if (traces.length === 0) {
                    return; // Skip if no valid traces
                }
                
                // Calculate max probability safely
                let maxProb = 0.1;
                try {
                    const allProbs = traces.flatMap(t => (t.y || []).filter(p => isFinite(p) && p >= 0));
                    if (allProbs.length > 0) {
                        const calculatedMax = Math.max(...allProbs);
                        if (isFinite(calculatedMax) && calculatedMax > 0) {
                            maxProb = calculatedMax;
                        }
                    }
                } catch (e) {
                    console.error('Error calculating max probability:', e);
                    maxProb = 0.1;
                }
                
                // Ensure maxProb is valid
                if (!isFinite(maxProb) || maxProb <= 0) {
                    maxProb = 0.1;
                }
                
                const layout = {
                    title: `Categorical: ${feat.name}`,
                    xaxis: { 
                        title: 'Values',
                        type: 'category',
                        color: '#e0e0e0',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    yaxis: { 
                        title: 'Probability',
                        range: [0, Math.max(maxProb * 1.1, 0.1)],
                        color: '#e0e0e0',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    barmode: 'group',
                    font: { size: 12, family: 'Segoe UI', color: '#e0e0e0' },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)'
                };
                
                const div = document.createElement('div');
                div.className = 'viz-card';
                div.id = `categoricalChart_${featIdx}`;
                div.style.minHeight = '400px';
                grid.appendChild(div);
                
                // Wait for DOM to be ready
                setTimeout(() => {
                    Plotly.newPlot(`categoricalChart_${featIdx}`, traces, layout, {
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ['pan2d', 'lasso2d']
                    }).then(() => {
                        console.log(`Categorical chart ${featIdx} rendered successfully`);
                    }).catch(err => {
                        console.error(`Error plotting categorical chart ${featIdx}:`, err);
                        div.innerHTML = `<div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Could not generate chart for ${feat.name}: ${err.message}
                        </div>`;
                    });
                }, 100);
            } catch (error) {
                console.error(`Error processing categorical feature ${featIdx}:`, error);
            }
        });
    }
    
    // 4. Confusion Matrix (if available)
    if (data.performance && data.performance.confusion_matrix) {
        try {
            const cm = data.performance.confusion_matrix;
            const labels = data.performance.class_labels;
            
            if (!cm || !Array.isArray(cm) || cm.length === 0 || !labels || labels.length === 0) {
                return; // Skip if invalid data
            }
            
            // Ensure all values in confusion matrix are numbers
            const validCM = cm.map(row => 
                row.map(val => {
                    const num = Number(val);
                    return isFinite(num) ? num : 0;
                })
            );
            
            const trace = {
                z: validCM,
                x: labels,
                y: labels,
                type: 'heatmap',
                colorscale: 'Blues',
                showscale: true,
                text: validCM.map(row => row.map(val => val.toString())),
                texttemplate: '%{text}',
                textfont: { color: '#ffffff' }
            };
            
            const layout = {
                title: 'Confusion Matrix',
                xaxis: { 
                    title: 'Predicted', 
                    type: 'category',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(255,255,255,0.1)'
                },
                yaxis: { 
                    title: 'Actual', 
                    type: 'category',
                    color: '#e0e0e0',
                    gridcolor: 'rgba(255,255,255,0.1)'
                },
                font: { size: 12, family: 'Segoe UI', color: '#e0e0e0' },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
            };
            
            const div = document.createElement('div');
            div.className = 'viz-card';
            div.id = 'confusionMatrixChart';
            div.style.minHeight = '400px';
            grid.appendChild(div);
            
            // Wait for DOM to be ready
            setTimeout(() => {
                Plotly.newPlot('confusionMatrixChart', [trace], layout, {
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d']
                }).then(() => {
                    console.log('Confusion matrix rendered successfully');
                }).catch(err => {
                    console.error('Error plotting confusion matrix:', err);
                    div.innerHTML = `<div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        Could not generate confusion matrix: ${err.message}
                    </div>`;
                });
            }, 100);
        } catch (error) {
            console.error('Error processing confusion matrix:', error);
        }
    }
}

// Setup Prediction Form
function setupPredictionForm() {
    if (!featureNames || featureNames.length === 0) return;
    
    const container = document.getElementById('predictionFormContainer');
    
    container.innerHTML = `
        <div class="mb-4">
            <h4>Enter Test Data</h4>
            <p class="text-muted">Enter values for each feature (one sample per row)</p>
        </div>
        <div id="predictionInputs">
            <div class="prediction-form-row">
                <h5>Sample 1</h5>
                <div class="row">
                    ${featureNames.map((name, idx) => `
                        <div class="col-md-6 mb-3">
                            <label class="form-label">${name}</label>
                            <input type="text" class="form-control feature-input" 
                                   data-feature="${idx}" placeholder="Enter ${name}">
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
        <div class="d-flex gap-2 mb-3">
            <button class="btn btn-primary" id="addSampleBtn">
                <i class="fas fa-plus me-2"></i>Add Sample
            </button>
            <button class="btn btn-success" id="predictBtn">
                <i class="fas fa-magic me-2"></i>Make Predictions
            </button>
            <button class="btn btn-secondary" id="clearBtn">
                <i class="fas fa-trash me-2"></i>Clear
            </button>
        </div>
    `;
    
    document.getElementById('addSampleBtn').addEventListener('click', addPredictionSample);
    document.getElementById('predictBtn').addEventListener('click', makePredictions);
    document.getElementById('clearBtn').addEventListener('click', clearPredictions);
}

// Add Prediction Sample
function addPredictionSample() {
    const container = document.getElementById('predictionInputs');
    const sampleNum = container.children.length + 1;
    
    const sampleDiv = document.createElement('div');
    sampleDiv.className = 'prediction-form-row';
    sampleDiv.innerHTML = `
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h5>Sample ${sampleNum}</h5>
            <button class="btn btn-sm btn-danger remove-sample">
                <i class="fas fa-times"></i>
            </button>
        </div>
        <div class="row">
            ${featureNames.map((name, idx) => `
                <div class="col-md-6 mb-3">
                    <label class="form-label">${name}</label>
                    <input type="text" class="form-control feature-input" 
                           data-feature="${idx}" placeholder="Enter ${name}">
                </div>
            `).join('')}
        </div>
    `;
    
    container.appendChild(sampleDiv);
    
    sampleDiv.querySelector('.remove-sample').addEventListener('click', function() {
        sampleDiv.remove();
    });
}

// Make Predictions
async function makePredictions() {
    const container = document.getElementById('predictionInputs');
    const samples = container.querySelectorAll('.prediction-form-row');
    
    if (samples.length === 0) {
        showAlert('Please add at least one sample', 'warning');
        return;
    }
    
    const testData = [];
    let isValid = true;
    
    samples.forEach(sample => {
        const inputs = sample.querySelectorAll('.feature-input');
        const row = [];
        
        inputs.forEach(input => {
            const value = input.value.trim();
            if (!value) {
                isValid = false;
                input.classList.add('is-invalid');
            } else {
                // Try to convert to number, otherwise keep as string
                const numValue = parseFloat(value);
                row.push(isNaN(numValue) ? value : numValue);
                input.classList.remove('is-invalid');
            }
        });
        
        if (row.length === featureNames.length) {
            testData.push(row);
        }
    });
    
    if (!isValid || testData.length === 0) {
        showAlert('Please fill all fields', 'danger');
        return;
    }
    
    try {
        showLoading('predictions');
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ test_data: testData })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayPredictions(data.results);
        } else {
            showAlert(data.error || 'Prediction failed', 'danger');
        }
    } catch (error) {
        showAlert('Error: ' + error.message, 'danger');
    } finally {
        hideLoading('predictions');
    }
}

// Display Predictions
function displayPredictions(results) {
    const container = document.getElementById('predictionResults');
    container.innerHTML = '<h4 class="mb-4">Prediction Results</h4>';
    
    results.forEach(result => {
        const card = document.createElement('div');
        card.className = 'prediction-result-card';
        
        const maxProb = result.confidence;
        const confidenceClass = maxProb >= 0.8 ? 'confidence-high' : 
                               maxProb >= 0.6 ? 'confidence-medium' : 'confidence-low';
        
        card.innerHTML = `
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h5>Sample ${result.sample}</h5>
                <span class="confidence-badge ${confidenceClass}">
                    ${(maxProb * 100).toFixed(1)}% Confidence
                </span>
            </div>
            <div class="mb-3">
                <h6 class="text-primary">
                    <i class="fas fa-check-circle me-2"></i>
                    Predicted Class: <strong>${result.prediction}</strong>
                </h6>
            </div>
            <div>
                <h6 class="mb-3">Class Probabilities:</h6>
                ${Object.entries(result.probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([cls, prob]) => `
                        <div class="mb-2">
                            <div class="d-flex justify-content-between mb-1">
                                <span><strong>${cls}</strong></span>
                                <span>${(prob * 100).toFixed(2)}%</span>
                            </div>
                            <div class="progress" style="height: 25px;">
                                <div class="progress-bar" role="progressbar" 
                                     style="width: ${prob * 100}%" 
                                     aria-valuenow="${prob * 100}" 
                                     aria-valuemin="0" 
                                     aria-valuemax="100">
                                    ${(prob * 100).toFixed(1)}%
                                </div>
                            </div>
                        </div>
                    `).join('')}
            </div>
        `;
        
        container.appendChild(card);
    });
    
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Clear Predictions
function clearPredictions() {
    const container = document.getElementById('predictionInputs');
    container.innerHTML = `
        <div class="prediction-form-row">
            <h5>Sample 1</h5>
            <div class="row">
                ${featureNames.map((name, idx) => `
                    <div class="col-md-6 mb-3">
                        <label class="form-label">${name}</label>
                        <input type="text" class="form-control feature-input" 
                               data-feature="${idx}" placeholder="Enter ${name}">
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    document.getElementById('predictionResults').innerHTML = '';
}

// Show Loading
function showLoading(section) {
    // Implementation depends on section
    if (section === 'sample') {
        // Sample datasets don't have buttons, just return
        return;
    }
    
    const buttons = document.querySelectorAll('button[type="submit"], #predictBtn, #trainManualBtn');
    buttons.forEach(btn => {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Loading...';
    });
}

// Hide Loading
function hideLoading(section) {
    if (section === 'sample') {
        return;
    }
    
    const buttons = document.querySelectorAll('button[type="submit"], #predictBtn, #trainManualBtn');
    buttons.forEach(btn => {
        btn.disabled = false;
        if (btn.id === 'predictBtn') {
            btn.innerHTML = '<i class="fas fa-magic me-2"></i>Make Predictions';
        } else if (btn.id === 'trainManualBtn') {
            btn.innerHTML = '<i class="fas fa-play me-2"></i>Train Model';
        } else {
            btn.innerHTML = btn.innerHTML.replace(/Loading\.\.\./g, '').replace(/<span[^>]*><\/span>/g, '');
            if (btn.textContent.includes('Upload')) {
                btn.innerHTML = '<i class="fas fa-upload me-2"></i>Upload & Train';
            }
        }
    });
}

// Show Alert
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

