// Main JavaScript for Naive Bayes Classifier Web Application

let modelInfo = null;
let featureNames = [];
let currentSmoothingFactor = 1.0;  // Add smoothing factor variable

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeSmoothScroll();
    initializeSmoothingFactor();  // Initialize smoothing factor functionality
});

// Initialize smoothing factor functionality
function initializeSmoothingFactor() {
    const smoothingInput = document.getElementById('smoothingFactor');
    const applyButton = document.getElementById('applySmoothing');
    const currentValueDisplay = document.getElementById('currentSmoothingValue');
    const smoothingValueElements = document.querySelectorAll('.smoothing-value');
    
    if (smoothingInput && applyButton && currentValueDisplay) {
        // Set initial value
        smoothingInput.value = currentSmoothingFactor;
        currentValueDisplay.textContent = currentSmoothingFactor.toFixed(1);
        
        // Update all dataset cards
        smoothingValueElements.forEach(el => {
            el.textContent = currentSmoothingFactor.toFixed(1);
        });
        
        // Apply button event listener
        applyButton.addEventListener('click', function() {
            const newValue = parseFloat(smoothingInput.value);
            if (!isNaN(newValue) && newValue >= 0) {
                currentSmoothingFactor = newValue;
                currentValueDisplay.textContent = currentSmoothingFactor.toFixed(1);
                
                // Update all dataset cards
                smoothingValueElements.forEach(el => {
                    el.textContent = currentSmoothingFactor.toFixed(1);
                });
                
                showAlert(`Smoothing factor updated to ${currentSmoothingFactor.toFixed(1)}`, 'info');
            } else {
                showAlert('Please enter a valid non-negative number for smoothing factor', 'warning');
            }
        });
        
        // Also update when input changes
        smoothingInput.addEventListener('change', function() {
            const newValue = parseFloat(smoothingInput.value);
            if (!isNaN(newValue) && newValue >= 0) {
                currentSmoothingFactor = newValue;
                currentValueDisplay.textContent = currentSmoothingFactor.toFixed(1);
                
                // Update all dataset cards
                smoothingValueElements.forEach(el => {
                    el.textContent = currentSmoothingFactor.toFixed(1);
                });
            }
        });
    }
}

// Initialize event listeners
function initializeEventListeners() {
    // CSV Upload Form
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleCSVUpload);
    }
    
    // Sample Dataset Cards
    document.querySelectorAll('.sample-dataset-card[data-dataset-id]').forEach(card => {
        card.addEventListener('click', handleSampleDatasetClick);
    });
    
    // Upload CSV Card - Show upload form when clicked
    const uploadCSVCard = document.getElementById('uploadCSVCard');
    if (uploadCSVCard) {
        uploadCSVCard.addEventListener('click', function() {
            const uploadFormContainer = document.getElementById('uploadFormContainer');
            if (uploadFormContainer) {
                uploadFormContainer.style.display = 'block';
                // Scroll to upload form
                uploadFormContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        });
    }
    
    // Close Upload Form Button
    const closeUploadForm = document.getElementById('closeUploadForm');
    if (closeUploadForm) {
        closeUploadForm.addEventListener('click', function() {
            const uploadFormContainer = document.getElementById('uploadFormContainer');
            if (uploadFormContainer) {
                uploadFormContainer.style.display = 'none';
            }
        });
    }
    
    // Toggle Dataset Display
    const toggleDatasetDisplay = document.getElementById('toggleDatasetDisplay');
    if (toggleDatasetDisplay) {
        toggleDatasetDisplay.addEventListener('click', function() {
            const tableContainer = document.getElementById('datasetTableContainer');
            const icon = this.querySelector('i');
            if (tableContainer.style.display === 'none') {
                tableContainer.style.display = 'block';
                icon.className = 'fas fa-chevron-up';
            } else {
                tableContainer.style.display = 'none';
                icon.className = 'fas fa-chevron-down';
            }
        });
    }
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
            body: JSON.stringify({ 
                dataset_id: datasetId,
                smoothing_factor: currentSmoothingFactor  // Include smoothing factor in request
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update training status
            updateTrainingStatus('Model trained successfully!', 'success');
            
            modelInfo = data.data;
            featureNames = data.data.feature_names;
            
            // Show model status with training confirmation
            showModelStatus(data.data, data.data.dataset_name, data.data.description, data.data.smoothing_factor);
            displayDatasetTable(data.data.dataset_rows, data.data.feature_names, data.data.target_column);
            
            showAlert(`${data.data.dataset_name} loaded and model trained successfully with smoothing factor ${data.data.smoothing_factor}!`, 'success');
            
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
                        c.innerHTML = `<div class="sample-icon"><i class="fas fa-cloud-sun"></i></div><h6>Weather Classification</h6><p class="small">Predict whether player will play tennis based on weather conditions</p><div class="mt-2"><span class="badge bg-info">Smoothing: <span class="smoothing-value">${currentSmoothingFactor.toFixed(1)}</span></span></div>`;
                    } else if (cardId === 'email') {
                        c.innerHTML = `<div class="sample-icon"><i class="fas fa-envelope"></i></div><h6>Email Spam Detection</h6><p class="small">Classify emails as spam or ham</p><div class="mt-2"><span class="badge bg-info">Smoothing: <span class="smoothing-value">${currentSmoothingFactor.toFixed(1)}</span></span></div>`;
                    } else if (cardId === 'customer') {
                        c.innerHTML = `<div class="sample-icon"><i class="fas fa-shopping-cart"></i></div><h6>Customer Purchase</h6><p class="small">Predict customer purchase behavior</p><div class="mt-2"><span class="badge bg-info">Smoothing: <span class="smoothing-value">${currentSmoothingFactor.toFixed(1)}</span></span></div>`;
                    }
                }
            });
            
            // Restore clicked card content with checkmark
            if (datasetId === 'weather') {
                card.innerHTML = `<div class="sample-icon"><i class="fas fa-cloud-sun"></i></div><h6>Weather Classification <i class="fas fa-check-circle text-success ms-1"></i></h6><p class="small">Model trained and ready</p><div class="mt-2"><span class="badge bg-info">Smoothing: <span class="smoothing-value">${currentSmoothingFactor.toFixed(1)}</span></span></div>`;
            } else if (datasetId === 'email') {
                card.innerHTML = `<div class="sample-icon"><i class="fas fa-envelope"></i></div><h6>Email Spam Detection <i class="fas fa-check-circle text-success ms-1"></i></h6><p class="small">Model trained and ready</p><div class="mt-2"><span class="badge bg-info">Smoothing: <span class="smoothing-value">${currentSmoothingFactor.toFixed(1)}</span></span></div>`;
            } else if (datasetId === 'customer') {
                card.innerHTML = `<div class="sample-icon"><i class="fas fa-shopping-cart"></i></div><h6>Customer Purchase <i class="fas fa-check-circle text-success ms-1"></i></h6><p class="small">Model trained and ready</p><div class="mt-2"><span class="badge bg-info">Smoothing: <span class="smoothing-value">${currentSmoothingFactor.toFixed(1)}</span></span></div>`;
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

// Show Model Status with smoothing factor
function showModelStatus(data, datasetName, description, smoothingFactor) {
    const modelStatus = document.getElementById('modelStatus');
    const modelInfoDiv = document.getElementById('modelInfo');
    
    if (modelStatus && modelInfoDiv) {
        modelInfoDiv.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <p><strong>Dataset:</strong> ${datasetName}</p>
                    <p><strong>Description:</strong> ${description}</p>
                    <p><strong>Features:</strong> ${data.n_features}</p>
                    <p><strong>Samples:</strong> ${data.n_samples}</p>
                </div>
                <div class="col-md-6">
                    <p><strong>Classes:</strong> ${data.classes.join(', ')}</p>
                    <p><strong>Training Accuracy:</strong> ${(data.training_accuracy * 100).toFixed(2)}%</p>
                    <p><strong>Smoothing Factor:</strong> ${smoothingFactor.toFixed(2)}</p>
                    <p><strong>Feature Types:</strong> ${data.feature_types.join(', ')}</p>
                </div>
            </div>
        `;
        modelStatus.style.display = 'block';
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
            // Hide dataset display for CSV uploads (will be shown after loading probabilities)
            document.getElementById('datasetDisplay').style.display = 'none';
            // Hide upload form after successful upload
            const uploadFormContainer = document.getElementById('uploadFormContainer');
            if (uploadFormContainer) {
                uploadFormContainer.style.display = 'none';
            }
            // Reset form
            document.getElementById('uploadForm').reset();
            // Update upload card to show success
            const uploadCSVCard = document.getElementById('uploadCSVCard');
            if (uploadCSVCard) {
                uploadCSVCard.innerHTML = `<div class="sample-icon"><i class="fas fa-file-upload"></i></div><h6>Upload CSV File <i class="fas fa-check-circle text-success ms-1"></i></h6><p class="small">Model trained and ready</p>`;
                uploadCSVCard.style.borderColor = 'rgba(139, 92, 246, 0.8)';
                uploadCSVCard.style.boxShadow = '0 0 15px rgba(139, 92, 246, 0.5)';
            }
            showAlert('Model trained successfully!', 'success');
            
            // Load feature probabilities
            await loadFeatureProbabilities();
            
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
    
    // Scroll to dataset display smoothly
    setTimeout(() => {
        displayDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
    
    // Load and display feature probabilities after dataset is shown
    loadFeatureProbabilities();
}

// Load Feature Probabilities
async function loadFeatureProbabilities() {
    const probabilitiesDiv = document.getElementById('featureProbabilities');
    const probabilitiesContainer = document.getElementById('featureProbabilitiesContainer');
    
    if (!modelInfo) {
        probabilitiesDiv.style.display = 'none';
        return;
    }
    
    try {
        const response = await fetch('/api/model-info');
        const modelData = await response.json();
        
        if (modelData.error) {
            probabilitiesDiv.style.display = 'none';
            return;
        }
        
        // Generate feature probability HTML
        let probabilitiesHtml = `
            <div class="row">
                <div class="col-12">
                    <p class="text-muted mb-3">
                        <i class="fas fa-info-circle me-1"></i>
                        These are the calculated probabilities for each feature based on the trained model.
                    </p>
                </div>
        `;
        
        // Display class probabilities
        probabilitiesHtml += `
            <div class="col-12 mb-4">
                <div class="card bg-dark border-primary">
                    <div class="card-header bg-gradient-primary text-white">
                        <h6 class="mb-0"><i class="fas fa-chart-pie me-2"></i>Class Probabilities</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
        `;
        
        Object.entries(modelData.class_probs).forEach(([className, prob]) => {
            probabilitiesHtml += `
                <div class="col-md-6 col-lg-4 mb-2">
                    <div class="d-flex justify-content-between">
                        <span>${className}:</span>
                        <span class="fw-bold">${(prob * 100).toFixed(2)}%</span>
                    </div>
                    <div class="progress mt-1" style="height: 8px;">
                        <div class="progress-bar bg-primary" role="progressbar" style="width: ${(prob * 100)}%"></div>
                    </div>
                </div>
            `;
        });
        
        probabilitiesHtml += `
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Display feature probabilities for each class
        Object.entries(modelData.feature_params).forEach(([className, features]) => {
            probabilitiesHtml += `
                <div class="col-12 mb-4">
                    <div class="card bg-dark border-info">
                        <div class="card-header bg-gradient-info text-white">
                            <h6 class="mb-0"><i class="fas fa-calculator me-2"></i>Feature Probabilities for Class: ${className}</h6>
                        </div>
                        <div class="card-body">
            `;
            
            // Loop through features
            for (let featIdx = 0; featIdx < modelData.n_features; featIdx++) {
                const featureName = modelData.feature_names[featIdx];
                const featureType = modelData.feature_types[featIdx];
                const featureParams = features[featIdx];
                
                if (featureType === 'numerical') {
                    // Numerical feature - show mean and std
                    probabilitiesHtml += `
                        <div class="mb-3 p-3 bg-dark rounded">
                            <h6 class="text-primary">${featureName} <span class="badge bg-secondary">Numerical</span></h6>
                            <div class="row">
                                <div class="col-md-6">
                                    <p class="mb-1"><strong>Mean (μ):</strong> ${featureParams.mean.toFixed(4)}</p>
                                </div>
                                <div class="col-md-6">
                                    <p class="mb-1"><strong>Standard Deviation (σ):</strong> ${featureParams.std.toFixed(4)}</p>
                                </div>
                            </div>
                            <div class="mt-2">
                                <small class="text-muted">
                                    <i class="fas fa-info-circle me-1"></i>
                                    For numerical features, we use Gaussian distribution: P(x|class) = (1/√(2πσ²)) × e^(-½((x-μ)/σ)²)
                                </small>
                            </div>
                        </div>
                    `;
                } else {
                    // Categorical feature - show probability distribution
                    probabilitiesHtml += `
                        <div class="mb-3 p-3 bg-dark rounded">
                            <h6 class="text-primary">${featureName} <span class="badge bg-secondary">Categorical</span></h6>
                            <div class="row">
                    `;
                    
                    Object.entries(featureParams.prob_dist).forEach(([value, prob]) => {
                        probabilitiesHtml += `
                            <div class="col-md-6 col-lg-4 mb-2">
                                <div class="d-flex justify-content-between">
                                    <span>${value}:</span>
                                    <span class="fw-bold">${(prob * 100).toFixed(2)}%</span>
                                </div>
                                <div class="progress mt-1" style="height: 8px;">
                                    <div class="progress-bar bg-info" role="progressbar" style="width: ${(prob * 100)}%"></div>
                                </div>
                            </div>
                        `;
                    });
                    
                    probabilitiesHtml += `
                            </div>
                            <div class="mt-2">
                                <small class="text-muted">
                                    <i class="fas fa-info-circle me-1"></i>
                                    For categorical features, probabilities are calculated with smoothing: P(value|class) = (count + smoothing_factor) / (total + smoothing_factor × unique_values)
                                </small>
                            </div>
                        </div>
                    `;
                }
            }
            
            probabilitiesHtml += `
                        </div>
                    </div>
                </div>
            `;
        });
        
        probabilitiesHtml += `
            </div>
        `;
        
        probabilitiesContainer.innerHTML = probabilitiesHtml;
        probabilitiesDiv.style.display = 'block';
        
        // Scroll to feature probabilities section
        setTimeout(() => {
            probabilitiesDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 300);
        
    } catch (error) {
        console.error('Error loading feature probabilities:', error);
        probabilitiesDiv.style.display = 'none';
    }
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
                ...Object.keys(data.categorical_features || {}).map(idx => `categoricalChart_${idx}`)
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
            ...Array.from(document.querySelectorAll('[id^="categoricalChart_"]')).map(el => el.id)
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
        data.categorical_features ? Object.keys(data.categorical_features).length : 0
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
                    bgcolor: 'rgba(0,0,0,0)'
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { t: 50, b: 50, l: 50, r: 150 }
            };
            
            const div = document.createElement('div');
            div.className = 'viz-card';
            div.id = 'classDistributionChart';
            div.style.minHeight = '400px';
            grid.appendChild(div);
            
            // Wait for DOM to be ready
            setTimeout(() => {
                Plotly.newPlot('classDistributionChart', [pieTrace], pieLayout, {
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d']
                }).then(() => {
                    console.log('Class distribution chart rendered successfully');
                }).catch(err => {
                    console.error('Error plotting class distribution chart:', err);
                    div.innerHTML = `<div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        Could not generate class distribution chart: ${err.message}
                    </div>`;
                });
            }, 100);
            
            chartsCreated++;
        } catch (error) {
            console.error('Error processing class distribution:', error);
        }
    }
    
    // 2. Numerical Features
    if (data.numerical_features) {
        Object.keys(data.numerical_features).forEach(featIdx => {
            try {
                const feat = data.numerical_features[featIdx];
                const traces = [];
                
                Object.keys(feat.classes || {}).forEach((cls, idx) => {
                    const clsData = feat.classes[cls];
                    if (clsData && clsData.data && clsData.data.length > 0) {
                        traces.push({
                            x: clsData.data,
                            type: 'histogram',
                            name: `Class ${cls}`,
                            opacity: 0.7,
                            marker: {
                                color: idx === 0 ? '#667eea' : 
                                       idx === 1 ? '#764ba2' : 
                                       idx === 2 ? '#10b981' : 
                                       idx === 3 ? '#f59e0b' : 
                                       idx === 4 ? '#ef4444' : 
                                       idx === 5 ? '#ec4899' : '#14b8a6'
                            },
                            hovertemplate: '<b>Class:</b> %{name}<br><b>Value:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
                        });
                    }
                });
                
                if (traces.length === 0) {
                    return; // Skip if no valid traces
                }
                
                // Calculate max value for y-axis range
                let maxValue = 1;
                try {
                    traces.forEach(trace => {
                        if (trace.x && trace.x.length > 0) {
                            const traceMax = Math.max(...trace.x.filter(x => isFinite(x)));
                            if (isFinite(traceMax) && traceMax > maxValue) {
                                maxValue = traceMax;
                            }
                        }
                    });
                } catch (e) {
                    console.error('Error calculating max value:', e);
                    maxValue = 1;
                }
                
                const layout = {
                    title: `Numerical: ${feat.name}`,
                    xaxis: { 
                        title: 'Values',
                        color: '#e0e0e0',
                        gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    yaxis: { 
                        title: 'Frequency',
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
    
}

// Setup Prediction Form
async function setupPredictionForm() {
    if (!featureNames || featureNames.length === 0) return;
    
    const container = document.getElementById('predictionFormContainer');
    
    try {
        // Fetch model information to determine feature types
        const response = await fetch('/api/model-info');
        const modelInfo = await response.json();
        
        if (modelInfo.error) {
            console.error('Error fetching model info:', modelInfo.error);
            return;
        }
        
        // Create input fields based on feature types
        let inputFieldsHtml = '';
        
        for (let i = 0; i < featureNames.length; i++) {
            const featureName = featureNames[i];
            const featureType = modelInfo.feature_types[i];
            
            if (featureType === 'categorical') {
                // For categorical features, create dropdown with unique values
                // Extract unique values from feature parameters
                const uniqueValues = new Set();
                Object.values(modelInfo.feature_params).forEach(classParams => {
                    if (classParams[i] && classParams[i].prob_dist) {
                        Object.keys(classParams[i].prob_dist).forEach(value => {
                            uniqueValues.add(value);
                        });
                    }
                });
                
                const valuesArray = Array.from(uniqueValues);
                
                inputFieldsHtml += `
                    <div class="col-md-6 mb-3">
                        <label class="form-label">${featureName} <span class="badge bg-secondary">Categorical</span></label>
                        <select class="form-select feature-input" data-feature="${i}">
                            <option value="">Select ${featureName}</option>
                            ${valuesArray.map(value => `<option value="${value}">${value}</option>`).join('')}
                        </select>
                    </div>
                `;
            } else {
                // For numerical features, keep text input
                inputFieldsHtml += `
                    <div class="col-md-6 mb-3">
                        <label class="form-label">${featureName} <span class="badge bg-secondary">Numerical</span></label>
                        <input type="text" class="form-control feature-input" 
                               data-feature="${i}" placeholder="Enter ${featureName}">
                    </div>
                `;
            }
        }
        
        container.innerHTML = `
            <div class="mb-4">
                <h4>Enter Test Data</h4>
                <p class="text-muted">Enter values for each feature (one sample per row)</p>
            </div>
            <div id="predictionInputs">
                <div class="prediction-form-row">
                    <h5>Sample 1</h5>
                    <div class="row">
                        ${inputFieldsHtml}
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
        
        // Attach event listeners after DOM is updated
        setTimeout(attachPredictionEventListeners, 0);
        
    } catch (error) {
        console.error('Error setting up prediction form:', error);
        // Fallback to original text inputs if there's an error
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
        
        // Attach event listeners after DOM is updated
        setTimeout(attachPredictionEventListeners, 0);
    }
}

// Attach event listeners for prediction form buttons
function attachPredictionEventListeners() {
    const addSampleBtn = document.getElementById('addSampleBtn');
    const predictBtn = document.getElementById('predictBtn');
    const clearBtn = document.getElementById('clearBtn');
    
    if (addSampleBtn) {
        // Remove existing event listeners to prevent duplicates
        const newAddSampleBtn = addSampleBtn.cloneNode(true);
        addSampleBtn.parentNode.replaceChild(newAddSampleBtn, addSampleBtn);
        newAddSampleBtn.addEventListener('click', addPredictionSample);
    }
    
    if (predictBtn) {
        // Remove existing event listeners to prevent duplicates
        const newPredictBtn = predictBtn.cloneNode(true);
        predictBtn.parentNode.replaceChild(newPredictBtn, predictBtn);
        newPredictBtn.addEventListener('click', makePredictions);
    }
    
    if (clearBtn) {
        // Remove existing event listeners to prevent duplicates
        const newClearBtn = clearBtn.cloneNode(true);
        clearBtn.parentNode.replaceChild(newClearBtn, clearBtn);
        newClearBtn.addEventListener('click', clearPredictions);
    }
}

// Add Prediction Sample
function addPredictionSample() {
    const container = document.getElementById('predictionInputs');
    const sampleNum = container.children.length + 1;
    
    // Create a temporary element to get the feature inputs from the first sample
    const firstSample = container.querySelector('.prediction-form-row');
    const firstSampleInputs = firstSample.querySelectorAll('.feature-input');
    
    let inputFieldsHtml = '';
    
    firstSampleInputs.forEach(input => {
        const featureIndex = input.getAttribute('data-feature');
        const featureName = input.previousElementSibling ? input.previousElementSibling.textContent.replace(' Categorical', '').replace(' Numerical', '') : `Feature ${parseInt(featureIndex) + 1}`;
        const isCategorical = input.previousElementSibling && input.previousElementSibling.querySelector('.badge') && 
                             input.previousElementSibling.querySelector('.badge').textContent === 'Categorical';
        
        if (isCategorical) {
            // For categorical features, copy the select element
            const selectHtml = input.outerHTML.replace(/Sample 1/g, `Sample ${sampleNum}`);
            inputFieldsHtml += `
                <div class="col-md-6 mb-3">
                    <label class="form-label">${featureName} <span class="badge bg-secondary">Categorical</span></label>
                    ${selectHtml}
                </div>
            `;
        } else {
            // For numerical features, create text input
            inputFieldsHtml += `
                <div class="col-md-6 mb-3">
                    <label class="form-label">${featureName} <span class="badge bg-secondary">Numerical</span></label>
                    <input type="text" class="form-control feature-input" 
                           data-feature="${featureIndex}" placeholder="Enter ${featureName}">
                </div>
            `;
        }
    });
    
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
            ${inputFieldsHtml}
        </div>
    `;
    
    container.appendChild(sampleDiv);
    
    // Add event listener to the remove button
    sampleDiv.querySelector('.remove-sample').addEventListener('click', function() {
        sampleDiv.remove();
    });
}

// Clear Predictions
function clearPredictions() {
    // Instead of manually creating the form, just call setupPredictionForm again
    setupPredictionForm();
    
    document.getElementById('predictionResults').innerHTML = '';
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

// Show Loading
function showLoading(section) {
    // Implementation depends on section
    if (section === 'sample') {
        // Sample datasets don't have buttons, just return
        return;
    }
    
    const buttons = document.querySelectorAll('button[type="submit"], #predictBtn');
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
    
    const buttons = document.querySelectorAll('button[type="submit"], #predictBtn');
    buttons.forEach(btn => {
        btn.disabled = false;
        if (btn.id === 'predictBtn') {
            btn.innerHTML = '<i class="fas fa-magic me-2"></i>Make Predictions';
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
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-dismiss after 5 seconds for success/info alerts
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    alertDiv.remove();
                }
            }, 5000);
        }
    }
}