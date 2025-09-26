// Main JavaScript for Medical Imaging Interface
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.querySelector('.btn-analyze');
    const loadingSpinner = document.getElementById('loadingSpinner');

    // File input change handler
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            handleFileSelection(file);
        }
    });

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (isValidImageFile(file)) {
                fileInput.files = files;
                handleFileSelection(file);
            } else {
                showAlert('Please select a valid image file (JPG, PNG, JPEG)', 'warning');
            }
        }
    });

    // Click to upload functionality
    uploadArea.addEventListener('click', function(e) {
        if (e.target !== fileInput && e.target !== analyzeBtn) {
            fileInput.click();
        }
    });

    // Form submission handler
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput.files[0]) {
            showAlert('Please select an image file first', 'warning');
            return;
        }

        showLoadingState();
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        // Submit form using fetch for better UX
        fetch('/api/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Redirect to show results
            window.location.href = '/predict-result?' + new URLSearchParams({
                prediction: data.prediction,
                confidence: data.confidence,
                img_src: data.img_src,
                is_pneumonia: data.is_pneumonia,
                filename: data.filename
            });
        })
        .catch(error => {
            console.error('Error:', error);
            hideLoadingState();
            showAlert('An error occurred during analysis. Please try again.', 'danger');
        });
    });

    // File selection handler
    function handleFileSelection(file) {
        if (!isValidImageFile(file)) {
            showAlert('Please select a valid image file (JPG, PNG, JPEG)', 'warning');
            return;
        }

        // Update UI to show selected file
        const uploadZone = document.querySelector('.upload-zone');
        const fileName = file.name;
        const fileSize = formatFileSize(file.size);

        uploadZone.innerHTML = `
            <i class="fas fa-image upload-icon" style="color: var(--medical-success);"></i>
            <h4>Image Selected</h4>
            <p><strong>${fileName}</strong> (${fileSize})</p>
            <div class="selected-file-preview mt-3">
                <img src="${URL.createObjectURL(file)}" alt="Preview" style="max-height: 150px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            </div>
            <button type="submit" class="btn btn-analyze mt-3">
                <i class="fas fa-search me-2"></i>Analyze Image
            </button>
            <button type="button" class="btn btn-outline-secondary mt-2" onclick="resetUpload()">
                <i class="fas fa-times me-2"></i>Remove
            </button>
        `;

        // Enable analyze button
        analyzeBtn.disabled = false;
    }

    // Validation function
    function isValidImageFile(file) {
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        const maxSize = 10 * 1024 * 1024; // 10MB

        if (!allowedTypes.includes(file.type)) {
            return false;
        }

        if (file.size > maxSize) {
            showAlert('File size should be less than 10MB', 'warning');
            return false;
        }

        return true;
    }

    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Show loading state
    function showLoadingState() {
        const uploadZone = document.querySelector('.upload-zone');
        uploadZone.style.display = 'none';
        loadingSpinner.style.display = 'block';
        
        // Add progress animation
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress >= 100) {
                progress = 100;
                clearInterval(progressInterval);
            }
            
            const progressText = loadingSpinner.querySelector('p');
            if (progressText) {
                progressText.innerHTML = `Analyzing image with AI... ${Math.round(progress)}%`;
            }
        }, 500);
    }

    // Hide loading state
    function hideLoadingState() {
        loadingSpinner.style.display = 'none';
        document.querySelector('.upload-zone').style.display = 'block';
    }

    // Reset upload area
    window.resetUpload = function() {
        fileInput.value = '';
        const uploadZone = document.querySelector('.upload-zone');
        uploadZone.innerHTML = `
            <i class="fas fa-cloud-upload-alt upload-icon"></i>
            <h4>Drop your X-ray here</h4>
            <p>or click to browse files</p>
            <input type="file" id="fileInput" name="file" accept="image/*" required>
            <button type="submit" class="btn btn-analyze" disabled>
                <i class="fas fa-search me-2"></i>Analyze Image
            </button>
        `;
        
        // Re-attach event listeners
        const newFileInput = document.getElementById('fileInput');
        newFileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                handleFileSelection(file);
            }
        });
    };

    // Show alert function
    function showAlert(message, type = 'info') {
        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.custom-alert');
        existingAlerts.forEach(alert => alert.remove());

        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} custom-alert fade-in`;
        alertDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            min-width: 300px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border-radius: 10px;
        `;
        
        const iconMap = {
            success: 'fas fa-check-circle',
            warning: 'fas fa-exclamation-triangle',
            danger: 'fas fa-times-circle',
            info: 'fas fa-info-circle'
        };

        alertDiv.innerHTML = `
            <i class="${iconMap[type]} me-2"></i>
            ${message}
            <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
        `;

        document.body.appendChild(alertDiv);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentElement) {
                alertDiv.remove();
            }
        }, 5000);
    }

    // Download report function
    window.downloadReport = function() {
        const prediction = document.querySelector('.diagnosis')?.textContent || 'Unknown';
        const confidence = document.querySelector('.confidence-value')?.textContent || '0%';
        const filename = document.querySelector('.filename')?.textContent || 'unknown.jpg';
        
        const reportContent = `
CHEST X-RAY PNEUMONIA DETECTION REPORT
=====================================

Patient File: ${filename}
Analysis Date: ${new Date().toLocaleString()}
AI Model: ResNet50 Deep Learning Model

ANALYSIS RESULTS:
Prediction: ${prediction}
Confidence Level: ${confidence}

INTERPRETATION:
${prediction.toLowerCase().includes('pneumonia') ? 
  'The AI analysis indicates possible signs of pneumonia in the chest X-ray. This result suggests that further medical evaluation by a qualified healthcare professional is recommended.' :
  'The AI analysis shows no obvious signs of pneumonia in the chest X-ray. However, this does not replace professional medical diagnosis and regular health check-ups are still recommended.'
}

IMPORTANT DISCLAIMER:
This automated analysis is for educational and research purposes only. 
It should not be used as the sole basis for medical diagnosis or treatment decisions. 
Always consult with qualified healthcare professionals for proper medical evaluation and diagnosis.

Generated by AI Medical Imaging System
        `.trim();

        const blob = new Blob([reportContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `pneumonia_detection_report_${Date.now()}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showAlert('Report downloaded successfully!', 'success');
    };

    // Smooth scrolling for internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add keyboard accessibility
    document.addEventListener('keydown', function(e) {
        // ESC key to reset upload
        if (e.key === 'Escape' && fileInput.files.length > 0) {
            resetUpload();
        }
        
        // Enter key on upload area to trigger file selection
        if (e.key === 'Enter' && e.target.closest('.upload-zone')) {
            fileInput.click();
        }
    });

    // Add focus indicators for accessibility
    const focusableElements = document.querySelectorAll('button, input, a, [tabindex]');
    focusableElements.forEach(element => {
        element.addEventListener('focus', function() {
            this.style.outline = '2px solid var(--medical-accent)';
            this.style.outlineOffset = '2px';
        });
        
        element.addEventListener('blur', function() {
            this.style.outline = '';
            this.style.outlineOffset = '';
        });
    });

    // Performance monitoring
    if ('performance' in window) {
        window.addEventListener('load', function() {
            const loadTime = performance.now();
            console.log(`Page loaded in ${loadTime.toFixed(2)}ms`);
        });
    }

    // Add animation to results when they appear
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    });

    document.querySelectorAll('.result-card, .info-card, .analysis-complete').forEach(el => {
        observer.observe(el);
    });

    // Set confidence bar width from data attribute
    const confidenceBar = document.querySelector('.confidence-width');
    if (confidenceBar && confidenceBar.dataset.width) {
        setTimeout(() => {
            confidenceBar.style.width = confidenceBar.dataset.width + '%';
        }, 100);
    }

    console.log('🏥 Medical Imaging Interface Loaded Successfully!');
});