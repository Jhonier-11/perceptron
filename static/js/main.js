// JavaScript personalizado para la aplicación del perceptrón

document.addEventListener('DOMContentLoaded', function() {
    // Inicializar tooltips de Bootstrap
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Inicializar popovers de Bootstrap
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Auto-ocultar alertas después de 5 segundos
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        if (!alert.querySelector('.btn-close')) {
            setTimeout(() => {
                alert.style.transition = 'opacity 0.5s ease-out';
                alert.style.opacity = '0';
                setTimeout(() => {
                    alert.remove();
                }, 500);
            }, 5000);
        }
    });

    // Validación de formularios en tiempo real
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input, select, textarea');
        
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                validateField(this);
            });
            
            input.addEventListener('input', function() {
                if (this.classList.contains('is-invalid')) {
                    validateField(this);
                }
            });
        });
    });

    // Animaciones de entrada
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            card.style.transition = 'opacity 0.5s ease-out, transform 0.5s ease-out';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
});

// Función para validar campos individuales
function validateField(field) {
    const value = field.value.trim();
    let isValid = true;
    let errorMessage = '';

    // Validaciones específicas según el tipo de campo
    if (field.type === 'number') {
        const min = parseFloat(field.getAttribute('min'));
        const max = parseFloat(field.getAttribute('max'));
        const numValue = parseFloat(value);

        if (value !== '' && (isNaN(numValue) || (min && numValue < min) || (max && numValue > max))) {
            isValid = false;
            if (isNaN(numValue)) {
                errorMessage = 'Debe ser un número válido';
            } else if (min && numValue < min) {
                errorMessage = `El valor mínimo es ${min}`;
            } else if (max && numValue > max) {
                errorMessage = `El valor máximo es ${max}`;
            }
        }
    } else if (field.required && value === '') {
        isValid = false;
        errorMessage = 'Este campo es requerido';
    }

    // Aplicar estilos de validación
    if (isValid) {
        field.classList.remove('is-invalid');
        field.classList.add('is-valid');
        removeErrorMessage(field);
    } else {
        field.classList.remove('is-valid');
        field.classList.add('is-invalid');
        showErrorMessage(field, errorMessage);
    }

    return isValid;
}

// Función para mostrar mensajes de error
function showErrorMessage(field, message) {
    removeErrorMessage(field);
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    
    field.parentNode.appendChild(errorDiv);
}

// Función para remover mensajes de error
function removeErrorMessage(field) {
    const existingError = field.parentNode.querySelector('.invalid-feedback');
    if (existingError) {
        existingError.remove();
    }
}

// Función para validar formulario completo
function validateForm(form) {
    const inputs = form.querySelectorAll('input, select, textarea');
    let isFormValid = true;

    inputs.forEach(input => {
        if (!validateField(input)) {
            isFormValid = false;
        }
    });

    return isFormValid;
}

// Función para mostrar notificaciones
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        <i class="fas fa-${getIconForType(type)} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Insertar al inicio del contenido principal
    const main = document.querySelector('main');
    if (main) {
        main.insertBefore(alertDiv, main.firstChild);
        
        // Auto-ocultar después de 5 segundos
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
}

// Función para obtener icono según tipo de notificación
function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Función para simular progreso de entrenamiento
function simulateTrainingProgress(progressElement, logElement, statusElement, maxEpochs) {
    let currentEpoch = 0;
    const interval = setInterval(() => {
        currentEpoch++;
        const progress = (currentEpoch / maxEpochs) * 100;
        
        // Actualizar barra de progreso
        if (progressElement) {
            progressElement.style.width = progress + '%';
            progressElement.textContent = Math.round(progress) + '%';
        }
        
        // Simular errores (decrecientes)
        const errors = Math.max(0, Math.floor(Math.random() * (maxEpochs - currentEpoch + 1)));
        
        // Actualizar log
        if (logElement) {
            const logEntry = document.createElement('div');
            logEntry.textContent = `Época ${currentEpoch}: ${errors} errores`;
            logElement.appendChild(logEntry);
            logElement.scrollTop = logElement.scrollHeight;
        }
        
        // Actualizar estado
        if (statusElement) {
            const epochSpan = statusElement.querySelector('#currentEpoch');
            const errorsSpan = statusElement.querySelector('#currentErrors');
            const weightsSpan = statusElement.querySelector('#currentWeights');
            const biasSpan = statusElement.querySelector('#currentBias');
            
            if (epochSpan) epochSpan.textContent = currentEpoch;
            if (errorsSpan) errorsSpan.textContent = errors;
            if (weightsSpan) weightsSpan.textContent = `[${(Math.random() * 2 - 1).toFixed(3)}, ${(Math.random() * 2 - 1).toFixed(3)}]`;
            if (biasSpan) biasSpan.textContent = (Math.random() * 2 - 1).toFixed(3);
        }
        
        // Verificar si terminó
        if (currentEpoch >= maxEpochs || errors === 0) {
            clearInterval(interval);
            
            // Mostrar resultado final
            if (logElement) {
                const finalLog = document.createElement('div');
                finalLog.className = 'text-success fw-bold';
                if (errors === 0) {
                    finalLog.textContent = `¡Convergencia alcanzada en la época ${currentEpoch}!`;
                } else {
                    finalLog.textContent = `Entrenamiento completado después de ${currentEpoch} épocas.`;
                }
                logElement.appendChild(finalLog);
            }
            
            return true; // Entrenamiento completado
        }
    }, 500); // Actualizar cada 500ms
    
    return interval; // Devolver ID del intervalo para poder cancelarlo
}

// Función para descargar archivos
function downloadFile(content, filename, contentType = 'text/plain') {
    const blob = new Blob([content], { type: contentType });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Función para formatear números
function formatNumber(num, decimals = 3) {
    return parseFloat(num).toFixed(decimals);
}

// Función para generar CSV
function generateCSV(data, headers) {
    let csv = headers.join(',') + '\n';
    data.forEach(row => {
        csv += row.join(',') + '\n';
    });
    return csv;
}

// Función para copiar al portapapeles
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copiado al portapapeles', 'success');
    }).catch(() => {
        showNotification('Error al copiar', 'danger');
    });
}

// Función para exportar datos como JSON
function exportAsJSON(data, filename) {
    const jsonString = JSON.stringify(data, null, 2);
    downloadFile(jsonString, filename, 'application/json');
}

// Función para validar archivos
function validateFile(file, allowedTypes, maxSize) {
    const fileType = file.type;
    const fileSize = file.size;
    
    if (!allowedTypes.includes(fileType)) {
        return { valid: false, message: 'Tipo de archivo no permitido' };
    }
    
    if (fileSize > maxSize) {
        return { valid: false, message: 'El archivo es demasiado grande' };
    }
    
    return { valid: true, message: 'Archivo válido' };
}

// Función para mostrar modal de confirmación
function showConfirmModal(title, message, callback) {
    const modalHtml = `
        <div class="modal fade" id="confirmModal" tabindex="-1">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">${title}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <p>${message}</p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancelar</button>
                        <button type="button" class="btn btn-primary" id="confirmBtn">Confirmar</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    const modal = new bootstrap.Modal(document.getElementById('confirmModal'));
    
    document.getElementById('confirmBtn').addEventListener('click', () => {
        modal.hide();
        if (callback) callback();
    });
    
    modal.show();
    
    // Limpiar modal después de cerrar
    document.getElementById('confirmModal').addEventListener('hidden.bs.modal', function() {
        this.remove();
    });
}

// Función para hacer peticiones AJAX
function makeAjaxRequest(url, method = 'GET', data = null) {
    return fetch(url, {
        method: method,
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        },
        body: data ? JSON.stringify(data) : null
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    });
}

// Función para obtener token CSRF
function getCSRFToken() {
    const token = document.querySelector('[name=csrfmiddlewaretoken]');
    return token ? token.value : '';
}

// Función para mostrar loading spinner
function showLoading(element) {
    element.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Cargando...';
    element.disabled = true;
}

// Función para ocultar loading spinner
function hideLoading(element, originalText) {
    element.innerHTML = originalText;
    element.disabled = false;
}

// Exportar funciones para uso global
window.PerceptronApp = {
    validateField,
    validateForm,
    showNotification,
    simulateTrainingProgress,
    downloadFile,
    formatNumber,
    generateCSV,
    copyToClipboard,
    exportAsJSON,
    validateFile,
    showConfirmModal,
    makeAjaxRequest,
    showLoading,
    hideLoading
};
