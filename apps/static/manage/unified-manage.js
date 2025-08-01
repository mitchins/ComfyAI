const BASE_PATH = "/manage-api";

let models = [];

document.addEventListener("DOMContentLoaded", () => {
    loadModels();
});

async function loadModels() {
    showLoading(true);
    try {
        const response = await fetch(`${BASE_PATH}/unified-models`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        models = await response.json();
        renderModels();
    } catch (error) {
        console.error('Error loading models:', error);
        showError(`Failed to load models: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

function showLoading(show) {
    const loading = document.getElementById('loading');
    loading.style.display = show ? 'block' : 'none';
}

function showError(message) {
    const container = document.getElementById('models-container');
    container.innerHTML = `
        <div class="card-panel red lighten-4">
            <span class="red-text text-darken-2">
                <i class="material-icons left">error</i>
                ${message}
            </span>
        </div>
    `;
}

function renderModels() {
    const container = document.getElementById('models-container');
    
    if (models.length === 0) {
        container.innerHTML = '<p>No models found.</p>';
        return;
    }

    // Group models by server
    const grouped = models.reduce((acc, model) => {
        if (!acc[model.server]) acc[model.server] = [];
        acc[model.server].push(model);
        return acc;
    }, {});

    let html = '';
    
    // Render each server group
    for (const [server, serverModels] of Object.entries(grouped)) {
        html += `<h5>${server.charAt(0).toUpperCase() + server.slice(1)} Server Models</h5>`;
        
        serverModels.forEach(model => {
            html += renderModel(model);
        });
    }
    
    container.innerHTML = html;
    
    // Initialize collapsible behavior
    initializeCollapsibles();
}

function renderModel(model) {
    const serverClass = `server-${model.server}`;
    
    // Handle special status formatting
    let statusText = model.status;
    let statusClass = `status-${model.status.replace('-', '_')}`;
    
    if (model.status.endsWith('-quants')) {
        const quantPart = model.status.split('-')[0];
        if (quantPart.includes('/')) {
            // Format: "2/4-quants" -> "2/4 SIZES"
            statusText = `${quantPart} SIZES`;
        } else {
            // Legacy format: "2-quants" -> "2 SIZES"
            statusText = `${quantPart} SIZES`;
        }
        statusClass = 'status-quants';
    } else {
        statusText = model.status.replace('-', ' ').toUpperCase();
    }
    
    let html = `
        <div class="model-section" data-model-id="${model.id}">
            <div class="model-header" onclick="toggleModel('${model.id}')">
                <div>
                    <span class="server-badge ${serverClass}">${model.server.toUpperCase()}</span>
                    <strong>${model.name}</strong>
                    <span class="status-badge ${statusClass}">${statusText}</span>
                    ${model.total_size_mb > 0 ? `<span class="grey-text">(${model.total_size_mb.toFixed(1)} MB)</span>` : ''}
                </div>
                <i class="material-icons expand-icon" id="icon-${model.id}">expand_more</i>
            </div>
            <div class="model-content" id="content-${model.id}">
                <p class="grey-text">${model.description}</p>
    `;
    
    if (model.architecture === 'multi-component') {
        html += renderMultiComponentModel(model);
    } else {
        html += renderSingleFileModel(model);
    }
    
    html += `
            </div>
        </div>
    `;
    
    return html;
}

function renderMultiComponentModel(model) {
    let html = '<div class="row">';
    
    // Show quantization options as simple choices
    if (model.quantizations.length > 1) {
        html += `
            <div class="col s12">
                <h6>Available Options:</h6>
        `;
        
        model.quantizations.forEach(quant => {
            const quantFiles = getFilesForQuantization(model, quant);
            const downloadedCount = quantFiles.filter(f => f.downloaded).length;
            const totalCount = quantFiles.length;
            const isFullyDownloaded = downloadedCount === totalCount;
            const isPartiallyDownloaded = downloadedCount > 0 && downloadedCount < totalCount;
            
            let statusText = 'Available';
            let statusClass = 'grey-text';
            let statusIcon = 'radio_button_unchecked';
            
            if (isFullyDownloaded) {
                statusText = 'Downloaded';
                statusClass = 'green-text';
                statusIcon = 'check_circle';
            } else if (isPartiallyDownloaded) {
                statusText = 'Partial';
                statusClass = 'orange-text';
                statusIcon = 'error';
            }
            
            html += `
                <div class="quantization-section">
                    <div class="row valign-wrapper">
                        <div class="col s1">
                            <i class="material-icons ${statusClass}">${statusIcon}</i>
                        </div>
                        <div class="col s5">
                            <strong>${quant}</strong>
                            <br><span class="${statusClass}">${statusText}</span>
                        </div>
                        <div class="col s6 right-align">
                            ${!isFullyDownloaded ? `
                                <button class="btn waves-effect waves-light blue" 
                                        onclick="downloadQuantization('${model.id}', '${quant}')">
                                    Download
                                </button>
                            ` : `
                                <button class="btn waves-effect waves-light red" 
                                        onclick="deleteQuantization('${model.id}', '${quant}')">
                                    Clear
                                </button>
                            `}
                        </div>
                    </div>
                    ${(isFullyDownloaded || isPartiallyDownloaded) ? `
                        <div class="file-list" style="margin-top: 10px; padding-left: 40px;">
                            ${quantFiles.map(file => `
                                <div class="file-item" style="font-size: 0.9em;">
                                    <span class="${file.downloaded ? 'file-downloaded' : 'file-not-downloaded'}">
                                        <i class="material-icons tiny">${file.downloaded ? 'check_circle' : 'radio_button_unchecked'}</i>
                                        ${file.path.split('/').pop()}
                                    </span>
                                    ${file.size_mb ? `<span class="grey-text">${file.size_mb.toFixed(1)} MB</span>` : ''}
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            `;
        });
        
        html += '</div>';
    } else {
        // Single quantization option
        const downloadedCount = model.files.filter(f => f.downloaded).length;
        const totalCount = model.files.length;
        const isFullyDownloaded = downloadedCount === totalCount;
        const isPartiallyDownloaded = downloadedCount > 0 && downloadedCount < totalCount;
        
        let statusText = 'Available';
        let statusClass = 'grey-text';
        let statusIcon = 'radio_button_unchecked';
        
        if (isFullyDownloaded) {
            statusText = 'Downloaded';
            statusClass = 'green-text';
            statusIcon = 'check_circle';
        } else if (isPartiallyDownloaded) {
            statusText = 'Partial';
            statusClass = 'orange-text';
            statusIcon = 'error';
        }
        
        html += `
            <div class="col s12">
                <div class="row valign-wrapper">
                    <div class="col s1">
                        <i class="material-icons ${statusClass}">${statusIcon}</i>
                    </div>
                    <div class="col s5">
                        <strong>Standard</strong>
                        <br><span class="${statusClass}">${statusText}</span>
                    </div>
                    <div class="col s6 right-align">
                        ${!isFullyDownloaded ? `
                            <button class="btn waves-effect waves-light blue" 
                                    onclick="downloadModel('${model.id}')">
                                Download
                            </button>
                        ` : `
                            <button class="btn waves-effect waves-light red" 
                                    onclick="deleteModel('${model.id}')">
                                Clear
                            </button>
                        `}
                    </div>
                </div>
                ${(isFullyDownloaded || isPartiallyDownloaded) ? `
                    <div class="file-list" style="margin-top: 10px; padding-left: 40px;">
                        ${model.files.map(file => `
                            <div class="file-item" style="font-size: 0.9em;">
                                <span class="${file.downloaded ? 'file-downloaded' : 'file-not-downloaded'}">
                                    <i class="material-icons tiny">${file.downloaded ? 'check_circle' : 'radio_button_unchecked'}</i>
                                    ${file.path.split('/').pop()}
                                </span>
                                ${file.size_mb ? `<span class="grey-text">${file.size_mb.toFixed(1)} MB</span>` : ''}
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    }
    
    html += '</div>';
    return html;
}

function renderSingleFileModel(model) {
    const file = model.files[0]; // Single file models have only one file
    
    let statusText = 'Available';
    let statusClass = 'grey-text';
    let statusIcon = 'radio_button_unchecked';
    
    if (file.downloaded) {
        statusText = 'Downloaded';
        statusClass = 'green-text';
        statusIcon = 'check_circle';
    }
    
    return `
        <div class="row valign-wrapper">
            <div class="col s1">
                <i class="material-icons ${statusClass}">${statusIcon}</i>
            </div>
            <div class="col s5">
                <strong>Standard</strong>
                <br><span class="${statusClass}">${statusText}</span>
            </div>
            <div class="col s6 right-align">
                ${!file.downloaded ? `
                    <button class="btn waves-effect waves-light blue" 
                            onclick="downloadModel('${model.id}')">
                        Download
                    </button>
                ` : `
                    <button class="btn waves-effect waves-light red" 
                            onclick="deleteModel('${model.id}')">
                        Clear
                    </button>
                `}
            </div>
        </div>
        ${file.downloaded ? `
            <div class="file-list" style="margin-top: 10px; padding-left: 40px;">
                <div class="file-item" style="font-size: 0.9em;">
                    <span class="file-downloaded">
                        <i class="material-icons tiny">check_circle</i>
                        ${file.path.split('/').pop()}
                    </span>
                    ${file.size_mb ? `<span class="grey-text">${file.size_mb.toFixed(1)} MB</span>` : ''}
                </div>
            </div>
        ` : ''}
    `;
}

function getFilesForQuantization(model, quantization) {
    // For multi-component models, we need to map quantization to specific files
    if (model.architecture === 'single-file') {
        return model.files;
    }
    
    // For multi-component models, try to filter files by quantization suffix
    const quantizationSuffixes = {
        'Q4': ['_q4.onnx'],
        'Q4_MIXED': ['_q4.onnx', '_quantized.onnx'],
        'Q4_F16': ['_q4f16.onnx', '_q4_f16.onnx'],
        'FP16': ['_fp16.onnx'],
        'INT8': ['_int8.onnx'],
        'BNB4': ['_bnb4.onnx'],
        'UINT8': ['_uint8.onnx'],
        'QUANTIZED': ['_quantized.onnx']
    };
    
    const suffixes = quantizationSuffixes[quantization] || [];
    
    // Special handling for FP32 - match files WITHOUT other quantization suffixes
    if (quantization === 'FP32') {
        const otherQuantSuffixes = [
            '_fp16.onnx', '_int8.onnx', '_uint8.onnx', '_q4.onnx', 
            '_q4f16.onnx', '_q4_f16.onnx', '_bnb4.onnx', '_quantized.onnx'
        ];
        
        const fp32Files = model.files.filter(file => {
            const fileLower = file.path.toLowerCase();
            // Include if it's a base file without quantization suffixes
            // Base files are like "embed_tokens.onnx", "decoder_model_merged.onnx", etc.
            return !otherQuantSuffixes.some(suffix => fileLower.includes(suffix.toLowerCase()));
        });
        
        return fp32Files.length > 0 ? fp32Files : [];
    }
    
    if (suffixes.length === 0) {
        // If no specific mapping, return all files
        return model.files;
    }
    
    // Filter files that match any of the suffixes
    const filteredFiles = model.files.filter(file => {
        const fileLower = file.path.toLowerCase();
        return suffixes.some(suffix => fileLower.includes(suffix.toLowerCase()));
    });
    
    // Only return matched files - no fallback to all files for specific quantizations
    return filteredFiles;
}

function toggleModel(modelId) {
    const content = document.getElementById(`content-${modelId}`);
    const icon = document.getElementById(`icon-${modelId}`);
    
    if (content.classList.contains('expanded')) {
        content.classList.remove('expanded');
        icon.classList.remove('rotated');
    } else {
        content.classList.add('expanded');
        icon.classList.add('rotated');
    }
}

function initializeCollapsibles() {
    // Models start collapsed - no additional initialization needed
    // since we're handling clicks manually
}

async function downloadModel(modelId) {
    try {
        const response = await fetch(`${BASE_PATH}/unified-models/download`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_id: modelId
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        M.toast({html: 'Download started!', classes: 'green'});
        
        // Refresh models after a short delay
        setTimeout(refreshModels, 2000);
        
    } catch (error) {
        console.error('Download error:', error);
        M.toast({html: `Download failed: ${error.message}`, classes: 'red'});
    }
}

async function downloadQuantization(modelId, quantization) {
    try {
        const response = await fetch(`${BASE_PATH}/unified-models/download`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_id: modelId,
                quantization: quantization
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        M.toast({html: `Download started for ${quantization}!`, classes: 'green'});
        
        // Refresh models after a short delay
        setTimeout(refreshModels, 2000);
        
    } catch (error) {
        console.error('Download error:', error);
        M.toast({html: `Download failed: ${error.message}`, classes: 'red'});
    }
}

async function deleteModel(modelId) {
    if (!confirm('Are you sure you want to clear this model?')) {
        return;
    }
    
    try {
        const response = await fetch(`${BASE_PATH}/unified-models/${modelId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        M.toast({html: 'Model cleared successfully!', classes: 'green'});
        refreshModels();
        
    } catch (error) {
        console.error('Delete error:', error);
        M.toast({html: `Delete failed: ${error.message}`, classes: 'red'});
    }
}

async function deleteQuantization(modelId, quantization) {
    if (!confirm(`Are you sure you want to clear the ${quantization} option?`)) {
        return;
    }
    
    try {
        const response = await fetch(`${BASE_PATH}/unified-models/${modelId}?quantization=${quantization}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        M.toast({html: `${quantization} cleared successfully!`, classes: 'green'});
        refreshModels();
        
    } catch (error) {
        console.error('Delete error:', error);
        M.toast({html: `Delete failed: ${error.message}`, classes: 'red'});
    }
}

function refreshModels() {
    loadModels();
}