// Enhanced drag and drop functionality
document.addEventListener('DOMContentLoaded', () => {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        const label = input.nextElementSibling;
        
        if (!label) return;
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            label.addEventListener(eventName, preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            label.addEventListener(eventName, () => {
                const div = label.querySelector('div');
                if (div) div.classList.add('border-indigo-500', 'bg-indigo-50');
            });
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            label.addEventListener(eventName, () => {
                const div = label.querySelector('div');
                if (div) div.classList.remove('border-indigo-500', 'bg-indigo-50');
            });
        });
        
        label.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length) {
                input.files = files;
                updateFileName(input.id);
            }
        });
    });
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}
