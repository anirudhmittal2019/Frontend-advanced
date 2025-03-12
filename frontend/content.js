const processedImages = new Map();
let eagleWatcher = null;
let currentState = 'idle';

const TARGETED_SHORT_URL = 'https://www.youtube.com/shorts/WdoY_EXiT80';
let videoCheckInterval = null;

function createEagleWatcher() {
    eagleWatcher = document.createElement('div');
    eagleWatcher.className = 'eagle-watcher';
    updateEagleState('jump');
    document.body.appendChild(eagleWatcher);
}

function updateEagleState(state) {
    if (!eagleWatcher) return;
    
    const stateGifs = {
        'confuse': 'confuse.gif',
        'jump': 'jump.gif',
        'sit': 'sit.gif',
        'shiny': 'shiny.gif'
    };
    
    const gifUrl = chrome.runtime.getURL(stateGifs[state]);
    eagleWatcher.innerHTML = `
        <img src="${gifUrl}" alt="Turk Eye Watcher Icon" style="width: 100%; height: 100%;">
    `;
    currentState = state;
}

async function processImage(img) {
    if (processedImages.has(img.src)) {
        return;
    }

    const imageId = btoa(img.src).substring(0, 12);
    processedImages.set(img.src, imageId);

    blockImage(img, imageId, true);
    updateEagleState('confuse');

    try {
        const response = await fetch('http://localhost:5000/check-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                imageUrl: img.src,
                imageId: imageId
            })
        });

        const data = await response.json();

        if (data.decision === 0) {
            unblockImage(img, imageId);
        } else {
            updateOverlayMessage(img, imageId, false);
        }

        updateEagleState('jump');
        
        setTimeout(() => {
            const hasBlockedImages = Array.from(document.getElementsByClassName('blocked-image-container')).length > 0;
            updateEagleState(hasBlockedImages ? 'jump' : 'sit');
        }, 1000);

    } catch (error) {
        console.error('Error:', error);
        updateOverlayMessage(img, imageId, false);
        updateEagleState('jump');
    }
}

function blockImage(img, imageId, isUnderReview = true) {
    const existingContainer = img.closest('.blocked-image-container');
    if (existingContainer) return;

    const container = document.createElement('div');
    container.className = 'blocked-image-container';
    container.dataset.imageId = imageId;
    img.parentNode.insertBefore(container, img);
    container.appendChild(img);

    const overlay = document.createElement('div');
    overlay.className = 'image-overlay';
    overlay.innerHTML = `
        <p>${isUnderReview ? 'Content Under Review' : 'Fake Content Detected'}</p>
        <button class="show-anyway-btn" data-image-id="${imageId}">
            I still want to see
        </button>
    `;
    container.appendChild(overlay);

    overlay.querySelector('.show-anyway-btn').addEventListener('click', function() {
        unblockImage(img, imageId);
    });
}

function updateOverlayMessage(img, imageId, isUnderReview) {
    const container = img.closest('.blocked-image-container');
    if (!container || container.dataset.imageId !== imageId) return;

    const messageElement = container.querySelector('.image-overlay p');
    if (messageElement) {
        messageElement.textContent = isUnderReview ? 'Content Under Review' : 'Fake Content Detected';
    }
}

function unblockImage(img, imageId) {
    const container = img.closest('.blocked-image-container');
    if (!container) return;

    if (container.dataset.imageId === imageId) {
        const parent = container.parentNode;
        parent.insertBefore(img, container);
        container.remove();
    }

    const hasBlockedImages = Array.from(document.getElementsByClassName('blocked-image-container')).length > 0;
    updateEagleState(hasBlockedImages ? 'jump' : 'sit');
}

const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
            if (node.nodeName === 'IMG') {
                processImage(node);
            } else if (node.getElementsByTagName) {
                const images = node.getElementsByTagName('img');
                Array.from(images).forEach(processImage);
            }
        });
    });
});

function init() {
    createEagleWatcher();

    const images = document.getElementsByTagName('img');
    Array.from(images).forEach(processImage);

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    checkForTargetedVideo();
}

function checkForTargetedVideo() {
    if (videoCheckInterval) {
        clearInterval(videoCheckInterval);
    }

    videoCheckInterval = setInterval(() => {
        if (window.location.href === TARGETED_SHORT_URL) {
            const videoElement = document.querySelector('video');
            if (videoElement && !videoElement.paused) {
                updateEagleState('confuse');
                
                setTimeout(() => {
                    updateEagleState('shiny');
                    videoElement.pause();
                }, 5000);
            }
        }
    }, 1000);
}

init();