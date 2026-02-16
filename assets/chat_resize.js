// Drag-to-resize for the chat window's left edge
document.addEventListener("DOMContentLoaded", function () {
    var handle = null;
    var chatWindow = null;
    var startX = 0;
    var startWidth = 0;

    function initResize(e) {
        chatWindow = document.getElementById("chat-window");
        if (!chatWindow) return;
        handle = document.getElementById("chat-resize-handle");
        startX = e.clientX;
        startWidth = chatWindow.getBoundingClientRect().width;
        if (handle) handle.classList.add("active");
        document.addEventListener("mousemove", doResize);
        document.addEventListener("mouseup", stopResize);
        e.preventDefault();
    }

    function doResize(e) {
        if (!chatWindow) return;
        // Dragging left increases width (panel anchored to right)
        var newWidth = startWidth + (startX - e.clientX);
        var minW = 320;
        var maxW = window.innerWidth * 0.8;
        newWidth = Math.max(minW, Math.min(maxW, newWidth));
        chatWindow.style.width = newWidth + "px";
        chatWindow.style.maxWidth = "none";
    }

    function stopResize() {
        if (handle) handle.classList.remove("active");
        document.removeEventListener("mousemove", doResize);
        document.removeEventListener("mouseup", stopResize);
        chatWindow = null;
    }

    // Use MutationObserver to attach listener whenever the handle appears
    var observer = new MutationObserver(function () {
        var h = document.getElementById("chat-resize-handle");
        if (h && !h._resizeInit) {
            h.addEventListener("mousedown", initResize);
            h._resizeInit = true;
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });

    // Also try binding immediately in case element already exists
    var h = document.getElementById("chat-resize-handle");
    if (h) {
        h.addEventListener("mousedown", initResize);
        h._resizeInit = true;
    }
});
