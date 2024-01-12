document.addEventListener('DOMContentLoaded', function () {
    var fileInput = document.getElementById('fileInput');
    var fileNameDisplay = document.getElementById('fileNameDisplay');
    var fileNameInput = document.getElementById('fileNameInput');
    var form = document.getElementById('form');

    // Restore the file name if it was previously selected
    if (fileNameInput.value !== '') {
      fileNameDisplay.textContent = fileNameInput.value;
    }

    fileInput.addEventListener('change', function () {
      updateFileName();
    });

    form.addEventListener('submit', function (event) {
      // Check if a new file has been selected
      if (fileInput.files.length > 0) {
        // Update the file name only if a new file has been selected
        updateFileName();
      } else {
        // If no new file is selected, restore the file name from the hidden input
        fileNameDisplay.textContent = fileNameInput.value || 'No file chosen';
      }
    });

    function updateFileName() {
      var fileName = fileInput.files[0] ? fileInput.files[0].name : 'No file chosen';
      fileNameDisplay.textContent = fileName;
      fileNameInput.value = fileName;
    }
  });