(() => {
  const navToggle = document.querySelector('.navbar-toggler');
  const navMenu = document.getElementById('mainNav');
  if (navToggle && navMenu) {
    navToggle.addEventListener('click', () => {
      const expanded = navToggle.getAttribute('aria-expanded') === 'true';
      navToggle.setAttribute('aria-expanded', String(!expanded));
      navMenu.classList.toggle('show', !expanded);
    });
  }
  const input = document.getElementById("imageInput");
  const previewWrap = document.getElementById("previewWrap");
  const preview = document.getElementById("imagePreview");
  const fileName = document.getElementById("fileName");
  const dropZone = document.getElementById("dropZone");
  const removeButton = document.getElementById("removeImage");
  let previewUrl;

  function showFile(file) {
    if (!file) return;
    const allowed = ["image/jpeg", "image/png"];
    if (!allowed.includes(file.type)) {
      window.alert("Choose a JPG, JPEG, or PNG image.");
      input.value = "";
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      window.alert("Choose an image smaller than 8 MB.");
      input.value = "";
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    fileName.textContent = file.name;
    previewWrap.hidden = false;
    dropZone.classList.add("has-file");
  }

  if (input) {
    input.addEventListener("change", () => showFile(input.files[0]));
    removeButton.addEventListener("click", () => {
      input.value = "";
      previewWrap.hidden = true;
      dropZone.classList.remove("has-file");
      preview.removeAttribute("src");
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    });
    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("dragging");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("dragging");
      });
    });
    dropZone.addEventListener("drop", (event) => {
      const [file] = event.dataTransfer.files;
      if (!file) return;
      const transfer = new DataTransfer();
      transfer.items.add(file);
      input.files = transfer.files;
      showFile(file);
    });
    document.getElementById("uploadForm").addEventListener("submit", (event) => {
      const file = input.files[0];
      if (!file) {
        event.preventDefault();
        window.alert("Choose an image before submitting.");
        return;
      }
      if (file.size > 8 * 1024 * 1024) {
        event.preventDefault();
        window.alert("Choose an image smaller than 8 MB.");
        return;
      }
      document.getElementById("submitButton").disabled = true;
      document.querySelector(".button-label").hidden = true;
      document.querySelector(".button-loading").hidden = false;
    });
  }
})();
