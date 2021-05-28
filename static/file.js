# Dynamicly change upload button text.
var loader = function(e) {
  let file = e.target.files;
  let show = "<span>Selected File: </span>" + file[0].name
  let output = document.getElementById("selector")
  output.innerHTML = show;
  output.classList.add("active")
};

let fileInput = document.getElementById("file")
fileInput.addEventListener("change", loader)