// Open calculated notebook figures at their original resolution.
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".nboutput img").forEach((image) => {
    if (image.closest("a")) return;
    const link = document.createElement("a");
    link.href = image.src;
    link.target = "_blank";
    link.rel = "noopener";
    link.title = "View figure at full size";
    link.setAttribute("aria-label", "View figure at full size");
    image.style.cursor = "zoom-in";
    image.replaceWith(link);
    link.appendChild(image);
  });
});
