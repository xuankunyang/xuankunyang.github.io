document.addEventListener("DOMContentLoaded", () => {
  addThemeToggleListener();
});

function getThemeMeta(preference, resolvedTheme) {
  const meta = {
    system: {
      label: "System"
    },
    light: {
      label: "Light"
    },
    dark: {
      label: "Dark"
    }
  };

  const current = meta[preference] || meta.system;
  return {
    label: current.label,
    resolvedLabel: resolvedTheme === "dark" ? "Dark" : "Light"
  };
}

function createSystemIcon() {
  const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  icon.setAttribute("data-theme-role", "system-icon");
  icon.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  icon.setAttribute("width", "18");
  icon.setAttribute("height", "18");
  icon.setAttribute("viewBox", "0 0 24 24");
  icon.setAttribute("fill", "none");
  icon.setAttribute("stroke", "currentColor");
  icon.setAttribute("stroke-width", "1.9");
  icon.setAttribute("stroke-linecap", "round");
  icon.setAttribute("stroke-linejoin", "round");
  icon.setAttribute("aria-hidden", "true");
  icon.innerHTML = '<rect x="3" y="4" width="18" height="13" rx="2"></rect><path d="M8 20h8"></path><path d="M12 17v3"></path>';
  return icon;
}

function ensureThemeIconMarkup(button) {
  const existingSystemIcon = button.querySelector("[data-theme-role='system-icon']");
  const svgIcons = button.querySelectorAll("svg");

  if (!existingSystemIcon && svgIcons.length >= 2) {
    svgIcons[0].setAttribute("data-theme-role", "dark-icon");
    svgIcons[1].setAttribute("data-theme-role", "light-icon");
    button.prepend(createSystemIcon());
  }
}

function setVisibleIcon(button, preference) {
  const iconMap = {
    system: "system-icon",
    light: "light-icon",
    dark: "dark-icon"
  };

  Object.values(iconMap).forEach((role) => {
    const icon = button.querySelector("[data-theme-role='" + role + "']");
    if (!icon) return;
    const isActive = role === iconMap[preference];
    icon.hidden = !isActive;
    icon.style.display = isActive ? "block" : "none";
  });
}

function updateThemeToggleButton(button) {
  ensureThemeIconMarkup(button);

  const preference = window.hbb.getThemePreference();
  const resolvedTheme = window.hbb.getResolvedTheme();
  const meta = getThemeMeta(preference, resolvedTheme);

  button.dataset.themePreference = preference;
  button.dataset.theme = resolvedTheme;
  button.dataset.themeLabel = meta.label;
  button.title = "Theme: " + meta.label;
  button.setAttribute("aria-label", "Theme mode: " + meta.label + ". Click to switch mode.");

  const modeResolved = button.querySelector("[data-theme-role='resolved']");
  if (modeResolved) {
    modeResolved.textContent = "Resolved: " + meta.resolvedLabel;
  }

  setVisibleIcon(button, preference);
}

function syncAllThemeToggleButtons() {
  const themeToggleButtons = document.querySelectorAll(".theme-toggle");
  themeToggleButtons.forEach((button) => {
    updateThemeToggleButton(button);
  });
}

function addThemeToggleListener() {
  const themeToggleButtons = document.querySelectorAll(".theme-toggle");

  window.hbb.syncThemeControls = syncAllThemeToggleButtons;
  syncAllThemeToggleButtons();

  themeToggleButtons.forEach((button) => {
    if (button.dataset.themeBound === "true") return;

    button.dataset.themeBound = "true";
    button.addEventListener("click", () => {
      const nextPreference = window.hbb.getNextThemePreference(window.hbb.getThemePreference());
      window.hbb.setThemePreference(nextPreference);
    });
  });

  document.addEventListener("hbThemeChange", syncAllThemeToggleButtons);
}
