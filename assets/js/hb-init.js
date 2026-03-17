const STORAGE_KEY = "wc-color-theme";
const VALID_THEMES = new Set(["light", "dark", "system"]);

function normalizeTheme(value, fallback = "system") {
  return VALID_THEMES.has(value) ? value : fallback;
}

function getRoot() {
  return document.documentElement;
}

function getSystemMediaQuery() {
  return window.matchMedia("(prefers-color-scheme: dark)");
}

function resolveTheme(preference) {
  return preference === "system"
    ? (getSystemMediaQuery().matches ? "dark" : "light")
    : preference;
}

function applyResolvedTheme(resolvedTheme) {
  const root = getRoot();

  if (resolvedTheme === "dark") {
    root.classList.add("dark");
    root.style.colorScheme = "dark";
  } else {
    root.classList.remove("dark");
    root.style.colorScheme = "light";
  }
}

function getDefaultTheme() {
  const root = getRoot();
  return normalizeTheme(root.dataset.wcThemeDefault || "system");
}

function getStoredThemePreference() {
  const stored = localStorage.getItem(STORAGE_KEY);
  return stored && VALID_THEMES.has(stored) ? stored : null;
}

function getThemePreference() {
  return getStoredThemePreference() || getDefaultTheme();
}

function getResolvedTheme() {
  return resolveTheme(getThemePreference());
}

function dispatchThemeChange() {
  const preference = getThemePreference();
  const resolvedTheme = resolveTheme(preference);

  document.dispatchEvent(new CustomEvent("hbThemeChange", {
    detail: {
      preference: preference,
      resolvedTheme: resolvedTheme,
      isDarkTheme: () => resolvedTheme === "dark"
    }
  }));
}

function syncThemeControls() {
  if (window.hbb && typeof window.hbb.syncThemeControls === "function") {
    window.hbb.syncThemeControls();
  }
}

function applyThemePreference(preference, shouldDispatch = true) {
  const nextPreference = normalizeTheme(preference);
  const resolvedTheme = resolveTheme(nextPreference);

  applyResolvedTheme(resolvedTheme);

  if (shouldDispatch) {
    syncThemeControls();
    dispatchThemeChange();
  }
}

function setThemePreference(preference) {
  const nextPreference = normalizeTheme(preference);
  localStorage.setItem(STORAGE_KEY, nextPreference);
  applyThemePreference(nextPreference);
}

function clearThemePreference() {
  localStorage.removeItem(STORAGE_KEY);
  applyThemePreference(getDefaultTheme());
}

let systemListenerBound = false;

export function initTheme() {
  applyThemePreference(getThemePreference(), false);

  if (!systemListenerBound) {
    getSystemMediaQuery().addEventListener("change", () => {
      if (getThemePreference() === "system") {
        applyThemePreference("system");
      }
    });
    systemListenerBound = true;
  }
}

export function applyHugoStyleFixes() {
  document.addEventListener("DOMContentLoaded", () => {
    const checkboxes = document.querySelectorAll("li input[type='checkbox'][disabled]");
    checkboxes.forEach((element) => {
      const parent = element.parentElement?.parentElement;
      if (parent) parent.classList.add("task-list");
    });

    const liNodes = document.querySelectorAll(".task-list li");
    liNodes.forEach((nodes) => {
      const textNodes = Array.from(nodes.childNodes).filter((node) => node.nodeType === 3 && node.textContent && node.textContent.trim().length > 1);
      if (textNodes.length > 0) {
        const span = document.createElement("label");
        textNodes[0].after(span);
        const input = nodes.querySelector("input[type='checkbox']");
        if (input) span.appendChild(input);
        span.appendChild(textNodes[0]);
      }
    });
  });
}

export {
  STORAGE_KEY,
  clearThemePreference,
  getDefaultTheme,
  getResolvedTheme,
  getThemePreference,
  normalizeTheme,
  setThemePreference
};
