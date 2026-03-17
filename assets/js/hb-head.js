import {
  applyHugoStyleFixes,
  clearThemePreference,
  getDefaultTheme,
  getResolvedTheme,
  getThemePreference,
  initTheme,
  normalizeTheme,
  setThemePreference
} from "./hb-init.js";

const THEME_ORDER = ["system", "light", "dark"];

function getNextThemePreference(currentPreference) {
  const currentIndex = THEME_ORDER.indexOf(normalizeTheme(currentPreference));
  return THEME_ORDER[(currentIndex + 1) % THEME_ORDER.length];
}

window.hbb = {
  defaultTheme: getDefaultTheme(),
  getThemePreference: getThemePreference,
  getResolvedTheme: getResolvedTheme,
  setThemePreference: setThemePreference,
  clearThemePreference: clearThemePreference,
  getNextThemePreference: getNextThemePreference,
  setDarkTheme: () => setThemePreference("dark"),
  setLightTheme: () => setThemePreference("light"),
  setSystemTheme: () => setThemePreference("system"),
  syncThemeControls: () => {}
};

initTheme();
applyHugoStyleFixes();
