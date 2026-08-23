const { app, BrowserWindow, Menu, Tray, nativeImage } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const http = require("http");
const os = require("os");
const path = require("path");

const orbitURL = "http://127.0.0.1:8765";
let mainWindow;
let tray;
let isQuitting = false;

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => showOrbit());
  app.whenReady().then(startDesktop);
}

function orbitExecutable() {
  return path.join(os.homedir(), ".orbit", "runtime", "bin", "orbit");
}

function autostartPath() {
  return path.join(os.homedir(), ".config", "autostart", "top.orbit.desktop.desktop");
}

function desktopExecEscape(value) {
  return `"${value.replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"`;
}

function registerAutostart() {
  const executable = process.env.APPIMAGE || process.execPath;
  const target = autostartPath();
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, `[Desktop Entry]\nType=Application\nName=Orbit\nComment=Orbit local AI\nExec=${desktopExecEscape(executable)}\nTerminal=false\nX-GNOME-Autostart-enabled=true\n`, "utf8");
}

function unregisterAutostart() {
  try { fs.unlinkSync(autostartPath()); } catch (error) { if (error.code !== "ENOENT") console.error(error); }
}

function run(file, args, onLine) {
  return new Promise((resolve, reject) => {
    const child = spawn(file, args, { env: { ...process.env, ORBIT_NO_BROWSER: "1" }, stdio: ["ignore", "pipe", "pipe"] });
    for (const stream of [child.stdout, child.stderr]) {
      stream.on("data", data => {
        const lines = data.toString().trim().split("\n");
        if (lines.length && lines.at(-1)) onLine?.(lines.at(-1));
      });
    }
    child.on("error", reject);
    child.on("exit", code => code === 0 ? resolve() : reject(new Error(`Orbit command failed (${code})`)));
  });
}

function healthy() {
  return new Promise(resolve => {
    const request = http.get(`${orbitURL}/api/health`, { timeout: 1000 }, response => {
      response.resume();
      resolve(response.statusCode === 200);
    });
    request.on("timeout", () => { request.destroy(); resolve(false); });
    request.on("error", () => resolve(false));
  });
}

function setLoadingStatus(text) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  mainWindow.webContents.executeJavaScript(`document.getElementById("status").textContent = ${JSON.stringify(text)}`).catch(() => {});
}

async function prepareOrbit() {
  if (!await healthy()) {
    const orbit = orbitExecutable();
    if (fs.existsSync(orbit)) {
      setLoadingStatus("Starting the local OpenAI-compatible API…");
      await run(orbit, ["start"], setLoadingStatus);
    } else {
      setLoadingStatus("First launch: installing the local AI runtime…");
      await run("/bin/sh", [path.join(process.resourcesPath, "install.sh")], setLoadingStatus);
    }
    for (let i = 0; i < 80 && !await healthy(); i++) await new Promise(resolve => setTimeout(resolve, 250));
  }
  if (!await healthy()) throw new Error("The local API did not start.");
  await mainWindow.loadURL(orbitURL);
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 980,
    minHeight: 640,
    title: "Orbit",
    icon: path.join(process.resourcesPath, "orbit-logo.png"),
    backgroundColor: "#f5f5f7",
    webPreferences: { contextIsolation: true, nodeIntegration: false },
  });
  mainWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;display:grid;place-items:center;height:100vh;margin:0;background:#f5f5f7;color:#5b5b61}</style><div id="status">Preparing Orbit…</div>')}`);
  mainWindow.on("close", event => {
    if (!isQuitting) {
      event.preventDefault();
      mainWindow.hide();
      mainWindow.setSkipTaskbar(true);
    }
  });
  mainWindow.on("minimize", event => {
    event.preventDefault();
    mainWindow.hide();
    mainWindow.setSkipTaskbar(true);
  });
}

function showOrbit() {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  mainWindow.setSkipTaskbar(false);
  mainWindow.show();
  mainWindow.restore();
  mainWindow.focus();
}

async function runOrbitQuietly(args) {
  const orbit = orbitExecutable();
  if (!fs.existsSync(orbit)) return;
  try { await run(orbit, args); } catch (error) { console.error(error); }
}

function quitApp() {
  unregisterAutostart();
  isQuitting = true;
  tray.destroy();
  app.quit();
}

async function stopAgent() {
  await runOrbitQuietly(["service", "uninstall"]);
  quitApp();
}

function createTray() {
  let icon = nativeImage.createFromPath(path.join(process.resourcesPath, "orbit-logo-transparent.png"));
  icon = icon.resize({ width: 20, height: 20 });
  tray = new Tray(icon);
  tray.setToolTip("Orbit · Local AI");
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: "Open Orbit", click: showOrbit },
    { label: "Local API · 127.0.0.1:8765", enabled: false },
    { label: "Unload Model", click: () => runOrbitQuietly(["unload"]) },
    { type: "separator" },
    { label: "Exit App (Agent Keeps Running)", click: quitApp },
    { label: "Stop Background Agent", click: stopAgent },
  ]));
  tray.on("double-click", showOrbit);
}

async function startDesktop() {
  app.on("before-quit", () => { isQuitting = true; });
  app.on("window-all-closed", () => {});
  registerAutostart();
  createWindow();
  createTray();
  try { await prepareOrbit(); }
  catch (error) { setLoadingStatus(`Orbit could not start: ${error.message}`); }
}
