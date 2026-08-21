import AppKit
import WebKit

private let orbitURL = URL(string: "http://127.0.0.1:8765")!
private let desktopAgentLabel = "top.orbit.desktop"

final class OrbitApp: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var statusLabel: NSTextField!
    private var spinner: NSProgressIndicator!
    private var statusItem: NSStatusItem!
    private var terminationInProgress = false
    private var systemIsShuttingDown = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        installLoginAgent()
        observeShutdown()
        buildApplicationMenu()
        buildStatusItem()
        buildWindow()
        NSApp.activate(ignoringOtherApps: true)
        Task { await prepareOrbit() }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        showWindow()
        return true
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if terminationInProgress { return .terminateNow }
        terminationInProgress = true
        if !systemIsShuttingDown {
            removeLoginAgent()
            uninstallOrbitServiceBeforeExit()
        }
        return .terminateNow
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        sender.orderOut(nil)
        NSApp.setActivationPolicy(.accessory)
        return false
    }

    func windowDidMiniaturize(_ notification: Notification) {
        guard let sender = notification.object as? NSWindow else { return }
        sender.deminiaturize(nil)
        sender.orderOut(nil)
        NSApp.setActivationPolicy(.accessory)
    }

    private func observeShutdown() {
        NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.willPowerOffNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in self?.systemIsShuttingDown = true }
    }

    private func buildApplicationMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem()
        main.addItem(appItem)
        let appMenu = NSMenu(title: "Orbit")
        appMenu.addItem(withTitle: "About Orbit", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit Orbit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu

        let editItem = NSMenuItem()
        main.addItem(editItem)
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        editMenu.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        editMenu.addItem(.separator())
        editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = editMenu
        NSApp.mainMenu = main
    }

    private func buildStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = statusItem.button {
            if let path = Bundle.main.path(forResource: "orbit-logo-transparent", ofType: "png"),
               let image = NSImage(contentsOfFile: path) {
                image.size = NSSize(width: 18, height: 18)
                image.isTemplate = true
                button.image = image
            } else {
                button.image = NSImage(systemSymbolName: "circle.dotted", accessibilityDescription: "Orbit")
            }
            button.toolTip = "Orbit"
        }

        let menu = NSMenu()
        let open = NSMenuItem(title: "Open Orbit", action: #selector(openOrbit), keyEquivalent: "o")
        open.target = self
        menu.addItem(open)
        let api = NSMenuItem(title: "Local API · 127.0.0.1:8765", action: nil, keyEquivalent: "")
        api.isEnabled = false
        menu.addItem(api)
        let unload = NSMenuItem(title: "Unload Model", action: #selector(unloadModel), keyEquivalent: "u")
        unload.target = self
        menu.addItem(unload)
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Quit Orbit", action: #selector(quitOrbit), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)
        statusItem.menu = menu
    }

    private func buildWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1180, height: 780), styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.delegate = self
        window.title = "Orbit"
        window.titlebarAppearsTransparent = true
        window.isMovableByWindowBackground = false
        window.center()
        window.minSize = NSSize(width: 820, height: 600)

        let container = NSView(frame: window.contentView!.bounds)
        container.autoresizingMask = [.width, .height]
        window.contentView = container
        webView = WKWebView(frame: container.bounds)
        webView.autoresizingMask = [.width, .height]
        webView.navigationDelegate = self
        webView.isHidden = true
        container.addSubview(webView)

        spinner = NSProgressIndicator(frame: NSRect(x: 0, y: 0, width: 34, height: 34))
        spinner.style = .spinning
        spinner.startAnimation(nil)
        statusLabel = NSTextField(labelWithString: "正在准备 Orbit…")
        statusLabel.font = .systemFont(ofSize: 16, weight: .medium)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.alignment = .center
        spinner.translatesAutoresizingMaskIntoConstraints = false
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(spinner)
        container.addSubview(statusLabel)
        NSLayoutConstraint.activate([
            spinner.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            spinner.centerYAnchor.constraint(equalTo: container.centerYAnchor, constant: -22),
            statusLabel.topAnchor.constraint(equalTo: spinner.bottomAnchor, constant: 14),
            statusLabel.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            statusLabel.leadingAnchor.constraint(greaterThanOrEqualTo: container.leadingAnchor, constant: 30),
        ])
        window.makeKeyAndOrderFront(nil)
    }

    @objc private func openOrbit() { showWindow() }

    private func showWindow() {
        NSApp.setActivationPolicy(.regular)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func unloadModel() {
        Task { try? await runOrbit(["unload"]) }
    }

    @objc private func quitOrbit() { NSApp.terminate(nil) }

    @MainActor private func setStatus(_ text: String) { statusLabel.stringValue = text }

    private func isHealthy() async -> Bool {
        var request = URLRequest(url: orbitURL.appending(path: "api/health"))
        request.timeoutInterval = 1
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode == 200 && String(data: data, encoding: .utf8)?.contains("\"orbit\"") == true
        } catch { return false }
    }

    private func run(_ executable: String, _ arguments: [String], onLine: ((String) -> Void)? = nil) async throws {
        try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            let pipe = Pipe()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = arguments
            process.standardOutput = pipe
            process.standardError = pipe
            pipe.fileHandleForReading.readabilityHandler = { handle in
                let text = String(data: handle.availableData, encoding: .utf8) ?? ""
                if let line = text.split(separator: "\n").last { onLine?(String(line)) }
            }
            process.terminationHandler = { task in
                pipe.fileHandleForReading.readabilityHandler = nil
                if task.terminationStatus == 0 { continuation.resume() }
                else { continuation.resume(throwing: NSError(domain: "Orbit", code: Int(task.terminationStatus), userInfo: [NSLocalizedDescriptionKey: "Orbit command failed (\(task.terminationStatus))"])) }
            }
            do { try process.run() } catch { continuation.resume(throwing: error) }
        }
    }

    private func orbitExecutable() -> String {
        FileManager.default.homeDirectoryForCurrentUser.appending(path: ".orbit/runtime/bin/orbit").path
    }

    private func runOrbit(_ arguments: [String]) async throws {
        let executable = orbitExecutable()
        guard FileManager.default.isExecutableFile(atPath: executable) else { return }
        try await run(executable, arguments)
    }

    private func uninstallOrbitServiceBeforeExit() {
        let executable = orbitExecutable()
        guard FileManager.default.isExecutableFile(atPath: executable) else { return }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = ["service", "uninstall"]
        do { try process.run() } catch { return }
        let deadline = Date().addingTimeInterval(4)
        while process.isRunning && Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.05))
        }
        if process.isRunning { process.terminate() }
    }

    private func loginAgentPath() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "Library/LaunchAgents/\(desktopAgentLabel).plist")
    }

    private func installLoginAgent() {
        guard let executable = Bundle.main.executablePath else { return }
        let path = loginAgentPath()
        let payload: [String: Any] = [
            "Label": desktopAgentLabel,
            "ProgramArguments": [executable],
            "RunAtLoad": true,
            "ProcessType": "Interactive",
        ]
        do {
            try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = try PropertyListSerialization.data(fromPropertyList: payload, format: .xml, options: 0)
            try data.write(to: path, options: .atomic)
        } catch {
            NSLog("Could not register Orbit login item: %@", error.localizedDescription)
        }
    }

    private func removeLoginAgent() {
        try? FileManager.default.removeItem(at: loginAgentPath())
    }

    private func prepareOrbit() async {
        if !(await isHealthy()) {
            let orbit = orbitExecutable()
            do {
                if FileManager.default.isExecutableFile(atPath: orbit) {
                    await setStatus("正在启动本机 API…")
                    try await run(orbit, ["start"])
                } else {
                    await setStatus("首次启动：正在安装本机 AI 运行时…")
                    guard let installer = Bundle.main.path(forResource: "install", ofType: "sh") else { throw NSError(domain: "Orbit", code: 1, userInfo: [NSLocalizedDescriptionKey: "安装器缺失"]) }
                    try await run("/usr/bin/env", ["ORBIT_NO_BROWSER=1", "/bin/sh", installer]) { line in Task { @MainActor in self.statusLabel.stringValue = line } }
                }
                for _ in 0..<80 {
                    if await isHealthy() { break }
                    try await Task.sleep(for: .milliseconds(250))
                }
            } catch {
                await setStatus("无法启动 Orbit：\(error.localizedDescription)")
                return
            }
        }
        guard await isHealthy() else { await setStatus("本机 API 未能启动，请重新打开 Orbit"); return }
        await MainActor.run {
            webView.load(URLRequest(url: orbitURL))
            webView.isHidden = false
            spinner.isHidden = true
            statusLabel.isHidden = true
        }
        await monitorOrbitService()
    }

    private func monitorOrbitService() async {
        while !Task.isCancelled && !terminationInProgress {
            try? await Task.sleep(for: .seconds(5))
            if terminationInProgress { return }
            if !(await isHealthy()) {
                try? await runOrbit(["start"])
                for _ in 0..<20 {
                    if await isHealthy() { break }
                    try? await Task.sleep(for: .milliseconds(250))
                }
            }
        }
    }
}

@main
private enum OrbitMain {
    private static let delegate = OrbitApp()

    static func main() {
        let application = NSApplication.shared
        application.delegate = delegate
        application.run()
    }
}
