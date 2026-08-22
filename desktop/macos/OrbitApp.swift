import AppKit
import CoreGraphics
import Darwin
import WebKit

private let orbitURL = URL(string: "http://127.0.0.1:8765")!
private let desktopAgentLabel = "top.orbit.desktop"
private let serviceMenuAgentLabel = "top.orbit.service-menu"
private let serviceMenuBundleIdentifier = "top.orbit.service-menu"

private final class LogoShineView: NSView {
    var image: NSImage? { didSet { needsDisplay = true } }
    private var shineOffset: CGFloat = -0.45
    private var shineTimer: Timer?

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        wantsLayer = true
    }

    func startShine() {
        guard shineTimer == nil else { return }
        shineTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            shineOffset += 0.022
            if shineOffset > 1.45 { shineOffset = -0.45 }
            needsDisplay = true
        }
    }

    func stopShine() {
        shineTimer?.invalidate()
        shineTimer = nil
    }

    deinit { stopShine() }

    override func draw(_ dirtyRect: NSRect) {
        guard let image else { return }
        image.draw(in: bounds, from: .zero, operation: .sourceOver, fraction: 1.0, respectFlipped: true, hints: nil)
        var proposed = bounds
        guard let mask = image.cgImage(forProposedRect: &proposed, context: nil, hints: nil),
              let context = NSGraphicsContext.current?.cgContext,
              let gradient = CGGradient(
                colorsSpace: CGColorSpaceCreateDeviceRGB(),
                colors: [NSColor.clear.cgColor, NSColor.white.withAlphaComponent(0.62).cgColor, NSColor.clear.cgColor] as CFArray,
                locations: [0.0, 0.5, 1.0]
              ) else { return }
        context.saveGState()
        context.clip(to: bounds, mask: mask)
        let start = CGPoint(x: bounds.width * (shineOffset - 0.42), y: bounds.height * 0.05)
        let end = CGPoint(x: bounds.width * (shineOffset + 0.42), y: bounds.height * 0.95)
        context.drawLinearGradient(gradient, start: start, end: end, options: [])
        context.restoreGState()
    }
}

private final class WindowDragHandle: NSView {
    override var mouseDownCanMoveWindow: Bool { true }

    override func mouseDown(with event: NSEvent) {
        window?.performDrag(with: event)
    }
}

/// WKWebView consumes mouse events before the window's background-move
/// behavior can see them.  Keep normal clicks intact, but turn a sustained
/// press into a native window drag so text is not selected while repositioning
/// the App window.
private final class OrbitWebView: WKWebView {
    private var dragTimer: Timer?
    private var initialMouseDown: NSEvent?

    override func mouseDown(with event: NSEvent) {
        initialMouseDown = event
        dragTimer?.invalidate()
        dragTimer = Timer.scheduledTimer(withTimeInterval: 0.28, repeats: false) { [weak self] _ in
            guard let self, let initialMouseDown, let window = self.window, window.isKeyWindow else { return }
            window.performDrag(with: initialMouseDown)
        }
        super.mouseDown(with: event)
    }

    override func mouseUp(with event: NSEvent) {
        dragTimer?.invalidate()
        dragTimer = nil
        initialMouseDown = nil
        super.mouseUp(with: event)
    }

    override func rightMouseDown(with event: NSEvent) {
        dragTimer?.invalidate()
        dragTimer = nil
        initialMouseDown = nil
        super.rightMouseDown(with: event)
    }

    deinit { dragTimer?.invalidate() }
}

final class OrbitApp: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var splashLogo: LogoShineView!
    private var statusLabel: NSTextField!
    private var spinner: NSProgressIndicator!
    private var terminationInProgress = false
    private var systemIsShuttingDown = false
    private var pageLoadAttempts = 0
    private var pageLoadTask: Task<Void, Never>?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        installLoginAgent()
        installServiceMenuAgent()
        ensureServiceMenu()
        observeShutdown()
        buildApplicationMenu()
        buildWindow()
        NSApp.activate(ignoringOtherApps: true)
        Task { await prepareOrbit() }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    @MainActor
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        showWindow()
        return true
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if terminationInProgress { return .terminateNow }
        terminationInProgress = true
        if !systemIsShuttingDown {
            removeLoginAgent()
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

    private func buildWindow() {
        // Desktop-first default size close to the ChatGPT macOS window.
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 820), styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView], backing: .buffered, defer: false)
        window.delegate = self
        window.title = "Orbit"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.titlebarSeparatorStyle = .none
        window.isMovableByWindowBackground = true
        window.center()
        window.minSize = NSSize(width: 980, height: 640)

        let container = NSView(frame: window.contentView!.bounds)
        container.autoresizingMask = [.width, .height]
        window.contentView = container
        let webConfiguration = WKWebViewConfiguration()
        let preferredLanguage = Locale.preferredLanguages.first?.lowercased() ?? Locale.current.identifier.lowercased()
        let nativeLanguage = preferredLanguage.hasPrefix("zh") ? "zh" : "en"
        let languageScript = WKUserScript(
            source: "window.orbitNativeLanguage='\(nativeLanguage)';",
            injectionTime: .atDocumentStart,
            forMainFrameOnly: true
        )
        webConfiguration.userContentController.addUserScript(languageScript)
        webView = OrbitWebView(frame: container.bounds, configuration: webConfiguration)
        webView.autoresizingMask = [.width, .height]
        webView.navigationDelegate = self
        // Keep the native web view transparent so the loaded Orbit surface is
        // visible immediately instead of being covered by WebKit's default
        // opaque white backing during a delayed first paint.
        webView.setValue(false, forKey: "drawsBackground")
        webView.wantsLayer = true
        webView.layer?.backgroundColor = NSColor.clear.cgColor
        webView.isHidden = true
        container.addSubview(webView)

        let dragHandle = WindowDragHandle(frame: .zero)
        dragHandle.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(dragHandle)
        NSLayoutConstraint.activate([
            dragHandle.topAnchor.constraint(equalTo: container.topAnchor),
            dragHandle.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 78),
            dragHandle.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            dragHandle.heightAnchor.constraint(equalToConstant: 28),
        ])

        splashLogo = LogoShineView(frame: .zero)
        if let path = Bundle.main.path(forResource: "orbit-logo-transparent", ofType: "png") {
            splashLogo.image = NSImage(contentsOfFile: path)
        }
        splashLogo.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(splashLogo)
        splashLogo.startShine()

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
            splashLogo.widthAnchor.constraint(equalToConstant: 104),
            splashLogo.heightAnchor.constraint(equalToConstant: 104),
            splashLogo.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            splashLogo.centerYAnchor.constraint(equalTo: container.centerYAnchor, constant: -106),
            spinner.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            spinner.centerYAnchor.constraint(equalTo: container.centerYAnchor, constant: -22),
            statusLabel.topAnchor.constraint(equalTo: spinner.bottomAnchor, constant: 14),
            statusLabel.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            statusLabel.leadingAnchor.constraint(greaterThanOrEqualTo: container.leadingAnchor, constant: 30),
        ])
        window.makeKeyAndOrderFront(nil)
    }

    @MainActor
    private func showWindow() {
        NSApp.setActivationPolicy(.regular)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        if webView.url == nil || webView.isLoading {
            loadOrbitPage()
        }
    }

    @MainActor private func setStatus(_ text: String) { statusLabel.stringValue = text }

    @MainActor
    private func loadOrbitPage() {
        pageLoadTask?.cancel()
        pageLoadAttempts += 1
        spinner.isHidden = false
        spinner.startAnimation(nil)
        statusLabel.isHidden = false
        statusLabel.stringValue = pageLoadAttempts > 1 ? "页面加载失败，正在重试 Orbit…" : "正在加载 Orbit…"
        webView.isHidden = true

        var request = URLRequest(url: orbitURL.appending(queryItems: [
            URLQueryItem(name: "desktop", value: "1"),
            URLQueryItem(name: "reload", value: String(pageLoadAttempts))
        ]))
        request.cachePolicy = .reloadIgnoringLocalCacheData
        webView.load(request)
    }

    @MainActor
    private func retryOrbitPage(after error: Error) {
        webView.stopLoading()
        webView.isHidden = true
        spinner.stopAnimation(nil)
        spinner.isHidden = true
        statusLabel.isHidden = false
        statusLabel.stringValue = "Orbit 页面暂时无法加载，正在重新连接本机服务…"
        pageLoadTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(500))
            guard !Task.isCancelled, let self else { return }
            self.loadOrbitPage()
        }
    }

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        Task { @MainActor in
            self.spinner.isHidden = false
            self.spinner.startAnimation(nil)
            self.statusLabel.isHidden = false
            self.statusLabel.stringValue = "正在加载 Orbit…"
            self.webView.isHidden = true
        }
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        pageLoadTask?.cancel()
        pageLoadAttempts = 0
        spinner.stopAnimation(nil)
        spinner.isHidden = true
        statusLabel.isHidden = true
        splashLogo.stopShine()
        splashLogo.isHidden = true
        webView.isHidden = false
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        Task { @MainActor in self.retryOrbitPage(after: error) }
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        Task { @MainActor in self.retryOrbitPage(after: error) }
    }

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

    private func serviceMenuBundle() -> URL? {
        guard let resourceURL = Bundle.main.resourceURL else { return nil }
        let bundle = resourceURL.appending(path: "../Helpers/OrbitServiceMenu.app").standardizedFileURL
        return FileManager.default.fileExists(atPath: bundle.path) ? bundle : nil
    }

    private func serviceMenuAgentPath() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "Library/LaunchAgents/\(serviceMenuAgentLabel).plist")
    }

    private func installServiceMenuAgent() {
        guard let bundle = serviceMenuBundle() else { return }
        let executable = bundle.appending(path: "Contents/MacOS/OrbitServiceMenu").path
        let path = serviceMenuAgentPath()
        let payload: [String: Any] = [
            "Label": serviceMenuAgentLabel,
            "ProgramArguments": [executable],
            "RunAtLoad": true,
            "KeepAlive": true,
            "ProcessType": "Background",
        ]
        do {
            try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = try PropertyListSerialization.data(fromPropertyList: payload, format: .xml, options: 0)
            try data.write(to: path, options: .atomic)
            activateLaunchAgent(label: serviceMenuAgentLabel, path: path)
        } catch {
            NSLog("Could not register Orbit API menu item: %@", error.localizedDescription)
        }
    }

    private func ensureServiceMenu() {
        guard let bundle = serviceMenuBundle() else { return }
        if NSRunningApplication.runningApplications(withBundleIdentifier: serviceMenuBundleIdentifier).isEmpty {
            let path = serviceMenuAgentPath()
            activateLaunchAgent(label: serviceMenuAgentLabel, path: path)
            if NSRunningApplication.runningApplications(withBundleIdentifier: serviceMenuBundleIdentifier).isEmpty {
                NSWorkspace.shared.open(bundle)
            }
        }
    }

    /// Writing a LaunchAgent plist does not load it into the current launchd
    /// session.  Without bootstrapping it here, the status-bar helper only
    /// appears after a later login (and can disappear when the app is hidden).
    private func activateLaunchAgent(label: String, path: URL) {
        let domain = "gui/\(getuid())"
        let service = "\(domain)/\(label)"

        let print = Process()
        print.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        print.arguments = ["print", service]
        try? print.run()
        print.waitUntilExit()
        if print.terminationStatus == 0 { return }

        // Do not create a second menu-bar process if the helper was already
        // started by LaunchServices; just leave the newly written agent for
        // the next login.
        if !NSRunningApplication.runningApplications(withBundleIdentifier: serviceMenuBundleIdentifier).isEmpty {
            return
        }

        let bootstrap = Process()
        bootstrap.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        bootstrap.arguments = ["bootstrap", domain, path.path]
        do {
            try bootstrap.run()
            bootstrap.waitUntilExit()
        } catch {
            NSLog("Could not load Orbit LaunchAgent: %@", error.localizedDescription)
        }
    }

    private func prepareOrbit() async {
        if !(await isHealthy()) {
            let orbit = orbitExecutable()
            do {
                if FileManager.default.isExecutableFile(atPath: orbit) {
                    await MainActor.run { self.statusLabel.stringValue = "正在启动本机 API…" }
                    try await run(orbit, ["start"])
                } else {
                    await MainActor.run { self.statusLabel.stringValue = "首次启动：正在安装本机 AI 运行时…" }
                    guard let installer = Bundle.main.path(forResource: "install", ofType: "sh") else { throw NSError(domain: "Orbit", code: 1, userInfo: [NSLocalizedDescriptionKey: "安装器缺失"]) }
                    try await run("/usr/bin/env", ["ORBIT_NO_BROWSER=1", "/bin/sh", installer]) { line in Task { @MainActor in self.statusLabel.stringValue = line } }
                }
                for _ in 0..<80 {
                    if await isHealthy() { break }
                    try await Task.sleep(for: .milliseconds(250))
                }
            } catch {
                await MainActor.run { self.statusLabel.stringValue = "无法启动 Orbit：\(error.localizedDescription)" }
                return
            }
        }
        guard await isHealthy() else {
            await MainActor.run { self.statusLabel.stringValue = "本机 API 未能启动，请重新打开 Orbit" }
            return
        }
        await MainActor.run {
            self.loadOrbitPage()
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
