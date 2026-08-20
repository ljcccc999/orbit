import AppKit
import WebKit

private let orbitURL = URL(string: "http://127.0.0.1:8765")!

@main
final class OrbitApp: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var statusLabel: NSTextField!
    private var spinner: NSProgressIndicator!

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        buildWindow()
        NSApp.activate(ignoringOtherApps: true)
        Task { await prepareOrbit() }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag { window.makeKeyAndOrderFront(nil) }
        return true
    }

    private func buildWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1180, height: 780), styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView], backing: .buffered, defer: false)
        window.title = "Orbit"
        window.titlebarAppearsTransparent = true
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
                else { continuation.resume(throwing: NSError(domain: "Orbit", code: Int(task.terminationStatus), userInfo: [NSLocalizedDescriptionKey: "Orbit 安装失败（代码 \(task.terminationStatus)）"])) }
            }
            do { try process.run() } catch { continuation.resume(throwing: error) }
        }
    }

    private func prepareOrbit() async {
        if !(await isHealthy()) {
            let orbit = FileManager.default.homeDirectoryForCurrentUser.appending(path: ".orbit/runtime/bin/orbit").path
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
    }
}
