import AppKit

private final class OrbitServiceMenu: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
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
            button.toolTip = "Orbit local API"
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
        let quit = NSMenuItem(title: "Stop Background Agent", action: #selector(quitAPI), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)
        statusItem.menu = menu
    }

    @objc private func openOrbit() {
        let app = Bundle.main.bundleURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        NSWorkspace.shared.open(app)
    }

    @objc private func unloadModel() {
        Task { try? await runOrbit(["unload"]) }
    }

    @objc private func quitAPI() {
        Task {
            try? await runOrbit(["service", "uninstall"])
            await MainActor.run {
                // The desktop App normally keeps the API alive after its
                // window is closed.  A menu-bar quit must be the explicit
                // exception, so terminate that companion first; otherwise
                // its health monitor would immediately start the API again.
                NSRunningApplication.runningApplications(withBundleIdentifier: "top.orbit.desktop")
                    .forEach { $0.terminate() }
                removeServiceMenuAgent()
                NSApp.terminate(nil)
            }
        }
    }

    private func removeServiceMenuAgent() {
        let path = FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "Library/LaunchAgents/top.orbit.service-menu.plist")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["bootout", "gui/\(getuid())", "top.orbit.service-menu"]
        try? process.run()
        process.waitUntilExit()
        try? FileManager.default.removeItem(at: path)
    }

    private func orbitExecutable() -> String {
        FileManager.default.homeDirectoryForCurrentUser.appending(path: ".orbit/runtime/bin/orbit").path
    }

    private func run(_ executable: String, _ arguments: [String]) async throws {
        try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = arguments
            process.terminationHandler = { task in
                if task.terminationStatus == 0 { continuation.resume() }
                else { continuation.resume(throwing: NSError(domain: "Orbit", code: Int(task.terminationStatus))) }
            }
            do { try process.run() } catch { continuation.resume(throwing: error) }
        }
    }

    private func runOrbit(_ arguments: [String]) async throws {
        let executable = orbitExecutable()
        guard FileManager.default.isExecutableFile(atPath: executable) else { return }
        try await run(executable, arguments)
    }
}

@main
private enum OrbitServiceMenuMain {
    private static let delegate = OrbitServiceMenu()

    static func main() {
        let application = NSApplication.shared
        application.delegate = delegate
        application.run()
    }
}
