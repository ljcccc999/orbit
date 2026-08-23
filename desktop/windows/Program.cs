using System.Diagnostics;
using System.Net;
using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.Win32;
using Microsoft.Web.WebView2.WinForms;

namespace OrbitDesktop;

internal static class Program
{
    private static readonly Uri OrbitUri = new("http://127.0.0.1:8765");
    private const string MutexName = @"Local\OrbitDesktop";
    private const string ShowEventName = @"Local\OrbitDesktopShow";

    [STAThread]
    private static void Main()
    {
        using var mutex = new Mutex(true, MutexName, out var firstInstance);
        if (!firstInstance)
        {
            try { using var signal = EventWaitHandle.OpenExisting(ShowEventName); signal.Set(); } catch { }
            return;
        }

        using var showEvent = new EventWaitHandle(false, EventResetMode.AutoReset, ShowEventName);
        ApplicationConfiguration.Initialize();
        Application.Run(new OrbitForm(showEvent));
    }

    private sealed class OrbitForm : Form
    {
        private readonly Label status = new() { Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleCenter, Text = "Preparing Orbit…", Font = new Font("Segoe UI", 12) };
        private readonly WebView2 web = new() { Dock = DockStyle.Fill, Visible = false };
        private readonly NotifyIcon tray;
        private bool explicitExit;

        public OrbitForm(EventWaitHandle showEvent)
        {
            Text = "Orbit"; Width = 1280; Height = 820; MinimumSize = new Size(980, 640); StartPosition = FormStartPosition.CenterScreen;
            Controls.Add(web); Controls.Add(status);
            var appIcon = LoadOrbitIcon("OrbitDesktop.orbit-logo.png");
            var trayIcon = LoadOrbitIcon("OrbitDesktop.orbit-logo-transparent.png");
            Icon = appIcon;
            tray = new NotifyIcon
            {
                Icon = trayIcon,
                Text = "Orbit · Local AI",
                Visible = true,
                ContextMenuStrip = BuildTrayMenu(),
            };
            tray.DoubleClick += (_, _) => ShowOrbit();
            Shown += async (_, _) => await PrepareAsync();
            FormClosing += OnFormClosing;
            Resize += (_, _) => { if (WindowState == FormWindowState.Minimized) HideToTray(); };
            RegisterStartup();
            _ = Task.Run(() => WaitForShow(showEvent));
        }

        private ContextMenuStrip BuildTrayMenu()
        {
            var menu = new ContextMenuStrip();
            menu.Items.Add("Open Orbit", null, (_, _) => ShowOrbit());
            menu.Items.Add(new ToolStripMenuItem("Local API · 127.0.0.1:8765") { Enabled = false });
            menu.Items.Add("Unload Model", null, async (_, _) => await RunOrbitQuietlyAsync("unload"));
            menu.Items.Add(new ToolStripSeparator());
            menu.Items.Add("Exit App (Agent Keeps Running)", null, (_, _) => ExitApp());
            menu.Items.Add("Stop Background Agent", null, async (_, _) => await StopAgentAsync());
            return menu;
        }

        private static Icon LoadOrbitIcon(string resourceName)
        {
            using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName) ?? throw new Exception("Orbit logo is missing.");
            using var bitmap = new Bitmap(stream);
            var handle = bitmap.GetHicon();
            try { using var temporary = Icon.FromHandle(handle); return (Icon)temporary.Clone(); }
            finally { DestroyIcon(handle); }
        }

        [DllImport("user32.dll", CharSet = CharSet.Auto)]
        private static extern bool DestroyIcon(IntPtr handle);

        private void WaitForShow(EventWaitHandle showEvent)
        {
            while (!IsDisposed)
            {
                showEvent.WaitOne();
                if (!IsDisposed && IsHandleCreated) BeginInvoke((Action)ShowOrbit);
            }
        }

        private void ShowOrbit()
        {
            ShowInTaskbar = true;
            Show();
            if (WindowState == FormWindowState.Minimized) WindowState = FormWindowState.Normal;
            Activate();
            BringToFront();
        }

        private void HideToTray()
        {
            Hide();
            ShowInTaskbar = false;
        }

        private void OnFormClosing(object? sender, FormClosingEventArgs eventArgs)
        {
            if (eventArgs.CloseReason == CloseReason.WindowsShutDown)
            {
                explicitExit = true;
                tray.Visible = false;
                return;
            }
            if (!explicitExit)
            {
                eventArgs.Cancel = true;
                HideToTray();
            }
        }

        private static void RegisterStartup()
        {
            using var key = Registry.CurrentUser.CreateSubKey(@"Software\Microsoft\Windows\CurrentVersion\Run");
            key?.SetValue("Orbit", $"\"{Application.ExecutablePath}\"");
        }

        private static void UnregisterStartup()
        {
            using var key = Registry.CurrentUser.OpenSubKey(@"Software\Microsoft\Windows\CurrentVersion\Run", writable: true);
            key?.DeleteValue("Orbit", throwOnMissingValue: false);
        }

        private static async Task<bool> HealthyAsync()
        {
            try
            {
                using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(1) };
                return (await client.GetAsync(new Uri(OrbitUri, "/api/health"))).StatusCode == HttpStatusCode.OK;
            }
            catch { return false; }
        }

        private async Task RunAsync(string file, string arguments)
        {
            var process = new Process { StartInfo = new ProcessStartInfo(file, arguments) { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true } };
            process.OutputDataReceived += (_, e) => { if (!string.IsNullOrWhiteSpace(e.Data)) BeginInvoke(() => status.Text = e.Data); };
            process.ErrorDataReceived += (_, e) => { if (!string.IsNullOrWhiteSpace(e.Data)) BeginInvoke(() => status.Text = e.Data); };
            process.Start(); process.BeginOutputReadLine(); process.BeginErrorReadLine(); await process.WaitForExitAsync();
            if (process.ExitCode != 0) throw new Exception($"Orbit setup failed ({process.ExitCode}).");
        }

        private static string OrbitExecutable() => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Orbit", "runtime", "Scripts", "orbit.exe");

        private async Task RunOrbitQuietlyAsync(string arguments)
        {
            var orbit = OrbitExecutable();
            if (!File.Exists(orbit)) return;
            try { await RunAsync(orbit, arguments); } catch { }
        }

        private void ExitApp()
        {
            UnregisterStartup();
            tray.Visible = false;
            explicitExit = true;
            Close();
            Application.Exit();
        }

        private async Task StopAgentAsync()
        {
            await RunOrbitQuietlyAsync("service uninstall");
            ExitApp();
        }

        private static string ExtractInstaller()
        {
            var path = Path.Combine(Path.GetTempPath(), $"orbit-install-{Guid.NewGuid():N}.ps1");
            using var source = Assembly.GetExecutingAssembly().GetManifestResourceStream("OrbitDesktop.install.ps1") ?? throw new Exception("Installer is missing.");
            using var target = File.Create(path); source.CopyTo(target); return path;
        }

        private async Task PrepareAsync()
        {
            try
            {
                if (!await HealthyAsync())
                {
                    var orbit = OrbitExecutable();
                    if (File.Exists(orbit)) await RunAsync(orbit, "start");
                    else
                    {
                        status.Text = "First launch: installing the local AI runtime…";
                        var installer = ExtractInstaller();
                        try { await RunAsync("powershell.exe", $"-NoProfile -ExecutionPolicy Bypass -File \"{installer}\""); }
                        finally { File.Delete(installer); }
                    }
                    for (var i = 0; i < 80 && !await HealthyAsync(); i++) await Task.Delay(250);
                }
                if (!await HealthyAsync()) throw new Exception("The local API did not start.");
                await web.EnsureCoreWebView2Async(); web.Source = OrbitUri; web.Visible = true; status.Visible = false;
            }
            catch (Exception error) { status.Text = "Orbit could not start:\n" + error.Message; }
        }
    }
}
