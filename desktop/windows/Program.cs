using System.Diagnostics;
using System.Net;
using System.Reflection;
using Microsoft.Web.WebView2.WinForms;

namespace OrbitDesktop;

internal static class Program
{
    private static readonly Uri OrbitUri = new("http://127.0.0.1:8765");

    [STAThread]
    private static void Main()
    {
        ApplicationConfiguration.Initialize();
        Application.Run(new OrbitForm());
    }

    private sealed class OrbitForm : Form
    {
        private readonly Label status = new() { Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleCenter, Text = "Preparing Orbit…", Font = new Font("Segoe UI", 12) };
        private readonly WebView2 web = new() { Dock = DockStyle.Fill, Visible = false };

        public OrbitForm()
        {
            Text = "Orbit"; Width = 1180; Height = 780; MinimumSize = new Size(820, 600); StartPosition = FormStartPosition.CenterScreen;
            Controls.Add(web); Controls.Add(status); Shown += async (_, _) => await PrepareAsync();
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
                    var orbit = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Orbit", "runtime", "Scripts", "orbit.exe");
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
