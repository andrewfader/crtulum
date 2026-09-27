//! Nested gamescope launch; only the child receives WSI/backend overrides.
use anyhow::{bail, Context, Result};
use std::process::{Command, ExitStatus};

pub fn launch(args: &[String]) -> Result<Option<ExitStatus>> {
    let test = args.iter().any(|a| a == "--gamescope-hdr-test");
    if !test && !args.iter().any(|a| a == "--gamescope") {
        return Ok(None);
    }
    if std::env::var_os("CRTULUM_IN_GAMESCOPE").is_some() {
        bail!("already launched inside gamescope; refusing recursive launch");
    }
    let child: Vec<_> = args
        .iter()
        .skip(1)
        .filter(|a| !matches!(a.as_str(), "--gamescope" | "--gamescope-hdr-test"))
        .collect();
    if child.iter().any(|a| {
        matches!(
            a.as_str(),
            "--shot" | "--render" | "--clip" | "--fetch-agent"
        )
    }) {
        bail!("gamescope is for live viewing; headless exports already render through Vulkan");
    }
    let mut cmd = Command::new("gamescope");
    cmd.args(["-w", "1000", "-h", "800", "-W", "1000", "-H", "800"]);
    if test {
        // SDL produces SDR output here. Force client HDR support, never force
        // PQ values onto an SDR output (--hdr-debug-force-output is not used).
        cmd.args(["--backend", "sdl", "--hdr-debug-force-support"]);
        eprintln!("[gamescope] HDR client → gamescope tone mapping → SDR desktop test");
    } else {
        cmd.args(["--backend", "wayland", "--hdr-enabled"]);
        eprintln!(
            "[gamescope] HDR where the parent compositor supports it; SDR fallback otherwise"
        );
    }
    // Xwayland is only the window protocol; all rendering remains Vulkan.
    // Use gamescope's WSI layer, not the separate generic HDR layer (which can
    // attach to the parent compositor and hang when it has no color management).
    cmd.args([
        "--",
        "env",
        "-u",
        "WAYLAND_DISPLAY",
        "-u",
        "WAYLAND_SOCKET",
        "DISABLE_HDR_WSI=1",
        "ENABLE_GAMESCOPE_WSI=1",
        "CRTULUM_IN_GAMESCOPE=1",
        "GAMESCOPE_WSI_MIN_IMAGE_COUNT=4",
        "CRTULUM_GAMESCOPE_MAINTENANCE1=1",
    ]);
    // Validate our Vulkan client, not gamescope's private Mesa WSI structures
    // (which are intentionally outside the public Vulkan validation schema).
    if let Some(layers) = std::env::var_os("VK_INSTANCE_LAYERS") {
        let mut setting = std::ffi::OsString::from("VK_INSTANCE_LAYERS=");
        setting.push(layers);
        cmd.env_remove("VK_INSTANCE_LAYERS").arg(setting);
    }
    cmd.arg(std::env::current_exe()?).args(child);
    Ok(Some(cmd.status().context(
        "launching gamescope (install gamescope and its Vulkan WSI layer)",
    )?))
}
