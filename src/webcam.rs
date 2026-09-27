//! Optional local V4L2 capture. The renderer consumes only the latest frame;
//! stopping capture kills/reaps ffmpeg and releases the camera immediately.
use std::{io::Read, process::{Child, Command, Stdio}, sync::{Arc, Mutex}, thread::JoinHandle, time::Instant};

pub const WIDTH: u32 = 640;
pub const HEIGHT: u32 = 480;

#[derive(Default)]
struct Latest {
    frame: Option<Vec<u8>>,
    error: Option<String>,
}

pub struct Webcam {
    child: Child,
    reader: Option<JoinHandle<()>>,
    latest: Arc<Mutex<Latest>>,
    last_frame: Instant,
    pub tan_half_fov: f32,
}

impl Webcam {
    pub fn start() -> anyhow::Result<Self> {
        let device = std::env::var("CRTULUM_WEBCAM_DEVICE").unwrap_or_else(|_| "/dev/video0".into());
        let fov: f32 = std::env::var("CRTULUM_WEBCAM_FOV").unwrap_or_else(|_| "70".into()).parse()?;
        anyhow::ensure!(fov.is_finite() && (20.0..=140.0).contains(&fov), "webcam horizontal FOV must be 20–140 degrees");
        let mut command = Command::new("ffmpeg");
        command.args(["-nostdin", "-hide_banner", "-loglevel", "error", "-f", "v4l2", "-video_size", "640x480", "-framerate", "30", "-i"])
            .arg(&device)
            // Preserve the camera's aspect ratio by cropping, not stretching.
            .args(["-an", "-vf", "fps=30,scale=640:480:force_original_aspect_ratio=increase,crop=640:480", "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"]);
        let capture = Self::launch(command, fov)?;
        eprintln!("[webcam] opening {device}; F4 stops capture");
        Ok(capture)
    }

    fn launch(mut command: Command, fov: f32) -> anyhow::Result<Self> {
        let mut child = command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn()?;
        let mut stdout = child.stdout.take().unwrap();
        let latest = Arc::new(Mutex::new(Latest::default()));
        let shared = latest.clone();
        let reader = std::thread::spawn(move || loop {
            let mut frame = vec![0; (WIDTH * HEIGHT * 4) as usize];
            if let Err(error) = stdout.read_exact(&mut frame) {
                shared.lock().unwrap().error = Some(format!("camera stream ended: {error}"));
                break;
            }
            shared.lock().unwrap().frame = Some(frame);
        });
        Ok(Self { child, reader: Some(reader), latest, last_frame: Instant::now(), tan_half_fov: (fov.to_radians() * 0.5).tan() })
    }

    pub fn poll(&mut self) -> anyhow::Result<Option<Vec<u8>>> {
        let mut latest = self.latest.lock().unwrap();
        if let Some(error) = latest.error.take() { anyhow::bail!(error); }
        let frame = latest.frame.take();
        if frame.is_some() { self.last_frame = Instant::now(); }
        anyhow::ensure!(self.last_frame.elapsed().as_secs() < 5, "camera timed out (no frames for 5 seconds)");
        Ok(frame)
    }
}

impl Drop for Webcam {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() { let _ = reader.join(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_delivers_frames_and_stop_reaps_the_process() {
        let mut command = Command::new("ffmpeg");
        command.args(["-nostdin", "-hide_banner", "-loglevel", "error", "-re", "-f", "lavfi", "-i", "color=red:size=640x480:rate=30", "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"]);
        let mut capture = Webcam::launch(command, 70.0).unwrap();
        let pid = capture.child.id();
        let frame = loop {
            if let Some(frame) = capture.poll().unwrap() { break frame; }
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        assert_eq!(frame.len(), (WIDTH * HEIGHT * 4) as usize);
        assert!(frame[0] > 240 && frame[1] < 10 && frame[2] < 10);
        drop(capture);
        assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
    }

    #[test]
    #[ignore = "opens a physical webcam; run explicitly with --ignored"]
    fn physical_camera_can_be_stopped_and_reopened() {
        for _ in 0..2 {
            let mut capture = Webcam::start().unwrap();
            loop {
                if let Some(frame) = capture.poll().unwrap() {
                    assert_eq!(frame.len(), (WIDTH * HEIGHT * 4) as usize);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            // Drop must release the camera before the next open.
        }
    }

    #[test]
    fn failed_capture_reports_error_instead_of_showing_a_stale_frame() {
        let mut command = Command::new("sh");
        command.args(["-c", "printf partial"]);
        let mut capture = Webcam::launch(command, 70.0).unwrap();
        capture.reader.take().unwrap().join().unwrap();
        assert!(capture.poll().is_err());
    }
}
