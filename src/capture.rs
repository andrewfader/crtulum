// M2: live screen/window capture via the xdg-desktop-portal ScreenCast interface
// + PipeWire (the same path OBS uses). Runs on its own thread and drops the latest
// frame into a shared slot that the render loop polls.

use std::os::fd::OwnedFd;
use std::sync::{Arc, Mutex};

use pipewire as pw;

/// One captured frame, packed tight (4 bytes/px, no row padding).
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub is_bgra: bool, // true = BGRA byte order, false = RGBA
    pub data: Vec<u8>,
    pub seq: u64,
}

pub type SharedFrame = Arc<Mutex<Option<Frame>>>;

/// Kicks off capture on a background thread. Returns immediately with the shared
/// frame slot; frames start appearing once the user picks a source in the portal.
pub fn spawn() -> SharedFrame {
    let shared: SharedFrame = Arc::new(Mutex::new(None));
    let handle = shared.clone();
    std::thread::spawn(move || {
        if let Err(e) = run(handle) {
            eprintln!("[capture] error: {e:#}");
        }
    });
    shared
}

fn run(shared: SharedFrame) -> anyhow::Result<()> {
    let (fd, node_id) = pollster::block_on(portal())?;
    eprintln!("[capture] portal ok, pipewire node {node_id}");
    pipewire_loop(fd, node_id, shared)
}

/// Portal handshake → (pipewire remote fd, node id). Pops the "pick a window/screen"
/// dialog on the user's desktop.
async fn portal() -> anyhow::Result<(OwnedFd, u32)> {
    use ashpd::desktop::screencast::{CursorMode, Screencast, SelectSourcesOptions, SourceType};
    use ashpd::desktop::PersistMode;

    let proxy = Screencast::new().await?;
    let session = proxy.create_session(Default::default()).await?;
    let options = SelectSourcesOptions::default()
        .set_cursor_mode(CursorMode::Embedded)
        .set_sources(SourceType::Monitor | SourceType::Window)
        .set_multiple(false)
        .set_persist_mode(PersistMode::DoNot);
    proxy.select_sources(&session, options).await?;

    let response = proxy
        .start(&session, None, Default::default())
        .await?
        .response()?;
    let stream = response
        .streams()
        .first()
        .ok_or_else(|| anyhow::anyhow!("portal returned no streams"))?;
    let node_id = stream.pipe_wire_node_id();
    let fd = proxy
        .open_pipe_wire_remote(&session, Default::default())
        .await?;

    // Keep the portal session (and its D-Bus connection) alive for the process
    // lifetime so the stream isn't torn down. Pragmatic for M2.
    std::mem::forget(session);
    std::mem::forget(proxy);

    Ok((fd, node_id))
}

// User data threaded through the PipeWire stream callbacks.
struct UserData {
    width: u32,
    height: u32,
    is_bgra: bool,
    seq: u64,
    shared: SharedFrame,
}

fn pipewire_loop(fd: OwnedFd, node_id: u32, shared: SharedFrame) -> anyhow::Result<()> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_fd_rc(fd, None)?;

    let stream = pw::stream::StreamBox::new(
        &core,
        "crtulum-capture",
        pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Screen",
        },
    )?;

    let data = UserData {
        width: 0,
        height: 0,
        is_bgra: true,
        seq: 0,
        shared,
    };

    let _listener = stream
        .add_local_listener_with_user_data(data)
        .state_changed(|_, _, old, new| {
            eprintln!("[capture] stream state {old:?} -> {new:?}");
        })
        .param_changed(|_, ud, id, param| {
            let Some(param) = param else { return };
            if id != pw::spa::param::ParamType::Format.as_raw() {
                return;
            }
            let Ok((mt, ms)) = pw::spa::param::format_utils::parse_format(param) else {
                return;
            };
            if mt != pw::spa::param::format::MediaType::Video
                || ms != pw::spa::param::format::MediaSubtype::Raw
            {
                return;
            }
            let mut info = pw::spa::param::video::VideoInfoRaw::new();
            if info.parse(param).is_err() {
                return;
            }
            ud.width = info.size().width;
            ud.height = info.size().height;
            ud.is_bgra = matches!(
                info.format(),
                pw::spa::param::video::VideoFormat::BGRx
                    | pw::spa::param::video::VideoFormat::BGRA
            );
            eprintln!(
                "[capture] negotiated {}x{} {:?}",
                ud.width,
                ud.height,
                info.format()
            );
        })
        .process(|stream, ud| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            let Some(d) = datas.first_mut() else {
                return;
            };
            let stride = d.chunk().stride().max(0) as usize;
            let (w, h) = (ud.width as usize, ud.height as usize);
            if w == 0 || h == 0 {
                return;
            }
            let row_bytes = w * 4;
            let Some(src) = d.data() else {
                return;
            };
            if stride < row_bytes || src.len() < stride * h {
                return;
            }
            // repack, stripping any row padding
            let mut packed = vec![0u8; row_bytes * h];
            for y in 0..h {
                let s = y * stride;
                packed[y * row_bytes..(y + 1) * row_bytes]
                    .copy_from_slice(&src[s..s + row_bytes]);
            }
            ud.seq += 1;
            let frame = Frame {
                width: ud.width,
                height: ud.height,
                is_bgra: ud.is_bgra,
                data: packed,
                seq: ud.seq,
            };
            if let Ok(mut guard) = ud.shared.lock() {
                *guard = Some(frame);
            }
        })
        .register()?;

    // Advertise the raw video formats we can consume, any reasonable size/rate.
    let obj = pw::spa::pod::object!(
        pw::spa::utils::SpaTypes::ObjectParamFormat,
        pw::spa::param::ParamType::EnumFormat,
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::MediaType,
            Id,
            pw::spa::param::format::MediaType::Video
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::MediaSubtype,
            Id,
            pw::spa::param::format::MediaSubtype::Raw
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::VideoFormat,
            Choice,
            Enum,
            Id,
            pw::spa::param::video::VideoFormat::BGRx,
            pw::spa::param::video::VideoFormat::RGBx,
            pw::spa::param::video::VideoFormat::BGRA,
            pw::spa::param::video::VideoFormat::RGBA
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::VideoSize,
            Choice,
            Range,
            Rectangle,
            pw::spa::utils::Rectangle {
                width: 1920,
                height: 1080
            },
            pw::spa::utils::Rectangle {
                width: 1,
                height: 1
            },
            pw::spa::utils::Rectangle {
                width: 8192,
                height: 8192
            }
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::VideoFramerate,
            Choice,
            Range,
            Fraction,
            pw::spa::utils::Fraction { num: 60, denom: 1 },
            pw::spa::utils::Fraction { num: 0, denom: 1 },
            pw::spa::utils::Fraction {
                num: 240,
                denom: 1
            }
        ),
    );
    let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(obj),
    )
    .map_err(|e| anyhow::anyhow!("pod serialize: {e:?}"))?
    .0
    .into_inner();
    let mut params = [pw::spa::pod::Pod::from_bytes(&values)
        .ok_or_else(|| anyhow::anyhow!("bad format pod"))?];

    stream.connect(
        pw::spa::utils::Direction::Input,
        Some(node_id),
        pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
        &mut params,
    )?;

    mainloop.run();
    Ok(())
}
