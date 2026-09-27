//! Deterministic, native-resolution GPU pass timings, independent of presentation.
use super::*;
use anyhow::{ensure, Context, Result};

pub fn run(args: &[String], preset: Preset) -> Result<()> {
    let value = |key: &str| -> Result<Option<&str>> {
        args.iter().position(|a| a == key).map(|i| {
            args.get(i + 1).map(String::as_str).filter(|s| !s.starts_with("--"))
                .with_context(|| format!("{key} needs a value"))
        }).transpose()
    };
    let size = value("--benchmark")?.context("--benchmark needs WxH")?;
    let (w, h) = size.split_once('x').context("--benchmark needs WxH")?;
    let (width, height): (u32, u32) = (w.parse()?, h.parse()?);
    let frames: u32 = value("--benchmark-frames")?.unwrap_or("120").parse()?;
    ensure!(width > 0 && height > 0 && frames > 0 && frames <= 10000,
        "dimensions must be positive; frames must be 1..10000");
    let output = value("--benchmark-output")?;
    let instance = gpu::instance();
    let adapter = gpu::adapter(&instance, None)?;
    let features = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    ensure!(adapter.features().contains(features), "GPU timestamp queries are required");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("benchmark"), required_features: features, required_limits: gpu::limits(&adapter),
    }, None))?;
    ensure!(width <= device.limits().max_texture_dimension_2d && height <= device.limits().max_texture_dimension_2d,
        "benchmark dimensions exceed GPU limit");
    let format = wgpu::TextureFormat::Rgba16Float;
    let mut res = build_resources(&device, &queue, format, preset);
    if let Some(size) = value("--benchmark-source")? {
        let (w, h) = size.split_once('x').context("--benchmark-source needs WxH")?;
        let (w, h): (u32, u32) = (w.parse()?, h.parse()?);
        ensure!(w > 0 && h > 0 && w <= device.limits().max_texture_dimension_2d
            && h <= device.limits().max_texture_dimension_2d, "invalid source dimensions");
        ensure!(w as u64 * h as u64 * 11 * 16 <= device.limits().max_storage_buffer_binding_size as u64,
            "source exceeds phosphor history capacity");
        let (sw, sh, pixels) = make_test_pattern();
        let original = image::RgbaImage::from_raw(sw, sh, pixels).unwrap();
        let pixels = image::imageops::resize(&original, w, h, image::imageops::FilterType::Nearest);
        res.set_source(&device, &queue, w, h, wgpu::TextureFormat::Rgba8UnormSrgb, &pixels);
    }
    res.hdr_output = 2; // Same linear BT.2020 encoding as the native GNOME HDR path.
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("benchmark HDR target"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = color.create_view(&Default::default());
    let depth = create_depth(&device, width, height);
    let orbit = Orbit { yaw: 0.24, pitch: 0.18, distance: 2.65 };
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("pass timings"), ty: wgpu::QueryType::Timestamp, count: 4,
    });
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: 32, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: 32, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut accum = Vec::new();
    let mut tube = Vec::new();
    let mut wall = Vec::new();
    // Warm caches/pipelines, but keep the timeline identical between builds.
    for frame in 0..frames + 30 {
        let start = std::time::Instant::now();
        res.exposure_group = frame + 1;
        write_uniforms(&queue, &res, &orbit, width as f32 / height as f32,
            frame as f32 / FIELD_HZ as f32, &preset, 1.0, true, (1.0 / FIELD_HZ) as f32,
            [1.0, 0.0, 0.0, 1.0], 0.0, (frame % 4) as f32, 1.0, false, 1.0, true, true);
        let mut enc = device.create_command_encoder(&Default::default());
        enc.write_timestamp(&queries, 0);
        accum_step(&mut enc, &mut res);
        enc.write_timestamp(&queries, 1);
        enc.write_timestamp(&queries, 2);
        draw_tube(&mut enc, &res, &view, &depth);
        enc.write_timestamp(&queries, 3);
        enc.resolve_query_set(&queries, 0..4, &resolved, 0);
        enc.copy_buffer_to_buffer(&resolved, 0, &readback, 0, 32);
        queue.submit(Some(enc.finish()));
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let data = slice.get_mapped_range();
        let ticks: Vec<u64> = data.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
        if frame >= 30 {
            let ms = queue.get_timestamp_period() as f64 / 1e6;
            accum.push(ticks[1].wrapping_sub(ticks[0]) as f64 * ms);
            tube.push(ticks[3].wrapping_sub(ticks[2]) as f64 * ms);
            wall.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        drop(data);
        readback.unmap();
    }
    if let Some(path) = output {
        let stride = (width * 8).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: stride as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(color.as_image_copy(), wgpu::ImageCopyBuffer {
            buffer: &buffer, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(stride), rows_per_image: Some(height) },
        }, wgpu::Extent3d { width, height, depth_or_array_layers: 1 });
        queue.submit(Some(enc.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let data = buffer.slice(..).get_mapped_range();
        let mut file = std::fs::File::create(path)?;
        use std::io::Write;
        for row in data.chunks_exact(stride as usize) { file.write_all(&row[..width as usize * 8])?; }
    }
    let stats = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        serde_json::json!({"median": values[values.len()/2], "p95": values[((values.len()-1) as f64 * 0.95).ceil() as usize]})
    };
    println!("{}", serde_json::json!({"preset": preset.name, "width": width, "height": height,
        "source": res.source_size,
        "frames": frames, "encoding": "HDR BT.2020 linear", "accum_ms": stats(accum),
        "tube_ms": stats(tube), "wall_ms": stats(wall)}));
    Ok(())
}

/// CPU stage durations include driver waits; these are not GPU execution times.
pub struct LiveProfile {
    frames: Vec<[f64; 5]>,
    previous: std::time::Instant,
}

impl LiveProfile {
    pub fn new() -> Option<Self> {
        (std::env::var("CRTULUM_PROFILE").as_deref() == Ok("1")).then(|| Self {
            frames: Vec::with_capacity(240), previous: std::time::Instant::now(),
        })
    }

    pub fn record(&mut self, start: std::time::Instant, accum: std::time::Instant,
        acquire: std::time::Instant, present: std::time::Instant) {
        let end = std::time::Instant::now();
        self.frames.push([(start - self.previous).as_secs_f64() * 1000.0,
            (accum - start).as_secs_f64() * 1000.0, (acquire - accum).as_secs_f64() * 1000.0,
            (present - acquire).as_secs_f64() * 1000.0, (end - present).as_secs_f64() * 1000.0]);
        self.previous = start;
        if self.frames.len() == 240 {
            let mut stats = serde_json::Map::new();
            for (i, name) in ["frame_interval", "simulation_cpu", "acquire", "draw_submit_present", "source_poll"].iter().enumerate() {
                let mut values: Vec<f64> = self.frames.iter().map(|v| v[i]).collect();
                values.sort_by(f64::total_cmp);
                stats.insert((*name).into(), serde_json::json!({"median_ms": values[120], "p95_ms": values[228]}));
            }
            eprintln!("[profile] {}", serde_json::Value::Object(stats));
            self.frames.clear();
        }
    }
}
