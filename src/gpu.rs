//! Physical Vulkan by default; an explicit CI build can test headless shaders on CPU.
use anyhow::{bail, Result};

pub fn instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    })
}

fn hardware_rank(kind: wgpu::DeviceType) -> Option<u8> {
    match kind {
        wgpu::DeviceType::DiscreteGpu => Some(0),
        wgpu::DeviceType::IntegratedGpu => Some(1),
        _ => None,
    }
}

fn adapter_rank(kind: wgpu::DeviceType, headless: bool) -> Option<u8> {
    hardware_rank(kind).or_else(|| {
        (cfg!(feature = "ci-software-vulkan") && headless && kind == wgpu::DeviceType::Cpu)
            .then_some(2)
    })
}

pub fn adapter(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<wgpu::Adapter> {
    let mut candidates = Vec::new();
    let mut found = Vec::new();
    for adapter in instance.enumerate_adapters(wgpu::Backends::VULKAN) {
        let info = adapter.get_info();
        let supported = surface.map_or(true, |s| adapter.is_surface_supported(s));
        found.push(format!("{} ({:?}{})", info.name, info.device_type,
            if supported { "" } else { ", cannot present to this window" }));
        if supported {
            if let Some(rank) = adapter_rank(info.device_type, surface.is_none()) {
                candidates.push((rank, adapter));
            }
        }
    }
    candidates.sort_by_key(|(rank, _)| *rank);
    if let Some((_, adapter)) = candidates.into_iter().next() {
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            eprintln!("[gpu] CI software-Vulkan build: numerical/export checks only; this does not verify physical GPU behavior");
        }
        eprintln!("[gpu] {} · Vulkan · {:?} · {} {}", info.name, info.device_type,
            info.driver, info.driver_info);
        return Ok(adapter);
    }
    bail!("a physical Vulkan GPU is required (discrete or integrated); software and virtual adapters are not supported. Detected: {}. Check the Vulkan driver and GPU device permissions; remove stale VK_DRIVER_FILES/VK_ICD_FILENAMES overrides if set",
        if found.is_empty() { "none".into() } else { found.join(", ") });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_devices_are_required_and_discrete_is_preferred() {
        use wgpu::DeviceType::*;
        assert!(hardware_rank(DiscreteGpu).unwrap() < hardware_rank(IntegratedGpu).unwrap());
        for kind in [Cpu, VirtualGpu, Other] {
            assert_eq!(hardware_rank(kind), None, "{kind:?} must not be a fallback");
        }
    }

    #[test]
    fn software_exception_is_build_time_and_headless_only() {
        use wgpu::DeviceType::*;
        assert_eq!(adapter_rank(Cpu, true), if cfg!(feature = "ci-software-vulkan") { Some(2) } else { None });
        assert_eq!(adapter_rank(Cpu, false), None);
        for kind in [VirtualGpu, Other] {
            assert_eq!(adapter_rank(kind, true), None);
        }
    }
}

/// History capacity scales with source resolution; request the hardware limits
/// for storage while retaining portable defaults for unrelated features.
pub fn limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let supported = adapter.limits();
    wgpu::Limits { max_storage_buffer_binding_size: supported.max_storage_buffer_binding_size,
        max_buffer_size: supported.max_buffer_size, ..wgpu::Limits::default() }
}
