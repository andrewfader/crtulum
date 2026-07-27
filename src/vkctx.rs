// A headless Vulkan host for libretro cores.
//
// The OpenGL path in `glctx.rs` hands a core a context and an FBO and gets pixels
// back. Vulkan inverts that: the core wants *our* instance, physical device, logical
// device and queue, renders into an image it owns, and hands us back an image view
// through a callback interface. On top of that it may want a say in how the device is
// created at all — that's the "context negotiation" interface, which cores like
// paraLLEl use to turn on the features their renderer needs.
//
// So the shape here is:
//
//   1. create a VkInstance (using the core's VkApplicationInfo if it supplied one);
//   2. pick a physical device, then either let the core create the logical device
//      (negotiation) or create one ourselves;
//   3. publish `retro_hw_render_interface_vulkan` — handles plus eight callbacks the
//      core drives us through;
//   4. each frame the core calls `set_image()` with what it rendered, optionally
//      `set_command_buffers()` with work for us to submit, then reports the frame;
//   5. we submit anything pending, copy the image into a host-visible buffer, and
//      hand the pixels to the same phosphor pipeline as everything else.
//
// This is a separate device from crtulum's own wgpu one, deliberately: sharing wgpu's
// device would mean reaching into its internals and externally synchronising its
// queue, for a copy that's already negligible at 320x240 or 640x480.
//
// Struct layouts and callback signatures follow libretro_vulkan.h — the field order
// is ABI, so it's transcribed from the header rather than remembered.

use std::ffi::{c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use ash::vk;

pub const HW_RENDER_INTERFACE_VULKAN: u32 = 0;
pub const HW_RENDER_INTERFACE_VULKAN_VERSION: u32 = 5;
pub const NEGOTIATION_INTERFACE_VULKAN: u32 = 0;

// ---------------------------------------------------------------------------
// ABI — libretro_vulkan.h
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RetroVulkanImage {
    pub image_view: vk::ImageView,
    pub image_layout: vk::ImageLayout,
    pub create_info: vk::ImageViewCreateInfo,
}

#[repr(C)]
pub struct RetroVulkanContext {
    pub gpu: vk::PhysicalDevice,
    pub device: vk::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub presentation_queue: vk::Queue,
    pub presentation_queue_family_index: u32,
}

type CreateDeviceFn = unsafe extern "C" fn(
    context: *mut RetroVulkanContext,
    instance: vk::Instance,
    gpu: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    required_device_extensions: *mut *const std::ffi::c_char,
    num_required_device_extensions: u32,
    required_device_layers: *mut *const std::ffi::c_char,
    num_required_device_layers: u32,
    required_features: *const vk::PhysicalDeviceFeatures,
) -> bool;

/// The frontend-supplied hook a v2 core calls to actually create the device: the core
/// assembles the `VkDeviceCreateInfo` it needs (queues, extensions, features) and
/// hands it back to us to submit, which is how it gets a device with its requirements
/// on it without the frontend having to guess.
type CreateDeviceWrapperFn = unsafe extern "C" fn(
    gpu: vk::PhysicalDevice,
    opaque: *mut c_void,
    create_info: *const vk::DeviceCreateInfo,
) -> vk::Device;

type CreateDevice2Fn = unsafe extern "C" fn(
    context: *mut RetroVulkanContext,
    instance: vk::Instance,
    gpu: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    create_device_wrapper: CreateDeviceWrapperFn,
    opaque: *mut c_void,
) -> bool;

/// `struct retro_hw_render_context_negotiation_interface_vulkan`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NegotiationInterface {
    pub interface_type: u32,
    pub interface_version: u32,
    pub get_application_info: Option<unsafe extern "C" fn() -> *const vk::ApplicationInfo>,
    pub create_device: Option<CreateDeviceFn>,
    pub destroy_device: Option<unsafe extern "C" fn()>,
    /// Version 2. Lets the core build the instance (and thus pick its extensions).
    pub create_instance: Option<CreateInstanceFn>,
    /// Version 2. Modern cores (SwanStation, DuckStation, paraLLEl) implement this
    /// one and leave `create_device` null.
    pub create_device2: Option<CreateDevice2Fn>,
}

type CreateInstanceWrapperFn = unsafe extern "C" fn(
    opaque: *mut c_void,
    create_info: *const vk::InstanceCreateInfo,
) -> vk::Instance;

type CreateInstanceFn = unsafe extern "C" fn(
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    app: *const vk::ApplicationInfo,
    create_instance_wrapper: CreateInstanceWrapperFn,
    opaque: *mut c_void,
) -> vk::Instance;

/// What the wrappers need to do their job, passed through as `opaque`.
///
/// `create_device` isn't knowable until the instance exists, so both are optional —
/// a Vulkan `PFN_` is a non-nullable function pointer and must never be conjured
/// from a null.
struct WrapperCtx {
    create_instance: Option<vk::PFN_vkCreateInstance>,
    create_device: Option<vk::PFN_vkCreateDevice>,
}

unsafe extern "C" fn create_instance_wrapper(
    opaque: *mut c_void,
    create_info: *const vk::InstanceCreateInfo,
) -> vk::Instance {
    if opaque.is_null() || create_info.is_null() {
        return vk::Instance::null();
    }
    let ctx = &*(opaque as *const WrapperCtx);
    let Some(create) = ctx.create_instance else { return vk::Instance::null() };
    let mut instance = vk::Instance::null();
    // Straight through: the instance extensions the core asks for here are the ones
    // its device path will go on to require.
    if create(create_info, std::ptr::null(), &mut instance) != vk::Result::SUCCESS {
        return vk::Instance::null();
    }
    instance
}

unsafe extern "C" fn create_device_wrapper(
    gpu: vk::PhysicalDevice,
    opaque: *mut c_void,
    create_info: *const vk::DeviceCreateInfo,
) -> vk::Device {
    if opaque.is_null() || create_info.is_null() {
        return vk::Device::null();
    }
    let ctx = &*(opaque as *const WrapperCtx);
    let Some(create) = ctx.create_device else { return vk::Device::null() };
    let mut device = vk::Device::null();
    // Pass the core's request through untouched — it knows what its renderer needs.
    if create(gpu, create_info, std::ptr::null(), &mut device) != vk::Result::SUCCESS {
        return vk::Device::null();
    }
    device
}

type SetImageFn = unsafe extern "C" fn(
    handle: *mut c_void,
    image: *const RetroVulkanImage,
    num_semaphores: u32,
    semaphores: *const vk::Semaphore,
    src_queue_family: u32,
);
type GetSyncIndexFn = unsafe extern "C" fn(handle: *mut c_void) -> u32;
type SetCommandBuffersFn =
    unsafe extern "C" fn(handle: *mut c_void, num_cmd: u32, cmd: *const vk::CommandBuffer);
type QueueFn = unsafe extern "C" fn(handle: *mut c_void);
type SetSignalSemaphoreFn = unsafe extern "C" fn(handle: *mut c_void, semaphore: vk::Semaphore);

/// `struct retro_hw_render_interface_vulkan` — what the core reads to drive us.
#[repr(C)]
pub struct HwRenderInterfaceVulkan {
    pub interface_type: u32,
    pub interface_version: u32,
    pub handle: *mut c_void,
    pub instance: vk::Instance,
    pub gpu: vk::PhysicalDevice,
    pub device: vk::Device,
    pub get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    pub get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    pub queue: vk::Queue,
    pub queue_index: u32,
    pub set_image: SetImageFn,
    pub get_sync_index: GetSyncIndexFn,
    pub get_sync_index_mask: GetSyncIndexFn,
    pub set_command_buffers: SetCommandBuffersFn,
    pub wait_sync_index: QueueFn,
    pub lock_queue: QueueFn,
    pub unlock_queue: QueueFn,
    pub set_signal_semaphore: SetSignalSemaphoreFn,
}

// ---------------------------------------------------------------------------
// The state the core's callbacks touch
// ---------------------------------------------------------------------------

/// A lock that can be taken and released from different FFI calls (and different
/// threads — paraLLEl runs its RDP on its own). A `MutexGuard` can't span two
/// callbacks, so this is the raw form. Hold times are one queue submit.
struct RawLock(AtomicBool);

impl RawLock {
    const fn new() -> RawLock {
        RawLock(AtomicBool::new(false))
    }
    fn lock(&self) {
        let mut spins = 0u32;
        while self
            .0
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            spins += 1;
            if spins < 64 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
    fn unlock(&self) {
        self.0.store(false, Ordering::Release);
    }
}

#[derive(Default)]
struct Pending {
    image: Option<RetroVulkanImage>,
    semaphores: Vec<vk::Semaphore>,
    src_queue_family: u32,
    cmd: Vec<vk::CommandBuffer>,
    signal: vk::Semaphore,
}

/// Handed to the core as the opaque `handle`; every callback casts it back.
pub struct Shared {
    queue_lock: RawLock,
    sync_index: AtomicU32,
    pending: Mutex<Pending>,
}

const SYNC_SLOTS: u32 = 4;

unsafe extern "C" fn cb_set_image(
    handle: *mut c_void,
    image: *const RetroVulkanImage,
    num_semaphores: u32,
    semaphores: *const vk::Semaphore,
    src_queue_family: u32,
) {
    let shared = &*(handle as *const Shared);
    let mut p = shared.pending.lock().unwrap();
    p.image = if image.is_null() { None } else { Some(*image) };
    p.semaphores.clear();
    if !semaphores.is_null() && num_semaphores > 0 {
        p.semaphores
            .extend_from_slice(std::slice::from_raw_parts(semaphores, num_semaphores as usize));
    }
    p.src_queue_family = src_queue_family;
}

unsafe extern "C" fn cb_get_sync_index(handle: *mut c_void) -> u32 {
    let shared = &*(handle as *const Shared);
    shared.sync_index.load(Ordering::Relaxed)
}

unsafe extern "C" fn cb_get_sync_index_mask(_handle: *mut c_void) -> u32 {
    (1 << SYNC_SLOTS) - 1
}

unsafe extern "C" fn cb_set_command_buffers(
    handle: *mut c_void,
    num_cmd: u32,
    cmd: *const vk::CommandBuffer,
) {
    let shared = &*(handle as *const Shared);
    let mut p = shared.pending.lock().unwrap();
    p.cmd.clear();
    if !cmd.is_null() && num_cmd > 0 {
        p.cmd.extend_from_slice(std::slice::from_raw_parts(cmd, num_cmd as usize));
    }
}

unsafe extern "C" fn cb_wait_sync_index(_handle: *mut c_void) {
    // We fence-wait on every frame before touching the image, so by the time the core
    // asks, the slot it's about to reuse is already idle.
}

unsafe extern "C" fn cb_lock_queue(handle: *mut c_void) {
    (&*(handle as *const Shared)).queue_lock.lock();
}

unsafe extern "C" fn cb_unlock_queue(handle: *mut c_void) {
    (&*(handle as *const Shared)).queue_lock.unlock();
}

unsafe extern "C" fn cb_set_signal_semaphore(handle: *mut c_void, semaphore: vk::Semaphore) {
    let shared = &*(handle as *const Shared);
    shared.pending.lock().unwrap().signal = semaphore;
}

// ---------------------------------------------------------------------------
// The host
// ---------------------------------------------------------------------------

pub struct VkHost {
    // Drop order matters: everything below is destroyed before the device/instance.
    readback: vk::Buffer,
    readback_mem: vk::DeviceMemory,
    readback_size: u64,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    iface: Box<HwRenderInterfaceVulkan>,
    shared: Box<Shared>,
    queue: vk::Queue,
    queue_family: u32,
    device: ash::Device,
    /// Set when the core created the device, in which case it destroys it too.
    core_destroy_device: Option<unsafe extern "C" fn()>,
    gpu: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_fns: vk::KhrSurfaceFn,
    /// Kept alive because the core holds the pointer for as long as the device lives.
    _wrapper_ctx: Box<WrapperCtx>,
    instance: ash::Instance,
    _entry: ash::Entry,
    scratch: Vec<u8>,
}

impl VkHost {
    /// `negotiation` is the core's context-negotiation interface, if it published one.
    pub fn new(negotiation: Option<&NegotiationInterface>) -> Result<VkHost> {
        let entry = unsafe { ash::Entry::load() }
            .context("no Vulkan loader — libvulkan.so.1 has to be installed for a Vulkan core")?;

        if let Some(n) = negotiation {
            eprintln!(
                "[emu] core negotiation interface: type={} version={} get_application_info={} \
                 create_device={} destroy_device={} create_instance={} create_device2={}",
                n.interface_type,
                n.interface_version,
                n.get_application_info.is_some(),
                n.create_device.is_some(),
                n.destroy_device.is_some(),
                n.create_instance.is_some(),
                n.create_device2.is_some(),
            );
        } else {
            eprintln!("[emu] core published no Vulkan negotiation interface");
        }

        // The core may specify the API version it needs through its application info.
        let app_info_from_core = negotiation
            .and_then(|n| n.get_application_info)
            .map(|f| unsafe { f() })
            .filter(|p| !p.is_null());
        let api_version = match app_info_from_core {
            Some(p) => unsafe { (*p).api_version }.max(vk::API_VERSION_1_1),
            None => vk::API_VERSION_1_1,
        };

        let app_name = CString::new("crtulum").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .engine_name(&app_name)
            .api_version(api_version);

        let mut wrapper_ctx = Box::new(WrapperCtx {
            create_instance: Some(entry.fp_v1_0().create_instance),
            create_device: None, // not resolvable until the instance exists
        });

        // A v2 core would rather build the instance itself, because the extensions it
        // enables here are the ones its device creation then depends on. Let it.
        let negotiated_instance = negotiation
            .filter(|n| n.interface_version >= 2)
            .and_then(|n| n.create_instance)
            .map(|create| unsafe {
                create(
                    entry.static_fn().get_instance_proc_addr,
                    app_info_from_core.unwrap_or(&*app_info as *const vk::ApplicationInfo),
                    create_instance_wrapper,
                    &mut *wrapper_ctx as *mut WrapperCtx as *mut c_void,
                )
            })
            .filter(|i| *i != vk::Instance::null());

        let instance = match negotiated_instance {
            Some(handle) => unsafe { ash::Instance::load(entry.static_fn(), handle) },
            None => {
                // VK_EXT_headless_surface is why this works without a window. Cores
                // negotiate their device by looking for a queue family that can
                // present to a surface — with no surface at all that search fails,
                // and a core will happily record the "not found" index and trip over
                // it later. A headless surface makes that search behave exactly as it
                // does under a window manager.
                let available = entry
                    .enumerate_instance_extension_properties(None)
                    .unwrap_or_default();
                let has = |want: &CStr| {
                    available.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == want)
                };
                let mut names: Vec<*const std::ffi::c_char> = Vec::new();
                if has(vk::KhrSurfaceFn::name()) && has(vk::ExtHeadlessSurfaceFn::name()) {
                    names.push(vk::KhrSurfaceFn::name().as_ptr());
                    names.push(vk::ExtHeadlessSurfaceFn::name().as_ptr());
                }
                unsafe {
                    entry.create_instance(
                        &vk::InstanceCreateInfo::builder()
                            .application_info(&app_info)
                            .enabled_extension_names(&names),
                        None,
                    )
                }
                .context("creating the Vulkan instance for the core")?
            }
        };
        let instance_from_core = negotiated_instance.is_some();
        wrapper_ctx.create_device = Some(instance.fp_v1_0().create_device);

        // Prefer a real GPU over a software rasteriser.
        let gpus = unsafe { instance.enumerate_physical_devices() }?;
        let pick = |want: vk::PhysicalDeviceType| {
            gpus.iter().copied().find(|g| {
                unsafe { instance.get_physical_device_properties(*g) }.device_type == want
            })
        };
        let gpu = pick(vk::PhysicalDeviceType::DISCRETE_GPU)
            .or_else(|| pick(vk::PhysicalDeviceType::INTEGRATED_GPU))
            .or_else(|| gpus.first().copied())
            .ok_or_else(|| anyhow!("no Vulkan physical device"))?;
        let props = unsafe { instance.get_physical_device_properties(gpu) };
        let gpu_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        // The windowless surface itself, if the extension made it in.
        let surface_fns = vk::KhrSurfaceFn::load(|name| unsafe {
            std::mem::transmute(entry.get_instance_proc_addr(instance.handle(), name.as_ptr()))
        });
        let headless_fns = vk::ExtHeadlessSurfaceFn::load(|name| unsafe {
            std::mem::transmute(entry.get_instance_proc_addr(instance.handle(), name.as_ptr()))
        });
        let mut surface = vk::SurfaceKHR::null();
        if negotiated_instance.is_none() {
            let info = vk::HeadlessSurfaceCreateInfoEXT::default();
            let r = unsafe {
                (headless_fns.create_headless_surface_ext)(
                    instance.handle(),
                    &info,
                    std::ptr::null(),
                    &mut surface,
                )
            };
            if r != vk::Result::SUCCESS {
                surface = vk::SurfaceKHR::null();
            }
        }

        // A queue family that can do everything a renderer needs.
        let families = unsafe { instance.get_physical_device_queue_family_properties(gpu) };
        let queue_family = families
            .iter()
            .position(|f| {
                f.queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER)
            })
            .ok_or_else(|| anyhow!("no graphics+compute queue family"))? as u32;

        // Either the core builds the logical device (so it can enable the features and
        // extensions its renderer depends on) or we do.
        let mut core_destroy_device = None;
        let mut negotiated_gpu = gpu;
        let mut device_from_core = true;

        // Version 2 first: it's what current cores implement, and a core that offers
        // it usually leaves the version 1 entry point null.
        let v2 = negotiation.filter(|n| n.interface_version >= 2).and_then(|n| n.create_device2);
        let (device, queue, queue_family) = if let Some(create2) = v2 {
            let mut ctx = RetroVulkanContext {
                gpu: vk::PhysicalDevice::null(),
                device: vk::Device::null(),
                queue: vk::Queue::null(),
                queue_family_index: 0,
                presentation_queue: vk::Queue::null(),
                presentation_queue_family_index: 0,
            };
            let ok = unsafe {
                create2(
                    &mut ctx,
                    instance.handle(),
                    gpu,
                    surface,
                    entry.static_fn().get_instance_proc_addr,
                    create_device_wrapper,
                    &mut *wrapper_ctx as *mut WrapperCtx as *mut c_void,
                )
            };
            if !ok || ctx.device == vk::Device::null() {
                bail!("the core declined to create a Vulkan device (negotiation v2)");
            }
            if ctx.gpu != vk::PhysicalDevice::null() {
                negotiated_gpu = ctx.gpu;
            }
            core_destroy_device = negotiation.and_then(|n| n.destroy_device);
            let device = unsafe { ash::Device::load(instance.fp_v1_0(), ctx.device) };
            let queue = if ctx.queue != vk::Queue::null() {
                ctx.queue
            } else {
                unsafe { device.get_device_queue(ctx.queue_family_index, 0) }
            };
            (device, queue, ctx.queue_family_index)
        } else {
        match negotiation.and_then(|n| n.create_device) {
            Some(create) => {
                let mut ctx = RetroVulkanContext {
                    gpu: vk::PhysicalDevice::null(),
                    device: vk::Device::null(),
                    queue: vk::Queue::null(),
                    queue_family_index: 0,
                    presentation_queue: vk::Queue::null(),
                    presentation_queue_family_index: 0,
                };
                let ok = unsafe {
                    create(
                        &mut ctx,
                        instance.handle(),
                        gpu,
                        surface,
                        entry.static_fn().get_instance_proc_addr,
                        std::ptr::null_mut(),
                        0,
                        std::ptr::null_mut(),
                        0,
                        std::ptr::null(),
                    )
                };
                if !ok || ctx.device == vk::Device::null() {
                    bail!("the core declined to create a Vulkan device");
                }
                core_destroy_device = negotiation.and_then(|n| n.destroy_device);
                let device = unsafe { ash::Device::load(instance.fp_v1_0(), ctx.device) };
                let queue = if ctx.queue != vk::Queue::null() {
                    ctx.queue
                } else {
                    unsafe { device.get_device_queue(ctx.queue_family_index, 0) }
                };
                (device, queue, ctx.queue_family_index)
            }
            None => {
                device_from_core = false;
                let priorities = [1.0f32];
                let queue_info = vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(queue_family)
                    .queue_priorities(&priorities);
                let infos = [queue_info.build()];
                // Enable everything the GPU reports. A core that didn't negotiate has
                // no way to ask for what it needs, so it assumes the frontend built a
                // capable device — and quietly does undefined things with any feature
                // that turns out to be off. This is what RetroArch does too.
                let features = unsafe { instance.get_physical_device_features(gpu) };
                let device = unsafe {
                    instance.create_device(
                        gpu,
                        &vk::DeviceCreateInfo::builder()
                            .queue_create_infos(&infos)
                            .enabled_features(&features),
                        None,
                    )
                }
                .context("creating the Vulkan device")?;
                let queue = unsafe { device.get_device_queue(queue_family, 0) };
                (device, queue, queue_family)
            }
        }
        };
        let gpu = negotiated_gpu;

        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }?;
        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;

        let shared = Box::new(Shared {
            queue_lock: RawLock::new(),
            sync_index: AtomicU32::new(0),
            pending: Mutex::new(Pending::default()),
        });

        let iface = Box::new(HwRenderInterfaceVulkan {
            interface_type: HW_RENDER_INTERFACE_VULKAN,
            interface_version: HW_RENDER_INTERFACE_VULKAN_VERSION,
            handle: &*shared as *const Shared as *mut c_void,
            instance: instance.handle(),
            gpu,
            device: device.handle(),
            get_device_proc_addr: instance.fp_v1_0().get_device_proc_addr,
            get_instance_proc_addr: entry.static_fn().get_instance_proc_addr,
            queue,
            queue_index: queue_family,
            set_image: cb_set_image,
            get_sync_index: cb_get_sync_index,
            get_sync_index_mask: cb_get_sync_index_mask,
            set_command_buffers: cb_set_command_buffers,
            wait_sync_index: cb_wait_sync_index,
            lock_queue: cb_lock_queue,
            unlock_queue: cb_unlock_queue,
            set_signal_semaphore: cb_set_signal_semaphore,
        });

        eprintln!(
            "[emu] hardware rendering: Vulkan {}.{} · {gpu_name} · {}",
            vk::api_version_major(api_version),
            vk::api_version_minor(api_version),
            match (instance_from_core, device_from_core) {
                (true, _) => "instance and device negotiated by the core",
                (false, true) => "device negotiated by the core",
                (false, false) => "instance and device created by crtulum",
            }
        );

        Ok(VkHost {
            readback: vk::Buffer::null(),
            readback_mem: vk::DeviceMemory::null(),
            readback_size: 0,
            cmd_pool,
            cmd,
            fence,
            iface,
            shared,
            queue,
            queue_family,
            device,
            core_destroy_device,
            gpu,
            surface,
            surface_fns,
            _wrapper_ctx: wrapper_ctx,
            instance,
            _entry: entry,
            scratch: Vec::new(),
        })
    }

    /// The pointer handed back for `GET_HW_RENDER_INTERFACE`.
    pub fn interface_ptr(&self) -> *const c_void {
        &*self.iface as *const HwRenderInterfaceVulkan as *const c_void
    }

    /// True once the core has actually given us something to show.
    pub fn has_image(&self) -> bool {
        self.shared.pending.lock().unwrap().image.is_some()
    }

    fn ensure_readback(&mut self, bytes: u64) -> Result<()> {
        if self.readback_size >= bytes && self.readback != vk::Buffer::null() {
            return Ok(());
        }
        unsafe {
            if self.readback != vk::Buffer::null() {
                self.device.destroy_buffer(self.readback, None);
                self.device.free_memory(self.readback_mem, None);
            }
            let buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(bytes)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let mem_props = self.instance.get_physical_device_memory_properties(self.gpu);
            let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let type_index = (0..mem_props.memory_type_count)
                .find(|i| {
                    req.memory_type_bits & (1 << i) != 0
                        && mem_props.memory_types[*i as usize].property_flags.contains(want)
                })
                .ok_or_else(|| anyhow!("no host-visible memory type for readback"))?;
            let mem = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::builder()
                    .allocation_size(req.size)
                    .memory_type_index(type_index),
                None,
            )?;
            self.device.bind_buffer_memory(buffer, mem, 0)?;
            self.readback = buffer;
            self.readback_mem = mem;
            self.readback_size = bytes;
        }
        Ok(())
    }

    /// Submit whatever the core left pending, copy its image out, and return it as
    /// RGBA8. Called once per frame, right after `retro_run`.
    pub fn read_frame(&mut self, width: u32, height: u32, out: &mut Vec<u8>) -> Result<()> {
        let (image, semaphores, cmd_buffers, signal, layout) = {
            let mut p = self.shared.pending.lock().unwrap();
            let img = p.image.ok_or_else(|| anyhow!("the core reported a GPU frame but never called set_image"))?;
            let sems = std::mem::take(&mut p.semaphores);
            let cmds = std::mem::take(&mut p.cmd);
            let signal = std::mem::replace(&mut p.signal, vk::Semaphore::null());
            (img, sems, cmds, signal, img.image_layout)
        };
        let vk_image = image.create_info.image;
        if vk_image == vk::Image::null() {
            bail!("the core's image view has no image attached");
        }

        let (w, h) = (width.max(1), height.max(1));
        let bytes = (w as u64) * (h as u64) * 4;
        self.ensure_readback(bytes)?;

        unsafe {
            self.device.reset_fences(&[self.fence])?;
            self.device
                .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            // Take the image from whatever layout the core left it in, copy it, and
            // put it back so the core's own tracking stays true.
            let to_src = vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(layout)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(vk_image)
                .subresource_range(range);
            self.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_src.build()],
            );

            let region = vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(0) // tightly packed
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
            self.device.cmd_copy_image_to_buffer(
                self.cmd,
                vk_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback,
                &[region.build()],
            );

            let back = vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(vk_image)
                .subresource_range(range);
            self.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[back.build()],
            );
            self.device.end_command_buffer(self.cmd)?;

            // The core's own command buffers (if it handed us any) go first, then our
            // copy — one submit, so ordering is guaranteed.
            let mut all: Vec<vk::CommandBuffer> = cmd_buffers;
            all.push(self.cmd);
            let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; semaphores.len()];
            let mut submit = vk::SubmitInfo::builder()
                .command_buffers(&all)
                .wait_semaphores(&semaphores)
                .wait_dst_stage_mask(&wait_stages);
            let signal_arr = [signal];
            if signal != vk::Semaphore::null() {
                submit = submit.signal_semaphores(&signal_arr);
            }

            // The core may be submitting from its own thread; take the same lock it
            // does before touching the queue.
            self.shared.queue_lock.lock();
            let submitted = self.device.queue_submit(self.queue, &[submit.build()], self.fence);
            self.shared.queue_lock.unlock();
            submitted?;

            self.device.wait_for_fences(&[self.fence], true, 5_000_000_000)?;

            let ptr = self.device.map_memory(
                self.readback_mem,
                0,
                bytes,
                vk::MemoryMapFlags::empty(),
            )? as *const u8;
            self.scratch.resize(bytes as usize, 0);
            std::ptr::copy_nonoverlapping(ptr, self.scratch.as_mut_ptr(), bytes as usize);
            self.device.unmap_memory(self.readback_mem);
        }

        // Vulkan images are usually B8G8R8A8; the phosphor pipeline wants RGBA8.
        out.resize(self.scratch.len(), 0);
        let swap = matches!(
            image.create_info.format,
            vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_SNORM
        );
        for (o, i) in out.chunks_exact_mut(4).zip(self.scratch.chunks_exact(4)) {
            if swap {
                o[0] = i[2];
                o[1] = i[1];
                o[2] = i[0];
            } else {
                o[0] = i[0];
                o[1] = i[1];
                o[2] = i[2];
            }
            o[3] = 255;
        }

        let next = (self.shared.sync_index.load(Ordering::Relaxed) + 1) % SYNC_SLOTS;
        self.shared.sync_index.store(next, Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for VkHost {
    fn drop(&mut self) {
        let trace = std::env::var_os("CRTULUM_TRACE_TEARDOWN").is_some();
        if trace { eprintln!("[teardown] vk: wait_idle"); }
        unsafe {
            let _ = self.device.device_wait_idle();
            if trace { eprintln!("[teardown] vk: destroy scratch objects"); }
            if self.readback != vk::Buffer::null() {
                self.device.destroy_buffer(self.readback, None);
                self.device.free_memory(self.readback_mem, None);
            }
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            if trace { eprintln!("[teardown] vk: destroy device"); }
            match self.core_destroy_device {
                // The core made the device, so the core takes it down.
                Some(destroy) => destroy(),
                None => self.device.destroy_device(None),
            }
            if self.surface != vk::SurfaceKHR::null() {
                (self.surface_fns.destroy_surface_khr)(self.instance.handle(), self.surface, std::ptr::null());
            }
            if trace { eprintln!("[teardown] vk: destroy instance"); }
            self.instance.destroy_instance(None);
            if trace { eprintln!("[teardown] vk: done"); }
        }
    }
}
