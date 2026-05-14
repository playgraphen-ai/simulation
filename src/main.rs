mod sim;
mod render;
mod ui;
mod compute;
mod log_export;

use bevy::prelude::*;
use bevy::render::settings::{Backends, RenderCreation, WgpuSettings};
use bevy::render::render_resource::WgpuFeatures;
use bevy::render::RenderPlugin as BevyRenderPlugin;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin, EntityCountDiagnosticsPlugin};
use std::time::Instant;
use std::sync::Mutex;
use std::sync::mpsc::{Sender, Receiver};

#[derive(Clone, Copy, Default, Eq, PartialEq, Debug, Hash, States)]
pub enum AppState {
    MainMenu,
    #[default]
    InGame,
}

#[derive(Resource)]
pub struct TimingsSender(pub Mutex<Sender<(f32, f32)>>); // (compute_ms, render_ms)

#[derive(Resource)]
pub struct TimingsReceiver(pub Mutex<Receiver<(f32, f32)>>);

#[derive(Resource, Default, Clone)]
pub struct DetailedTimings {
    pub frame_start: Option<Instant>,
    pub update_start: Option<Instant>,
    
    pub last_update_ms: f32,
    pub last_compute_ms: f32,
    pub last_render_ms: f32,
    
    pub acc_update_ms: f32,
    pub acc_compute_ms: f32,
    pub acc_render_ms: f32,
    pub acc_frames: u32,
}

fn main() {
    // Backend: DX12 is the only one that works on this ARM64 + Adreno box.
    // Override via REAL_CITY_BACKEND=vulkan|gl|primary.
    let backends = match std::env::var("REAL_CITY_BACKEND").as_deref() {
        Ok("vulkan") => Backends::VULKAN,
        Ok("gl") => Backends::GL,
        Ok("primary") => Backends::PRIMARY,
        _ => Backends::DX12,
    };

    // The Adreno D3D12 driver mis-sizes Multi-Draw-Indirect buffers in the
    // shadow pass (see wgpu_core::command::render::multi_draw_indirect assert).
    // Force wgpu to negotiate the device *without* MDI so Bevy falls back to
    // regular draws. Belt and braces: we also pass `NoIndirectDrawing` on the
    // main camera in render/camera.rs, which only handles the main pass.
    let disabled = WgpuFeatures::MULTI_DRAW_INDIRECT_COUNT
        | WgpuFeatures::INDIRECT_FIRST_INSTANCE;

    let (tx, rx) = std::sync::mpsc::channel();

    let mut app = App::new();
    
    app.insert_resource(ClearColor(Color::srgb(0.15, 0.35, 0.15)))
        .init_resource::<DetailedTimings>()
        .insert_resource(TimingsReceiver(Mutex::new(rx)))
        .add_plugins(DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Real City".into(),
                    resolution: (1600u32, 900u32).into(),
                    ..default()
                }),
                ..default()
            })
            .set(BevyRenderPlugin {
                render_creation: RenderCreation::Automatic(WgpuSettings {
                    backends: Some(backends),
                    disabled_features: Some(disabled),
                    ..default()
                }),
                ..default()
            })
            // Hardcode asset path to the project so the binary works even when
            // launched from target/debug/ or another working directory.
            .set(AssetPlugin {
                file_path: concat!(env!("CARGO_MANIFEST_DIR"), "/assets").to_string(),
                ..default()
            }))
        .init_state::<AppState>()
        .add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            SystemInformationDiagnosticsPlugin::default(),
            EntityCountDiagnosticsPlugin::default(),
            sim::SimPlugin,
            render::WorldPlugin,
            ui::UiPlugin,
            compute::ComputePlugin,
            log_export::LogPlugin,
        ))
        .add_systems(First, start_timing)
        .add_systems(Update, receive_timings)
        .add_systems(Last, end_update_timing);

    if let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) {
        render_app.insert_resource(TimingsSender(Mutex::new(tx)));
        render_app.add_systems(bevy::render::ExtractSchedule, start_render_recording);
        render_app.add_systems(bevy::render::Render, end_render_recording.in_set(bevy::render::RenderSystems::Cleanup));
    }

    app.run();
}

#[derive(Resource, Default)]
struct RenderRecordingTimer(Option<Instant>);

fn start_render_recording(mut timer: Local<RenderRecordingTimer>) {
    timer.0 = Some(Instant::now());
}

fn end_render_recording(
    mut timer: Local<RenderRecordingTimer>,
    tx: Res<TimingsSender>,
) {
    if let Some(start) = timer.0.take() {
        if let Ok(tx) = tx.0.lock() {
            let _ = tx.send((0.0, start.elapsed().as_secs_f32() * 1000.0));
        }
    }
}

fn start_timing(mut timings: ResMut<DetailedTimings>) {
    let now = Instant::now();
    timings.frame_start = Some(now);
    timings.update_start = Some(now);
    // Reset frame-local timings
    timings.last_compute_ms = 0.0;
    timings.last_render_ms = 0.0;
}

fn receive_timings(
    receiver: Res<TimingsReceiver>,
    mut timings: ResMut<DetailedTimings>,
) {
    if let Ok(rx) = receiver.0.lock() {
        while let Ok((compute, render)) = rx.try_recv() {
            if compute > 0.0 {
                timings.last_compute_ms += compute;
            }
            if render > 0.0 {
                timings.last_render_ms += render;
            }
        }
        timings.acc_compute_ms += timings.last_compute_ms;
        timings.acc_render_ms += timings.last_render_ms;
    }
}

fn end_update_timing(mut timings: ResMut<DetailedTimings>) {
    if let Some(start) = timings.update_start {
        timings.last_update_ms = start.elapsed().as_secs_f32() * 1000.0;
        timings.acc_update_ms += timings.last_update_ms;
        timings.acc_frames += 1;
    }
}
