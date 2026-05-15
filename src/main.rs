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

#[derive(Clone, Copy, Default, Debug)]
pub struct TimingEvent {
    pub total_compute: f32,
    pub logic: f32,
    pub bldg: f32,
    pub road: f32,
    pub pathfind: f32,
    pub recount: f32,
    pub render: f32,
    
    // Cycle info
    pub recount_cycle: Option<(u32, u32)>,
    pub logic_cycle: Option<(u32, u32)>,
    pub bldg_cycle: Option<(u32, u32)>,
    pub road_cycle: Option<(u32, u32)>,
    
    // Readback latencies
    pub rb_stats_ms: Option<f32>,
    pub rb_person_ms: Option<f32>,
    pub rb_bldg_ms: Option<f32>,

    // Counts for throughput
    pub logic_count: Option<u32>,
    pub path_count: Option<u32>,

    // Occupancy capacities and counts
    pub occ_people: Option<(u32, u32)>,
    pub occ_bldgs: Option<(u32, u32)>,
    pub occ_segments: Option<(u32, u32)>,
}

#[derive(Resource)]
pub struct TimingsSender(pub Mutex<Sender<TimingEvent>>);

#[derive(Resource)]
pub struct TimingsReceiver(pub Mutex<Receiver<TimingEvent>>);

#[derive(Resource, Default, Clone)]
pub struct DetailedTimings {
    pub frame_start: Option<Instant>,
    pub update_start: Option<Instant>,
    
    pub last_update_ms: f32,
    pub last_compute_ms: f32,
    pub last_render_ms: f32,
    
    pub last_logic_ms: f32,
    pub last_bldg_ms: f32,
    pub last_road_ms: f32,
    pub last_pathfind_ms: f32,
    pub last_recount_ms: f32,
    
    // Smoothed values for UI
    pub smooth_update_ms: f32,
    pub smooth_compute_ms: f32,
    pub smooth_render_ms: f32,
    
    pub smooth_logic_ms: f32,
    pub smooth_bldg_ms: f32,
    pub smooth_road_ms: f32,
    pub smooth_pathfind_ms: f32,
    pub smooth_recount_ms: f32,

    pub peak_logic_ms: f32,
    pub peak_bldg_ms: f32,
    pub peak_road_ms: f32,
    pub peak_pathfind_ms: f32,
    pub peak_recount_ms: f32,
    
    // Cycle progress
    pub recount_cycle: Option<(u32, u32)>,
    pub logic_cycle: (u32, u32),
    pub bldg_cycle: (u32, u32),
    pub road_cycle: (u32, u32),

    // Data Readback latencies
    pub rb_stats_ms: f32,
    pub rb_person_ms: f32,
    pub rb_bldg_ms: f32,

    // Throughput (items / ms)
    pub throughput_logic: f32,
    pub throughput_path: f32,

    // Buffer Occupancy %
    pub occ_people: f32,
    pub occ_buildings: f32,
    pub occ_segments: f32,

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
        render_app.init_resource::<RenderRecordingTimer>();
        render_app.add_systems(bevy::render::ExtractSchedule, start_render_recording);
        render_app.add_systems(bevy::render::Render, end_render_recording.in_set(bevy::render::RenderSystems::Cleanup));
    }

    app.run();
}

#[derive(Resource, Default)]
struct RenderRecordingTimer(Option<Instant>);

fn start_render_recording(mut timer: ResMut<RenderRecordingTimer>) {
    timer.0 = Some(Instant::now());
}

fn end_render_recording(
    mut timer: ResMut<RenderRecordingTimer>,
    tx: Res<TimingsSender>,
) {
    if let Some(start) = timer.0.take() {
        if let Ok(tx) = tx.0.lock() {
            let mut event = TimingEvent::default();
            event.render = start.elapsed().as_secs_f32() * 1000.0;
            let _ = tx.send(event);
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
    timings.last_logic_ms = 0.0;
    timings.last_bldg_ms = 0.0;
    timings.last_road_ms = 0.0;
    timings.last_pathfind_ms = 0.0;
    timings.last_recount_ms = 0.0;
}

fn receive_timings(
    receiver: Res<TimingsReceiver>,
    mut timings: ResMut<DetailedTimings>,
) {
    if let Ok(rx) = receiver.0.lock() {
        while let Ok(event) = rx.try_recv() {
            if event.total_compute > 0.0 { timings.last_compute_ms += event.total_compute; }
            if event.logic > 0.0 { timings.last_logic_ms += event.logic; }
            if event.bldg > 0.0 { timings.last_bldg_ms += event.bldg; }
            if event.road > 0.0 { timings.last_road_ms += event.road; }
            if event.pathfind > 0.0 { timings.last_pathfind_ms += event.pathfind; }
            if event.recount > 0.0 { timings.last_recount_ms += event.recount; }
            if event.render > 0.0 { timings.last_render_ms += event.render; }

            if let Some(cycle) = event.recount_cycle { timings.recount_cycle = Some(cycle); }
            if let Some(cycle) = event.logic_cycle { timings.logic_cycle = cycle; }
            if let Some(cycle) = event.bldg_cycle { timings.bldg_cycle = cycle; }
            if let Some(cycle) = event.road_cycle { timings.road_cycle = cycle; }

            if let Some(ms) = event.rb_stats_ms { timings.rb_stats_ms = ms; }
            if let Some(ms) = event.rb_person_ms { timings.rb_person_ms = ms; }
            if let Some(ms) = event.rb_bldg_ms { timings.rb_bldg_ms = ms; }

            if let Some(count) = event.logic_count { 
                if event.logic > 0.0 {
                    let alpha = 0.1;
                    let throughput = count as f32 / event.logic;
                    timings.throughput_logic = timings.throughput_logic * (1.0 - alpha) + throughput * alpha;
                }
            }
            if let Some(count) = event.path_count {
                if event.pathfind > 0.0 {
                    let alpha = 0.1;
                    let throughput = count as f32 / event.pathfind;
                    timings.throughput_path = timings.throughput_path * (1.0 - alpha) + throughput * alpha;
                }
            }

            if let Some((occ, cap)) = event.occ_people {
                if cap > 0 { timings.occ_people = (occ as f32 / cap as f32) * 100.0; }
            }
            if let Some((occ, cap)) = event.occ_bldgs {
                if cap > 0 { timings.occ_buildings = (occ as f32 / cap as f32) * 100.0; }
            }
            if let Some((occ, cap)) = event.occ_segments {
                if cap > 0 { timings.occ_segments = (occ as f32 / cap as f32) * 100.0; }
            }
        }
        
        // Apply smoothing (EWMA)
        let alpha = 0.1;
        // Only smooth if there's actually a non-zero value this frame, 
        // to avoid decaying to 0 during frames where a slice doesn't run.
        if timings.last_compute_ms > 0.0 {
            timings.smooth_compute_ms = timings.smooth_compute_ms * (1.0 - alpha) + timings.last_compute_ms * alpha;
        }
        if timings.last_logic_ms > 0.0 {
            timings.smooth_logic_ms = timings.smooth_logic_ms * (1.0 - alpha) + timings.last_logic_ms * alpha;
            if timings.last_logic_ms > timings.peak_logic_ms {
                timings.peak_logic_ms = timings.last_logic_ms;
            }
        }
        if timings.last_bldg_ms > 0.0 {
            timings.smooth_bldg_ms = timings.smooth_bldg_ms * (1.0 - alpha) + timings.last_bldg_ms * alpha;
            if timings.last_bldg_ms > timings.peak_bldg_ms {
                timings.peak_bldg_ms = timings.last_bldg_ms;
            }
        }
        if timings.last_road_ms > 0.0 {
            timings.smooth_road_ms = timings.smooth_road_ms * (1.0 - alpha) + timings.last_road_ms * alpha;
            if timings.last_road_ms > timings.peak_road_ms {
                timings.peak_road_ms = timings.last_road_ms;
            }
        }
        if timings.last_pathfind_ms > 0.0 {
            timings.smooth_pathfind_ms = timings.smooth_pathfind_ms * (1.0 - alpha) + timings.last_pathfind_ms * alpha;
            if timings.last_pathfind_ms > timings.peak_pathfind_ms {
                timings.peak_pathfind_ms = timings.last_pathfind_ms;
            }
        }
        if timings.last_recount_ms > 0.0 {
            timings.smooth_recount_ms = timings.smooth_recount_ms * (1.0 - alpha) + timings.last_recount_ms * alpha;
            if timings.last_recount_ms > timings.peak_recount_ms {
                timings.peak_recount_ms = timings.last_recount_ms;
            }
        }
        
        timings.smooth_render_ms = timings.smooth_render_ms * (1.0 - alpha) + timings.last_render_ms * alpha;

        timings.acc_compute_ms += timings.last_compute_ms;
        timings.acc_render_ms += timings.last_render_ms;
    }
}

fn end_update_timing(mut timings: ResMut<DetailedTimings>) {
    if let Some(start) = timings.update_start {
        timings.last_update_ms = start.elapsed().as_secs_f32() * 1000.0;
        
        let alpha = 0.1;
        timings.smooth_update_ms = timings.smooth_update_ms * (1.0 - alpha) + timings.last_update_ms * alpha;
        
        timings.acc_update_ms += timings.last_update_ms;
        timings.acc_frames += 1;
    }
}
