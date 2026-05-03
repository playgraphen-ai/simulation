//! Real City - Bevy city sim with GPU-backed simulation.
//!
//! Architecture: People / Roads / Buildings are stored in GPU storage textures
//! (Rgba32Float). Compute shaders simulate activity, pathfinding and road speeds.
//! CPU side owns authoring (UI tools, zoning, construction) and renders instanced
//! meshes from the same data.

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

#[derive(Clone, Copy, Default, Eq, PartialEq, Debug, Hash, States)]
pub enum AppState {
    MainMenu,
    #[default]
    InGame,
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

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.15, 0.35, 0.15)))
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
        ))
        .run();
}
