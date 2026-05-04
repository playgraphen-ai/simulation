//! Compute layer.
//!
//! Two-level design:
//!
//! 1. **CPU fallback ("fast path")**: a tight loop that reproduces exactly what
//!    the compute shaders will do — activity countdown, destination picking,
//!    path progression, segment speed update. This keeps the whole game
//!    playable while the GPU pipeline is wired up.
//! 2. **GPU compute path**: upload the three data textures, dispatch the
//!    `sim_people.wgsl` and `pathfind.wgsl` shaders, read counters back. The
//!    shaders are authored and present in assets/shaders/; binding them into
//!    Bevy's RenderApp pipeline cache is the remaining step.
//!
//! The CPU fallback is always run. When the GPU path is enabled, it will
//! replace the CPU body.

pub mod cpu_sim;
pub mod spawn;
pub mod gpu_sim;
pub mod gpu_pathfinding;

use bevy::prelude::*;

use crate::sim::textures::{create_data_textures, upload_dirty_textures, DataTextures};
use crate::sim::{
    buildings::BuildingData,
    people::PeopleData,
    roads::RoadData,
    ActivityDurations,
    SimSettings,
};
use crate::AppState;

pub struct ComputePlugin;

impl Plugin for ComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<cpu_sim::SimTiming>()
            .init_resource::<gpu_sim::GpuSimParams>()
            .init_resource::<gpu_sim::GpuSimTextures>()
            .add_plugins(gpu_sim::GpuSimPlugin)
            .add_plugins(gpu_pathfinding::GpuPathfindingPlugin)
            .add_systems(Startup, setup_data_textures)
            .add_systems(Update, (
                spawn::handle_spawn_requests,
                sync_gpu_textures_and_params,
                gpu_sim::clear_pending_spawns,
                gpu_sim::apply_gpu_readback,
                cpu_sim::cpu_sim_tick,
                upload_dirty_textures_system,
            ).chain().run_if(in_state(AppState::InGame)));
    }
}

fn setup_data_textures(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    people: Res<PeopleData>,
    roads: Res<RoadData>,
    buildings: Res<BuildingData>,
) {
    let dt = create_data_textures(&mut images, &people, &roads, &buildings);
    commands.insert_resource(dt);
}

fn sync_gpu_textures_and_params(
    dt: Option<Res<DataTextures>>,
    mut gpu_tex: ResMut<gpu_sim::GpuSimTextures>,
    time: Res<Time>,
    durations: Res<ActivityDurations>,
    settings: Res<SimSettings>,
    people: Res<PeopleData>,
    buildings: Res<BuildingData>,
    roads: Res<RoadData>,
    pending: ResMut<spawn::PendingGpuSpawns>,
    mut gpu_params: ResMut<gpu_sim::GpuSimParams>,
    mut path_params: ResMut<gpu_pathfinding::PathParams>,
) {
    if let Some(dt) = dt {
        gpu_tex.people = Some(dt.people.clone());
        gpu_tex.roads = Some(dt.roads.clone());
        gpu_tex.buildings = Some(dt.buildings.clone());
    }
    gpu_sim::update_gpu_sim_params(&time, &durations, &settings, &people, &buildings, &roads, pending, &mut gpu_params);
    path_params.roads_tex_w = roads.tex_width;
    path_params.segments_count = roads.segments.len() as u32;
    path_params.max_path_len = 256;
}

fn upload_dirty_textures_system(
    mut images: ResMut<Assets<Image>>,
    dt: Option<Res<DataTextures>>,
    mut people: ResMut<PeopleData>,
    mut roads: ResMut<RoadData>,
    mut buildings: ResMut<BuildingData>,
    grid: Res<crate::sim::grid::CityGrid>,
) {
    let Some(dt) = dt else { return; };
    upload_dirty_textures(&mut images, &dt, &mut people, &mut roads, &mut buildings, &grid);
}
