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
use serde::{Deserialize, Serialize};

use crate::sim::textures::{create_data_textures, upload_dirty_textures, DataTextures};
use crate::sim::{
    buildings::BuildingData,
    people::PeopleData,
    roads::RoadData,
    ActivityDurations,
    SimSettings,
};
use crate::AppState;

#[derive(Serialize, Deserialize, Clone, Resource)]
pub struct ScheduleConfig {
    pub buildings_frames: u32,
    pub people_logic_frames: u32,
    pub roads_frames: u32,
    pub pathfind_frames: u32,
    pub stats_frames: u32,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self { buildings_frames: 10, people_logic_frames: 10, roads_frames: 10, pathfind_frames: 59, stats_frames: 1 }
    }
}

#[derive(Resource)]
pub struct SimScheduleState {
    pub config: ScheduleConfig,
    pub current_frame: u32,
    pub cycle_frames: u32,
}

impl Default for SimScheduleState {
    fn default() -> Self {
        let config: ScheduleConfig = std::fs::read_to_string("assets/sim_schedule.json")
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let cycle_frames = config.buildings_frames + config.people_logic_frames + config.roads_frames + config.pathfind_frames + config.stats_frames;
        Self { config, current_frame: 0, cycle_frames }
    }
}

pub struct ComputePlugin;

impl Plugin for ComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<cpu_sim::SimTiming>()
            .init_resource::<gpu_sim::GpuSimParams>()
            .init_resource::<gpu_sim::GpuSimTextures>()
            .init_resource::<SimScheduleState>()
            .add_plugins(gpu_sim::GpuSimPlugin)
            .add_plugins(gpu_pathfinding::GpuPathfindingPlugin)
            .add_systems(Startup, setup_data_textures)
            .add_systems(Update, (
                spawn::handle_spawn_requests,
                update_schedule_state,
                sync_gpu_textures_and_params,
                gpu_sim::clear_pending_spawns,
                gpu_sim::apply_gpu_readback,
                cpu_sim::cpu_sim_tick,
                upload_dirty_textures_system,
            ).chain().run_if(in_state(AppState::InGame)));
    }
}

fn update_schedule_state(mut state: ResMut<SimScheduleState>) {
    state.current_frame = (state.current_frame + 1) % state.cycle_frames;
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
    schedule: Res<SimScheduleState>,
) {
    if let Some(dt) = dt {
        gpu_tex.people = Some(dt.people.clone());
        gpu_tex.roads = Some(dt.roads.clone());
        gpu_tex.buildings = Some(dt.buildings.clone());
    }
    gpu_sim::update_gpu_sim_params(&time, &durations, &settings, &people, &buildings, &roads, pending, &mut gpu_params);
    
    // Slice logic
    gpu_params.cycle_frames = schedule.cycle_frames;
    gpu_params.b_count = 0;
    gpu_params.logic_count = 0;
    gpu_params.r_count = 0;
    
    let c = &schedule.config;
    let mut frame = schedule.current_frame;
    
    if frame < c.buildings_frames {
        let f = frame;
        let slice = (buildings.items.len() as u32 + c.buildings_frames - 1) / c.buildings_frames;
        gpu_params.b_start = f * slice;
        gpu_params.b_count = slice.min(buildings.items.len() as u32 - gpu_params.b_start.min(buildings.items.len() as u32));
    } else {
        frame -= c.buildings_frames;
        if frame < c.people_logic_frames {
            let f = frame;
            let slice = (people.len as u32 + c.people_logic_frames - 1) / c.people_logic_frames;
            gpu_params.logic_start = f * slice;
            gpu_params.logic_count = slice.min(people.len as u32 - gpu_params.logic_start.min(people.len as u32));
        } else {
            frame -= c.people_logic_frames;
            if frame < c.roads_frames {
                let f = frame;
                let slice = (roads.segments.len() as u32 + c.roads_frames - 1) / c.roads_frames;
                gpu_params.r_start = f * slice;
                gpu_params.r_count = slice.min(roads.segments.len() as u32 - gpu_params.r_start.min(roads.segments.len() as u32));
            } else {
                frame -= c.roads_frames;
                path_params.do_dispatch = 1;
                // Dispatch logic is handled in gpu_pathfinding.rs
            }
        }
    }
    
    if schedule.current_frame < c.buildings_frames + c.people_logic_frames + c.roads_frames {
        path_params.do_dispatch = 0;
    }

    if schedule.current_frame == 0 {
        gpu_params.reset_stats = 1;
    } else {
        gpu_params.reset_stats = 0;
    }

    if schedule.current_frame >= schedule.cycle_frames - c.stats_frames {
        gpu_params.do_readback = 1;
    } else {
        gpu_params.do_readback = 0;
    }

    if schedule.current_frame == c.buildings_frames {
        path_params.reset_path_queue = 1;
    } else {
        path_params.reset_path_queue = 0;
    }

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
