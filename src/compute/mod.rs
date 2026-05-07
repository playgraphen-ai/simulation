//! Compute layer.
//!
//! GPU compute path: upload the three data textures, dispatch the
//! `sim_people.wgsl` and `pathfind.wgsl` shaders, read counters back.
//! The CPU fallback has been removed to free up cycles.

pub mod spawn;
pub mod gpu_sim;
pub mod gpu_pathfinding;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::sim::textures::{create_data_textures, DataTextures};
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
        app.init_resource::<gpu_sim::GpuSimParams>()
            .init_resource::<gpu_sim::GpuSimTextures>()
            .init_resource::<SimScheduleState>()
            .add_plugins(gpu_sim::GpuSimPlugin)
            .add_plugins(gpu_pathfinding::GpuPathfindingPlugin)
            .add_systems(Startup, setup_data_textures)
            .add_systems(Update, (
                spawn::handle_spawn_requests,
                update_schedule_state,
                sync_gpu_textures_and_params,
                gpu_sim::apply_gpu_readback,
                queue_texture_updates_system,
            ).chain().run_if(in_state(AppState::InGame)));
    }
}

fn queue_texture_updates_system(
    mut commands: Commands,
    mut people: ResMut<PeopleData>,
    mut roads: ResMut<RoadData>,
    mut buildings: ResMut<BuildingData>,
) {
    let mut ext = gpu_sim::ExtractedTextureUpdates::default();
    if people.dirty {
        let width = people.tex_width as usize;
        let elements = people.len as usize;
        if elements > 0 {
            let mut bytes = bytemuck::cast_slice(&people.rows[..elements]).to_vec();
            // Pad bytes to full width row to satisfy WGPU
            let row_bytes = width * 16;
            let remainder = bytes.len() % row_bytes;
            if remainder != 0 {
                bytes.extend(vec![0u8; row_bytes - remainder]);
            }
            ext.people = Some(bytes);
        }
        people.dirty = false;
    }
    if roads.dirty {
        let width = roads.tex_width as usize;
        let elements = roads.segments.len();
        if elements > 0 {
            let mut bytes = bytemuck::cast_slice(&roads.rows[..elements]).to_vec();
            let row_bytes = width * 16;
            let remainder = bytes.len() % row_bytes;
            if remainder != 0 {
                bytes.extend(vec![0u8; row_bytes - remainder]);
            }
            ext.roads = Some(bytes);
        }
        
        let mut pts_bytes: Vec<u8> = bytemuck::cast_slice(
            &roads.all_points.iter().flat_map(|&(x, y)| vec![x as f32, y as f32]).collect::<Vec<f32>>()
        ).to_vec();
        if !pts_bytes.is_empty() {
            let row_bytes = 1024 * 8; // Rg32Float = 8 bytes, width = 1024
            let remainder = pts_bytes.len() % row_bytes;
            if remainder != 0 {
                pts_bytes.extend(vec![0u8; row_bytes - remainder]);
            }
            ext.road_points = Some(bytemuck::cast_slice(&pts_bytes).to_vec()); // Store as f32 internally in resource
        }
        
        roads.dirty = false;
    }
    if buildings.dirty {
        let width = buildings.tex_width as usize;
        let elements = buildings.items.len();
        if elements > 0 {
            let mut bytes = bytemuck::cast_slice(&buildings.rows[..elements]).to_vec();
            let row_bytes = width * 16;
            let remainder = bytes.len() % row_bytes;
            if remainder != 0 {
                bytes.extend(vec![0u8; row_bytes - remainder]);
            }
            ext.buildings = Some(bytes);
        }
        buildings.dirty = false;
    }
    commands.insert_resource(ext);
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
    grid: Res<crate::sim::grid::CityGrid>,
) {
    let dt = create_data_textures(&mut images, &people, &roads, &buildings, grid.width, grid.height);
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
    grid: Res<crate::sim::grid::CityGrid>,
    mut pending: ResMut<spawn::PendingGpuSpawns>,
    mut gpu_params: ResMut<gpu_sim::GpuSimParams>,
    mut path_params: ResMut<gpu_pathfinding::PathParams>,
    schedule: Res<SimScheduleState>,
    scenario: Option<Res<crate::sim::scenario::ScenarioLayout>>,
) {
    if let Some(dt) = dt {
        gpu_tex.people = Some(dt.people.clone());
        gpu_tex.roads = Some(dt.roads.clone());
        gpu_tex.buildings = Some(dt.buildings.clone());
        gpu_tex.road_points = Some(dt.road_points.clone());
    }
    let entry_seg = scenario.map(|s| s.entry_seg).unwrap_or(0);
    gpu_sim::update_gpu_sim_params(&time, &durations, &settings, &people, &buildings, &roads, &grid, &mut gpu_params, entry_seg);
    
    let frame = schedule.current_frame;

    if frame == 0 {
        path_params.reset_path_queue = 1;
        // Also flush pending spawns on the first frame of the cycle
        if pending.count > 0 {
            let spawn_this_cycle = pending.count.min(1000);
            gpu_params.spawn_count = spawn_this_cycle;
            gpu_params.spawn_start_index = people.len.saturating_sub(pending.count);
            pending.count -= spawn_this_cycle;
        } else {
            gpu_params.spawn_count = 0;
            gpu_params.spawn_start_index = 0;
        }
    } else {
        path_params.reset_path_queue = 0;
        gpu_params.spawn_count = 0;
        gpu_params.spawn_start_index = 0;
    }

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
                if frame < c.pathfind_frames {
                    path_params.do_dispatch = 1;
                } else {
                    path_params.do_dispatch = 0;
                }
            }
        }
    }

    if schedule.current_frame == 0 {
        gpu_params.reset_stats = 1;
    } else {
        gpu_params.reset_stats = 0;
    }

    if schedule.current_frame >= schedule.cycle_frames - c.stats_frames {
        gpu_params.do_stats_readback = 1;
    } else {
        gpu_params.do_stats_readback = 0;
    }

    path_params.roads_tex_w = roads.tex_width;
    path_params.segments_count = roads.segments.len() as u32;
    path_params.max_path_len = 256;
}

