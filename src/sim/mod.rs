//! Simulation data layer.
//!
//! Owns the authoritative CPU shadow of the three data textures that the
//! compute shaders mutate: people, road segments, buildings. The CPU shadow is
//! the staging area where UI tools write (zoning, road construction, spawns);
//! each frame the shadows are uploaded to the GPU storage textures, and
//! counters / summaries are read back for HUD display.

pub mod grid;
pub mod people;
pub mod roads;
pub mod buildings;
pub mod textures;
pub mod counters;
pub mod scenario;
pub mod constants;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use crate::AppState;

pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<grid::CityGrid>()
            .init_resource::<people::PeopleData>()
            .init_resource::<roads::RoadData>()
            .init_resource::<buildings::BuildingData>()
            .init_resource::<counters::SimCounters>()
            .init_resource::<SimConfig>()
            .init_resource::<SimSettings>()
            .init_resource::<GameTime>()
            .init_resource::<crate::compute::spawn::PendingGpuSpawns>()
            .add_message::<SpawnPeopleRequest>()
            // `startup` resets the grid, then `build_starter_scenario` paints
            // the demo map on top of it — chain them so ordering is explicit.
            .add_systems(OnEnter(AppState::InGame), (startup, scenario::build_starter_scenario).chain());
        app.add_systems(Update, (
                counters::update_counters_system,
                update_game_time,
            ).run_if(in_state(AppState::InGame)));
    }
}

use noise::{NoiseFn, OpenSimplex};
use rayon::prelude::*;

pub fn startup(mut grid: ResMut<grid::CityGrid>) {
    let w = 1280;
    let h = 1280;
    grid.reset(w, h);
    
    let elev_noise = OpenSimplex::new(42);
    let moist_noise = OpenSimplex::new(1337);
    
    let w_usize = w as usize;
    let grid_ref = &mut *grid;
    let biomes = &mut grid_ref.biomes;
    let tiles = &mut grid_ref.tiles;
    let elevations = &mut grid_ref.elevations;
    
    biomes.par_chunks_mut(w_usize)
        .zip(tiles.par_chunks_mut(w_usize))
        .zip(elevations.par_chunks_mut(w_usize))
        .enumerate()
        .for_each(|(y, ((biomes_row, tiles_row), elevations_row))| {
            let y_f = y as f64;
            let ny = y_f * 0.04;
            
            for (x, ((biome, tile), elevation)) in biomes_row.iter_mut()
                .zip(tiles_row.iter_mut())
                .zip(elevations_row.iter_mut())
                .enumerate() 
            {
                let x_f = x as f64;
                let nx = x_f * 0.04;
                
                // FBM for elevation
                let e = (elev_noise.get([nx, ny]) 
                      + 0.5 * elev_noise.get([nx * 2.0, ny * 2.0])
                      + 0.25 * elev_noise.get([nx * 4.0, ny * 4.0])) / 1.75;
                
                // Moisture
                let m = moist_noise.get([nx * 0.8, ny * 0.8]);
                
                if e < -0.2 {
                    *biome = grid::Biome::Water;
                    *tile = grid::Tile::Water;
                    *elevation = -0.5;
                } else {
                    *elevation = ((e + 0.2) * 3.0) as f32;
                    
                    if m < -0.2 {
                        *biome = grid::Biome::Desert;
                    } else if m > 0.3 {
                        *biome = grid::Biome::Forest;
                    } else {
                        *biome = grid::Biome::Plains;
                    }
                }
            }
        });
}

/// Unified simulation configuration loaded from assets/sim_schedule.json.
#[derive(Resource, Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SimConfig {
    // Scheduling parameters (previously in ScheduleConfig)
    pub gc_frames: u32,
    pub buildings_frames: u32,
    pub people_logic_frames: u32,
    pub roads_frames: u32,
    pub pathfind_frames: u32,
    pub stats_frames: u32,

    // Activity durations and economic parameters (previously in ActivityDurations)
    pub home: f32,
    pub work: f32,
    pub shop: f32,
    /// Probability 0..1 of going to work (vs. shopping) after home.
    pub home_to_work_prob: f32,
    pub tax_income: f32,
    pub tax_rent: f32,
    pub tax_consumption: f32,
}

impl SimConfig {
    pub fn load() -> Self {
        // Try to load from assets/sim_schedule.json
        std::fs::read_to_string("assets/sim_schedule.json")
            .ok()
            .and_then(|s| serde_json::from_str::<SimConfig>(&s).ok())
            .unwrap_or_default()
    }
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            gc_frames: 1,
            buildings_frames: 10,
            people_logic_frames: 10,
            roads_frames: 10,
            pathfind_frames: 59,
            stats_frames: 1,
            home: 150.0,
            work: 225.0,
            shop: 75.0,
            home_to_work_prob: 0.6,
            tax_income: 0.15,
            tax_rent: 0.1,
            tax_consumption: 0.08,
        }
    }
}

/// Global settings that the user can tweak via sliders.
#[derive(Resource, Clone, Copy, Debug)]
pub struct SimSettings {
    pub abandon_multiplier: f32, // <1.0 = slower decay, >1.0 = faster decay
    pub rent_cost: f32,
    pub work_salary: f32,
    pub shop_cost: f32,
    pub shop_food_gain: f32,
    pub tax_income: f32,
    pub tax_rent: f32,
    pub tax_consumption: f32,
    pub collisions_enabled: f32, // 1.0 for true, 0.0 for false
}

impl SimSettings {
    pub fn new(config: &SimConfig) -> Self {
        Self {
            abandon_multiplier: 1.0,
            rent_cost: 20.0,
            work_salary: 50.0,
            shop_cost: 10.0,
            shop_food_gain: 100.0,
            tax_income: config.tax_income,
            tax_rent: config.tax_rent,
            tax_consumption: config.tax_consumption,
            collisions_enabled: 1.0,
        }
    }
}

impl Default for SimSettings {
    fn default() -> Self {
        Self::new(&SimConfig::default())
    }
}

/// Tracks both real-world time and scaled simulation time.
#[derive(Resource, Default, Debug)]
pub struct GameTime {
    pub real_elapsed_secs: f32,
}

impl GameTime {
    /// Returns (days, hours, minutes) in simulation time.
    /// Scale: 5 real minutes = 1 simulation day.
    pub fn simulation_time(&self) -> (u32, u32, u32) {
        // 5 real mins = 1440 sim mins
        // 1 real sec = 4.8 sim mins
        let total_sim_mins = (self.real_elapsed_secs * 4.8) as u32;
        let days = total_sim_mins / (24 * 60);
        let hours = (total_sim_mins % (24 * 60)) / 60;
        let mins = total_sim_mins % 60;
        (days, hours, mins)
    }
}

fn update_game_time(time: Res<Time>, mut game_time: ResMut<GameTime>) {
    game_time.real_elapsed_secs += time.delta_secs();
}

/// UI-driven request to spawn N people at the map edge.
#[derive(Message, Debug, Clone, Copy)]
pub struct SpawnPeopleRequest {
    pub count: u32,
}
