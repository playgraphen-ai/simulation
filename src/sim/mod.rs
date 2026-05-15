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
            .init_resource::<ActivityDurations>()
            .init_resource::<SimSettings>()
            .init_resource::<GameTime>()
            .init_resource::<crate::compute::spawn::PendingGpuSpawns>()
            .add_message::<SpawnPeopleRequest>()
            // `startup` resets the grid, then `build_starter_scenario` paints
            // the demo map on top of it — chain them so ordering is explicit.
            .add_systems(OnEnter(AppState::InGame), (startup, scenario::build_starter_scenario).chain());
        app.add_systems(Update, (
                counters::update_counters_system,
                crate::log_export::json_logger_system,
                update_game_time,
            ).run_if(in_state(AppState::InGame)));
    }
}

use noise::{NoiseFn, OpenSimplex};

pub fn startup(mut grid: ResMut<grid::CityGrid>) {
    let w = 1280;
    let h = 1280;
    grid.reset(w, h);
    
    let elev_noise = OpenSimplex::new(42);
    let moist_noise = OpenSimplex::new(1337);
    
    for y in 0..h {
        for x in 0..w {
            let i = grid.idx(x, y);
            let nx = x as f64 * 0.04;
            let ny = y as f64 * 0.04;
            
            // FBM for elevation
            let e = 1.0 * elev_noise.get([nx, ny]) 
                  + 0.5 * elev_noise.get([nx * 2.0, ny * 2.0])
                  + 0.25 * elev_noise.get([nx * 4.0, ny * 4.0]);
            let e = e / 1.75;
            
            // Moisture
            let m = moist_noise.get([nx * 0.8, ny * 0.8]);
            
            if e < -0.2 {
                grid.biomes[i] = grid::Biome::Water;
                grid.tiles[i] = grid::Tile::Water;
                grid.elevations[i] = -0.5; // Flat water
            } else {
                let height = ((e + 0.2) * 3.0) as f32; // scale hills
                grid.elevations[i] = height;
                
                if m < -0.2 {
                    grid.biomes[i] = grid::Biome::Desert;
                } else if m > 0.3 {
                    grid.biomes[i] = grid::Biome::Forest;
                } else {
                    grid.biomes[i] = grid::Biome::Plains;
                }
            }
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
    pub tax_income: f32,
    pub tax_rent: f32,
    pub tax_consumption: f32,
    pub collisions_enabled: f32, // 1.0 for true, 0.0 for false
}

impl Default for SimSettings {
    fn default() -> Self {
        let dur = ActivityDurations::default();
        Self {
            abandon_multiplier: 1.0,
            rent_cost: 20.0,
            work_salary: 50.0,
            shop_cost: 30.0,
            tax_income: dur.tax_income,
            tax_rent: dur.tax_rent,
            tax_consumption: dur.tax_consumption,
            collisions_enabled: 1.0,
        }
    }
}

/// How long each activity lasts on the CPU side, in seconds. Changes here are
/// not applied retroactively to in-progress activities — the compute shader
/// reads this resource only when a person transitions to a new activity.
#[derive(Resource, Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ActivityDurations {
    pub home: f32,
    pub work: f32,
    pub shop: f32,
    /// Probability 0..1 of going to work (vs. shopping) after home.
    pub home_to_work_prob: f32,
    pub tax_income: f32,
    pub tax_rent: f32,
    pub tax_consumption: f32,
}

impl Default for ActivityDurations {
    fn default() -> Self {
        // Try to load from assets/sim_schedule.json
        std::fs::read_to_string("assets/sim_schedule.json")
            .ok()
            .and_then(|s| serde_json::from_str::<ActivityDurations>(&s).ok())
            .unwrap_or(Self { 
                home: 150.0, 
                work: 225.0, 
                shop: 75.0, 
                home_to_work_prob: 0.6,
                tax_income: 0.15,
                tax_rent: 0.1,
                tax_consumption: 0.08,
            })
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
