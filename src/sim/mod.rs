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

use bevy::prelude::*;
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
            .add_message::<SpawnPeopleRequest>()
            // `startup` resets the grid, then `build_starter_scenario` paints
            // the demo map on top of it — chain them so ordering is explicit.
            .add_systems(OnEnter(AppState::InGame), (startup, scenario::build_starter_scenario).chain());
        app.add_systems(Update, (
                counters::update_counters_system,
                crate::log_export::json_logger_system,
            ).run_if(in_state(AppState::InGame)));
    }
}

use noise::{NoiseFn, OpenSimplex};

pub fn startup(mut grid: ResMut<grid::CityGrid>) {
    let w = 128;
    let h = 128;
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
}

impl Default for SimSettings {
    fn default() -> Self {
        Self {
            abandon_multiplier: 1.0,
            rent_cost: 20.0,
            work_salary: 50.0,
            shop_cost: 30.0,
        }
    }
}

/// How long each activity lasts on the CPU side, in seconds. Changes here are
/// not applied retroactively to in-progress activities — the compute shader
/// reads this resource only when a person transitions to a new activity.
#[derive(Resource, Clone, Copy, Debug)]
pub struct ActivityDurations {
    pub home: f32,
    pub work: f32,
    pub shop: f32,
    /// Probability 0..1 of going to work (vs. shopping) after home.
    pub home_to_work_prob: f32,
}

impl Default for ActivityDurations {
    fn default() -> Self {
        Self { home: 30.0, work: 45.0, shop: 15.0, home_to_work_prob: 0.6 }
    }
}

/// UI-driven request to spawn N people at the map edge.
#[derive(Message, Debug, Clone, Copy)]
pub struct SpawnPeopleRequest {
    pub count: u32,
}
